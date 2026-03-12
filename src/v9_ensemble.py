"""
V9: LightGBM + XGBoost 双模型融合 + 服务统计特征
- 服务统计特征 (service_count, has_internet, avg_charge_per_service)
- 数值变换 (Frequency, Log1p, Sqrt, Rank)
- Target Encoding
- 双模型加权融合

预期提升: +0.0005-0.0007
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import warnings
import os
import sys
from contextlib import contextmanager

warnings.filterwarnings("ignore")


@contextmanager
def suppress_gpu_warnings():
    devnull = open(os.devnull, "w")
    old_stderr = os.dup(2)
    os.dup2(devnull.fileno(), 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        devnull.close()


print("V9: LightGBM + XGBoost Ensemble + Service Features")

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X = train.drop(["id", "Churn"], axis=1)
y = train["Churn"].map({"Yes": 1, "No": 0})
X_test = test.drop(["id"], axis=1)

print(f"训练集: {X.shape}, 测试集: {X_test.shape}")
print(f"流失率: {y.mean():.4f}")

CAT_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

df = pd.concat([X, X_test], axis=0, ignore_index=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("特征工程")
print("\n[1] 服务统计特征...")
df["service_count"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
df["has_internet"] = (df["InternetService"] != "No").astype(int)
df["has_phone"] = (df["PhoneService"] == "Yes").astype(int)
df["avg_charge_per_service"] = df["MonthlyCharges"] / (df["service_count"] + 1)
df["total_services"] = df["service_count"] + df["has_internet"] + df["has_phone"]
print(f"  service_count: mean={df['service_count'].mean():.2f}")
print(f"  has_internet: {df['has_internet'].mean():.2%}")
print(f"  avg_charge_per_service: mean={df['avg_charge_per_service'].mean():.2f}")

print("\n[2] 数值变换...")
for col in NUM_COLS:
    freq = df[col].value_counts(normalize=True)
    df[f"FREQ_{col}"] = df[col].map(freq).fillna(0)

for col in NUM_COLS:
    df[f"LOG1P_{col}"] = np.log1p(df[col])

for col in NUM_COLS:
    df[f"SQRT_{col}"] = np.sqrt(df[col])

for col in NUM_COLS:
    df[f"RANK_{col}"] = df[col].rank(pct=True)

print(f"  数值变换特征: {len(NUM_COLS) * 4} 个")

print("\n[3] 交叉特征...")
df["Contract_InternetService"] = df["Contract"] + "_" + df["InternetService"]
df["tenure_MonthlyCharges"] = df["tenure"] * df["MonthlyCharges"]
df["SeniorCitizen_TechSupport"] = (
    df["SeniorCitizen"].astype(str) + "_" + df["TechSupport"]
)
df["Payment_Contract"] = df["PaymentMethod"] + "_" + df["Contract"]
df["Internet_Security"] = df["InternetService"] + "_" + df["OnlineSecurity"]
df["tenure_service"] = df["tenure"] * df["service_count"]
df["charge_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)

CROSS_COLS = [
    "Contract_InternetService",
    "SeniorCitizen_TechSupport",
    "Payment_Contract",
    "Internet_Security",
]

X_transformed = df.iloc[: len(X)].copy()
X_test_transformed = df.iloc[len(X) :].copy()
print(f"\n变换后特征数: {X_transformed.shape[1]}")


def target_encode_cv(X_train, y_train, X_test, cat_cols, n_splits=5, smoothing=5):
    """CV 内 Target Encoding，避免 leakage"""
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    global_mean = y_train.mean()

    for col in cat_cols:
        oof_encoding = np.zeros(len(X_train))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]

            encoding_map = {}
            for cat in X_train[col].unique():
                mask = X_tr[col] == cat
                count = mask.sum()
                mean = y_tr[mask].mean() if count > 0 else global_mean
                encoding_map[cat] = (count * mean + smoothing * global_mean) / (
                    count + smoothing
                )

            oof_encoding[val_idx] = (
                X_train.iloc[val_idx][col].map(encoding_map).fillna(global_mean)
            )

        X_train_encoded[f"TE_{col}"] = oof_encoding

        encoding_map = {}
        for cat in X_train[col].unique():
            mask = X_train[col] == cat
            count = mask.sum()
            mean = y_train[mask].mean() if count > 0 else global_mean
            encoding_map[cat] = (count * mean + smoothing * global_mean) / (
                count + smoothing
            )

        X_test_encoded[f"TE_{col}"] = X_test[col].map(encoding_map).fillna(global_mean)

    X_train_encoded = X_train_encoded.drop(cat_cols, axis=1)
    X_test_encoded = X_test_encoded.drop(cat_cols, axis=1)

    return X_train_encoded, X_test_encoded


print("模型训练")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X_transformed))
oof_lgb = np.zeros(len(X_transformed))
test_xgb = np.zeros(len(X_test_transformed))
test_lgb = np.zeros(len(X_test_transformed))

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.01,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "tree_method": "hist",
    "device": "cuda",
    "n_jobs": -1,
}

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.01,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "device": "gpu",
    "verbose": -1,
}

ALL_CAT_COLS = CAT_COLS + CROSS_COLS

for fold, (train_idx, val_idx) in enumerate(skf.split(X_transformed, y), 1):
    print(f"Fold {fold}/5")

    X_tr, X_val = X_transformed.iloc[train_idx], X_transformed.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_tr_enc, X_val_enc = target_encode_cv(
        X_tr, y_tr, X_val, ALL_CAT_COLS, n_splits=5, smoothing=5
    )

    print(f"编码后特征数: {X_tr_enc.shape[1]}")

    print("\n[XGBoost] 训练中...")
    dtrain = xgb.DMatrix(X_tr_enc, label=y_tr)
    dval = xgb.DMatrix(X_val_enc, label=y_val)

    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=10000,
        evals=[(dval, "val")],
        early_stopping_rounds=200,
        verbose_eval=False,
    )

    oof_xgb[val_idx] = xgb_model.predict(dval)
    xgb_auc = roc_auc_score(y_val, oof_xgb[val_idx])
    print(f"XGBoost Fold {fold} AUC: {xgb_auc:.6f} (iter: {xgb_model.best_iteration})")

    print("\n[LightGBM] 训练中...")
    lgb_train = lgb.Dataset(X_tr_enc, label=y_tr)
    lgb_val = lgb.Dataset(X_val_enc, label=y_val, reference=lgb_train)

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=10000,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False),
        ],
    )

    oof_lgb[val_idx] = lgb_model.predict(X_val_enc)
    lgb_auc = roc_auc_score(y_val, oof_lgb[val_idx])
    print(f"LightGBM Fold {fold} AUC: {lgb_auc:.6f} (iter: {lgb_model.best_iteration})")

    _, X_test_enc = target_encode_cv(
        X_tr, y_tr, X_test_transformed, ALL_CAT_COLS, n_splits=5, smoothing=5
    )

    dtest = xgb.DMatrix(X_test_enc)
    test_xgb += xgb_model.predict(dtest) / 5
    test_lgb += lgb_model.predict(X_test_enc) / 5

print("模型评估")
xgb_oof_auc = roc_auc_score(y, oof_xgb)
lgb_oof_auc = roc_auc_score(y, oof_lgb)

print(f"\nXGBoost  OOF AUC: {xgb_oof_auc:.6f}")
print(f"LightGBM OOF AUC: {lgb_oof_auc:.6f}")

print("\n寻找最优融合权重...")
best_auc = 0
best_weight = 0.5
for w in np.arange(0.3, 0.8, 0.05):
    ensemble_oof = w * oof_xgb + (1 - w) * oof_lgb
    auc = roc_auc_score(y, ensemble_oof)
    if auc > best_auc:
        best_auc = auc
        best_weight = w
    print(f"  XGB权重={w:.2f}: AUC={auc:.6f}")

print(f"\n最优权重: XGB={best_weight:.2f}, LGB={1-best_weight:.2f}")
print(f"融合后 OOF AUC: {best_auc:.6f}")

test_preds = best_weight * test_xgb + (1 - best_weight) * test_lgb

print("最终结果")
print(f"XGBoost  OOF AUC: {xgb_oof_auc:.6f}")
print(f"LightGBM OOF AUC: {lgb_oof_auc:.6f}")
print(f"融合后   OOF AUC: {best_auc:.6f}")
print(f"提升: {best_auc - max(xgb_oof_auc, lgb_oof_auc):+.6f}")

submission = pd.DataFrame({"id": test["id"], "Churn": test_preds})
submission.to_csv("submissions/v9_ensemble.csv", index=False)
print(f"\n提交文件已保存: submissions/v9_ensemble.csv")
print(
    f"预测分布: min={test_preds.min():.4f}, max={test_preds.max():.4f}, mean={test_preds.mean():.4f}"
)

# ps:不需要每次重新计算特征，可以用云服务器训练。
