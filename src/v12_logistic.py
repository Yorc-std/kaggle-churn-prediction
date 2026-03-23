import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
import os
import pickle
import hashlib
import glob

warnings.filterwarnings("ignore")


def precompute_fold_encodings(X, y, X_test, cat_cols, n_splits=10, smoothing=5):
    for old_cache in glob.glob("cache/te_encodings*.pkl"):
        os.remove(old_cache)
        print(f"已删除旧缓存: {old_cache}")

    cols_hash = hashlib.md5(str(sorted(cat_cols)).encode()).hexdigest()[:8]
    cache_path = f"cache/te_encodings_{cols_hash}.pkl"

    print("计算Target Encoding...")
    os.makedirs("cache", exist_ok=True)
    global_mean = y.mean()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_encodings = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        X_tr_enc, X_val_enc = X_tr.copy(), X_val.copy()

        for col in cat_cols:
            encoding_map = {}
            for cat in X[col].unique():
                mask = X_tr[col] == cat
                count = mask.sum()
                mean = y_tr[mask].mean() if count > 0 else global_mean
                encoding_map[cat] = (count * mean + smoothing * global_mean) / (
                    count + smoothing
                )
            X_tr_enc[f"TE_{col}"] = X_tr[col].map(encoding_map).fillna(global_mean)
            X_val_enc[f"TE_{col}"] = X_val[col].map(encoding_map).fillna(global_mean)

        fold_encodings[fold] = {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "X_tr_enc": X_tr_enc.drop(cat_cols, axis=1),
            "X_val_enc": X_val_enc.drop(cat_cols, axis=1),
        }

    X_test_enc = X_test.copy()
    for col in cat_cols:
        encoding_map = {}
        for cat in X[col].unique():
            mask = X[col] == cat
            count = mask.sum()
            mean = y[mask].mean() if count > 0 else global_mean
            encoding_map[cat] = (count * mean + smoothing * global_mean) / (
                count + smoothing
            )
        X_test_enc[f"TE_{col}"] = X_test[col].map(encoding_map).fillna(global_mean)

    result = {
        "fold_encodings": fold_encodings,
        "X_test_enc": X_test_enc.drop(cat_cols, axis=1),
    }
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"已缓存: {cache_path}")
    return result


print("V12: Logistic Regression")

print("\n加载预处理特征...")
train = pd.read_csv("data/train_features.csv")
test = pd.read_csv("data/test_features.csv")

y = train["Churn"]
test_ids = test["id"]
X = train.drop(["id", "Churn"], axis=1)
X_test = test.drop(["id"], axis=1)

print(f"训练集: {X.shape}, 测试集: {X_test.shape}")
print(f"流失率: {y.mean():.4f}")

ALL_CAT_COLS = [
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

print("\n预计算Target Encoding...")
encodings = precompute_fold_encodings(
    X, y, X_test, ALL_CAT_COLS, n_splits=10, smoothing=5
)

oof_lr = np.zeros(len(X))
test_lr = np.zeros(len(X_test))

for fold in range(1, 11):
    print(f"\nFold {fold}")

    fold_data = encodings["fold_encodings"][fold]
    train_idx = fold_data["train_idx"]
    val_idx = fold_data["val_idx"]
    X_tr_enc = fold_data["X_tr_enc"]
    X_val_enc = fold_data["X_val_enc"]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_enc)
    X_val_scaled = scaler.transform(X_val_enc)

    print("  [LogisticRegression] 训练中...")
    lr_model = LogisticRegression(
        C=1.0,
        max_iter=3000,
        random_state=42,
        class_weight="balanced",
        solver="saga",  # 大数据集推荐
        n_jobs=-1,  # 并行加速
    )
    lr_model.fit(X_tr_scaled, y_tr)

    oof_lr[val_idx] = lr_model.predict_proba(X_val_scaled)[:, 1]
    print(
        f"  [LogisticRegression] Fold {fold} AUC: {roc_auc_score(y_val, oof_lr[val_idx]):.6f}"
    )

    X_test_enc = encodings["X_test_enc"]
    X_test_scaled = scaler.transform(X_test_enc)
    test_lr += lr_model.predict_proba(X_test_scaled)[:, 1] / 10

print("\n模型评估")
lr_oof_auc = roc_auc_score(y, oof_lr)
print(f"LogisticRegression OOF AUC: {lr_oof_auc:.6f}")

submission = pd.DataFrame({"id": test_ids, "Churn": test_lr})
submission.to_csv("submissions/v12_logistic.csv", index=False)
print(f"\n提交文件已保存: submissions/v12_logistic.csv")
print(
    f"预测分布: min={test_lr.min():.5f}, max={test_lr.max():.5f}, mean={test_lr.mean():.5f}"
)

for cache_file in glob.glob("cache/te_encodings*.pkl"):
    os.remove(cache_file)
    print(f"\n已删除缓存文件: {cache_file}")
