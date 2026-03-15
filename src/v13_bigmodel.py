import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import warnings
import math
import os
import pickle
import hashlib
import glob

warnings.filterwarnings("ignore")

XGB_LR_START = 0.1
XGB_LR_END = 0.005
XGB_LR_DECAY_ITER = 1000

LGB_LR_START = 0.03
LGB_LR_END = 0.005
LGB_LR_DECAY_ITER = 1000

NUM_BOOST_ROUND = 3000
EARLY_STOPPING_ROUNDS = 200


def get_learning_rate(current_iter, lr_start, lr_end, lr_decay_iter, mode="cosine"):
    progress = min(current_iter / lr_decay_iter, 1.0)
    if mode == "cosine":
        lr = lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * progress))
    elif mode == "linear":
        lr = lr_start - (lr_start - lr_end) * progress
    elif mode == "exp":
        lr = lr_start * (lr_end / lr_start) ** progress
    else:
        lr = lr_start
    return lr


def target_encode_cv(X_train, y_train, X_test, cat_cols, n_splits=6, smoothing=5):
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


def precompute_fold_encodings(X, y, X_test, cat_cols, n_splits=6, smoothing=5):
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


class XgbLearningRateCallback(xgb.callback.TrainingCallback):
    def __init__(
        self,
        lr_start=XGB_LR_START,
        lr_end=XGB_LR_END,
        lr_decay_iter=XGB_LR_DECAY_ITER,
        mode="cosine",
    ):
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.lr_decay_iter = lr_decay_iter
        self.mode = mode

    def after_iteration(self, model, epoch, evals_log):
        new_lr = get_learning_rate(
            epoch, self.lr_start, self.lr_end, self.lr_decay_iter, self.mode
        )
        model.set_param("learning_rate", new_lr)
        return False


def lgb_learning_rate_callback(env):
    new_lr = get_learning_rate(
        env.iteration, LGB_LR_START, LGB_LR_END, LGB_LR_DECAY_ITER, mode="cosine"
    )
    env.model.reset_parameter({"learning_rate": new_lr})


print("V13: LightGBM + XGBoost Ensemble (双重信号理论优化)")

print("\n加载预处理特征...")
train = pd.read_csv("data/train_features_v2.csv")
test = pd.read_csv("data/test_features_v2.csv")

y = train["Churn"]
test_ids = test["id"]
X = train.drop(["id", "Churn"], axis=1)
X_test = test.drop(["id"], axis=1)

print(f"训练集: {X.shape}, 测试集: {X_test.shape}")
print(f"流失率: {y.mean():.4f}")
print(f"\n特征列表 ({len(X.columns)} 个):")
print(f"  {list(X.columns[:10])} ... (前10个)")

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

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": XGB_LR_START,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.7,
    "colsample_bytree": 0.5,
    "colsample_bylevel": 0.6,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "gamma": 0.5,
    "random_state": 42,
    "tree_method": "hist",
    "device": "cuda",
    "n_jobs": -1,
    "scale_pos_weight": 3.0,
}

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": LGB_LR_START,
    "max_depth": 12,
    "num_leaves": 127,
    "min_child_samples": 10,
    "subsample": 0.9,
    "colsample_bytree": 0.3,
    "reg_alpha": 0.01,
    "reg_lambda": 0.1,
    "min_split_gain": 0,
    "random_state": 123,
    "n_jobs": -1,
    "device": "cpu",
    "verbose": -1,
    "scale_pos_weight": 4.0,
}

print("\n参数配置:")
print(
    f"  XGBoost: depth={xgb_params['max_depth']}, colsample={xgb_params['colsample_bytree']}, "
    f"alpha={xgb_params['reg_alpha']}, lambda={xgb_params['reg_lambda']}"
)
print(
    f"  LightGBM: depth={lgb_params['max_depth']}, leaves={lgb_params['num_leaves']}, "
    f"colsample={lgb_params['colsample_bytree']}"
)

print("\n预计算Target Encoding...")
encodings = precompute_fold_encodings(
    X, y, X_test, ALL_CAT_COLS, n_splits=6, smoothing=5
)

oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
test_xgb = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))

for fold in range(1, 7):
    print(f"\nFold {fold}")

    fold_data = encodings["fold_encodings"][fold]
    train_idx = fold_data["train_idx"]
    val_idx = fold_data["val_idx"]
    X_tr_enc = fold_data["X_tr_enc"]
    X_val_enc = fold_data["X_val_enc"]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    print("  [XGBoost] 训练中...")
    dtrain = xgb.DMatrix(X_tr_enc, label=y_tr)
    dval = xgb.DMatrix(X_val_enc, label=y_val)

    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=300,
        custom_metric=lambda pred, dtrain: (
            "auc",
            roc_auc_score(dtrain.get_label(), pred),
        ),
        callbacks=[XgbLearningRateCallback()],
    )

    oof_xgb[val_idx] = xgb_model.predict(dval)
    print(
        f"  [XGBoost] Fold {fold} AUC: {roc_auc_score(y_val, oof_xgb[val_idx]):.6f} (iter: {xgb_model.best_iteration})"
    )

    print("  [LightGBM] 训练中...")
    lgb_train = lgb.Dataset(X_tr_enc, label=y_tr)
    lgb_val = lgb.Dataset(X_val_enc, label=y_val, reference=lgb_train)

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=300),
            lgb_learning_rate_callback,
        ],
    )

    oof_lgb[val_idx] = lgb_model.predict(X_val_enc)
    print(
        f"  [LightGBM] Fold {fold} AUC: {roc_auc_score(y_val, oof_lgb[val_idx]):.6f} (iter: {lgb_model.best_iteration})"
    )

    X_test_enc = encodings["X_test_enc"]
    dtest = xgb.DMatrix(X_test_enc)
    test_xgb += xgb_model.predict(dtest) / 6
    test_lgb += lgb_model.predict(X_test_enc) / 6

print("\n模型评估")
xgb_oof_auc = roc_auc_score(y, oof_xgb)
lgb_oof_auc = roc_auc_score(y, oof_lgb)
print(f"XGBoost  OOF AUC: {xgb_oof_auc:.6f}")
print(f"LightGBM OOF AUC: {lgb_oof_auc:.6f}")

correlation = np.corrcoef(oof_xgb, oof_lgb)[0, 1]
print(f"\n模型诊断:")
print(f"  XGB与LGB OOF相关性: {correlation:.4f}")
if correlation > 0.9:
    print("  警告: 相关性过高(>0.9), 两模型趋同, 融合价值有限")
elif correlation < 0.7:
    print("  警告: 相关性过低(<0.7), 可能存在模型不稳定")
else:
    print("  理想状态: 相关性在0.7-0.85范围, 模型互补性好")

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
submission = pd.DataFrame({"id": test_ids, "Churn": test_preds})
submission.to_csv("submissions/v13_ensemble.csv", index=False)
print(f"\n提交文件已保存: submissions/v13_ensemble.csv")
print(
    f"预测分布: min={test_preds.min():.5f}, max={test_preds.max():.5f}, mean={test_preds.mean():.5f}"
)

for cache_file in glob.glob("cache/te_encodings*.pkl"):
    os.remove(cache_file)
    print(f"\n已删除缓存文件: {cache_file}")
