#!/usr/bin/env python3
"""
V8: Numerical Transformations
- Frequency Encoding (稀有度)
- Log1p 变换 (处理偏态)
- Sqrt 变换 (压缩极值)
- Rank 变换 (百分位，鲁棒性强)

预期提升: +0.0002-0.0004
理论依据: Feature Engineering for ML (Alice Zheng)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 加载数据
# ============================================================
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

X = train.drop(['id', 'Churn'], axis=1)
y = train['Churn'].map({'Yes': 1, 'No': 0})
X_test = test.drop(['id'], axis=1)

print(f"训练集: {X.shape}, 测试集: {X_test.shape}")
print(f"流失率: {y.mean():.4f}")

# ============================================================
# 2. 特征工程
# ============================================================

# 类别特征
CAT_COLS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod']

# 数值特征
NUM_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

# 合并训练集和测试集用于特征工程
df = pd.concat([X, X_test], axis=0, ignore_index=True)

# ============================================================
# 2.1 数值变换 (新增)
# ============================================================

# 2.1.1 Frequency Encoding (稀有度)
for col in NUM_COLS:
    freq = df[col].value_counts(normalize=True)
    df[f'FREQ_{col}'] = df[col].map(freq).fillna(0)
    print(f"FREQ_{col}: {df[f'FREQ_{col}'].nunique()} unique values")

# 2.1.2 Log1p 变换 (处理偏态分布)
for col in NUM_COLS:
    df[f'LOG1P_{col}'] = np.log1p(df[col])
    print(f"LOG1P_{col}: mean={df[f'LOG1P_{col}'].mean():.2f}, std={df[f'LOG1P_{col}'].std():.2f}")

# 2.1.3 Sqrt 变换 (压缩极值)
for col in NUM_COLS:
    df[f'SQRT_{col}'] = np.sqrt(df[col])
    print(f"SQRT_{col}: mean={df[f'SQRT_{col}'].mean():.2f}, std={df[f'SQRT_{col}'].std():.2f}")

# 2.1.4 Rank 变换 (百分位，鲁棒性强)
for col in NUM_COLS:
    df[f'RANK_{col}'] = df[col].rank(pct=True)
    print(f"RANK_{col}: min={df[f'RANK_{col}'].min():.4f}, max={df[f'RANK_{col}'].max():.4f}")

# ============================================================
# 2.2 核心交叉特征 (保留 V2 的)
# ============================================================
df['Contract_InternetService'] = df['Contract'] + '_' + df['InternetService']
df['tenure_MonthlyCharges'] = df['tenure'] * df['MonthlyCharges']
df['SeniorCitizen_TechSupport'] = df['SeniorCitizen'].astype(str) + '_' + df['TechSupport']

# 将交叉特征也加入类别特征列表
CROSS_COLS = ['Contract_InternetService', 'SeniorCitizen_TechSupport']

# ============================================================
# 2.3 分离训练集和测试集
# ============================================================
X_transformed = df.iloc[:len(X)].copy()
X_test_transformed = df.iloc[len(X):].copy()

print(f"\n变换后特征数: {X_transformed.shape[1]}")

# ============================================================
# 3. Target Encoding (CV 内计算避免 leakage)
# ============================================================
def target_encode_cv(X_train, y_train, X_test, cat_cols, n_splits=5, smoothing=5):
    """CV 内 Target Encoding，避免 leakage"""
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    # 全局均值
    global_mean = y_train.mean()
    
    # 对每个类别特征
    for col in cat_cols:
        # 训练集：用 CV 内的 OOF 编码
        oof_encoding = np.zeros(len(X_train))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            
            # 计算训练折的编码
            encoding_map = {}
            for cat in X_train[col].unique():
                mask = X_tr[col] == cat
                count = mask.sum()
                mean = y_tr[mask].mean() if count > 0 else global_mean
                # 平滑
                encoding_map[cat] = (count * mean + smoothing * global_mean) / (count + smoothing)
            
            # 应用到验证折
            oof_encoding[val_idx] = X_train.iloc[val_idx][col].map(encoding_map).fillna(global_mean)
        
        X_train_encoded[f'TE_{col}'] = oof_encoding
        
        # 测试集：用全部训练数据的编码
        encoding_map = {}
        for cat in X_train[col].unique():
            mask = X_train[col] == cat
            count = mask.sum()
            mean = y_train[mask].mean() if count > 0 else global_mean
            encoding_map[cat] = (count * mean + smoothing * global_mean) / (count + smoothing)
        
        X_test_encoded[f'TE_{col}'] = X_test[col].map(encoding_map).fillna(global_mean)
    
    # 删除原始类别特征
    X_train_encoded = X_train_encoded.drop(cat_cols, axis=1)
    X_test_encoded = X_test_encoded.drop(cat_cols, axis=1)
    
    return X_train_encoded, X_test_encoded

# ============================================================
# 4. 5-Fold CV + XGBoost
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_transformed))
test_preds = np.zeros(len(X_test_transformed))

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.01,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_transformed, y), 1):
    print(f"\n{'='*60}")
    print(f"Fold {fold}")
    print(f"{'='*60}")
    
    # 分割数据
    X_tr, X_val = X_transformed.iloc[train_idx], X_transformed.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Target Encoding (CV 内计算)
    X_tr_encoded, X_val_encoded = target_encode_cv(
        X_tr, y_tr, X_val, CAT_COLS + CROSS_COLS, n_splits=5, smoothing=5
    )
    
    print(f"编码后特征数: {X_tr_encoded.shape[1]}")
    
    # 训练
    dtrain = xgb.DMatrix(X_tr_encoded, label=y_tr)
    dval = xgb.DMatrix(X_val_encoded, label=y_val)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=200,
        verbose_eval=100
    )
    
    # 预测验证集
    oof_preds[val_idx] = model.predict(dval)
    val_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold} AUC: {val_auc:.5f}")
    
    # 预测测试集（累加）
    # 测试集也需要 Target Encoding
    _, X_test_encoded = target_encode_cv(
        X_tr, y_tr, X_test_transformed, CAT_COLS + CROSS_COLS, n_splits=5, smoothing=5
    )
    dtest = xgb.DMatrix(X_test_encoded)
    test_preds += model.predict(dtest) / 5

# ============================================================
# 5. 整体 OOF AUC
# ============================================================
oof_auc = roc_auc_score(y, oof_preds)
print(f"\n{'='*60}")
print(f"Overall OOF AUC: {oof_auc:.5f}")
print(f"{'='*60}")

# ============================================================
# 6. 生成提交文件
# ============================================================
submission = pd.DataFrame({
    'id': test['id'],
    'Churn': test_preds
})
submission.to_csv('submissions/v8_numerical_transforms.csv', index=False)
print(f"\n提交文件已保存: submissions/v8_numerical_transforms.csv")
print(f"预测分布: min={test_preds.min():.4f}, max={test_preds.max():.4f}, mean={test_preds.mean():.4f}")
