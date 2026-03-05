"""
Baseline Model - LightGBM
简单的特征工程 + LightGBM 分类器
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# 读取数据
print("读取数据...")
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample = pd.read_csv('data/sample_submission.csv')

# 保存 id
train_id = train['id']
test_id = test['id']

# 目标变量编码
y = (train['Churn'] == 'Yes').astype(int)

# 删除 id 和目标变量
train = train.drop(['id', 'Churn'], axis=1)
test = test.drop(['id'], axis=1)

print(f"训练集形状: {train.shape}")
print(f"测试集形状: {test.shape}")

# 类别特征编码
print("\n特征工程...")
cat_features = train.select_dtypes(include=['object']).columns.tolist()
print(f"类别特征: {len(cat_features)} 个")

for col in cat_features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    train, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")

# LightGBM 模型
print("\n训练模型...")
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

# 验证集评估
y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
val_auc = roc_auc_score(y_val, y_val_pred)
print(f"\n验证集 AUC: {val_auc:.6f}")

# 预测测试集
print("\n预测测试集...")
y_test_pred = model.predict(test, num_iteration=model.best_iteration)

# 生成提交文件
submission = pd.DataFrame({
    'id': test_id,
    'Churn': y_test_pred
})

submission.to_csv('submissions/baseline_lgb.csv', index=False)
print(f"\n提交文件已保存: submissions/baseline_lgb.csv")
print(f"预测概率范围: [{y_test_pred.min():.4f}, {y_test_pred.max():.4f}]")
