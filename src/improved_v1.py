"""
Improved Model V1 - Target Encoding + Feature Engineering + 5-Fold CV
改进点:
1. Target Encoding 替代 LabelEncoder (避免 Leakage)
2. 交叉特征 (业务逻辑驱动)
3. 5-Fold Stratified CV (更稳定的评估)
4. 数值特征处理 (异常值 + 离散化)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 读取数据
# ============================================================
print("=" * 60)
print("读取数据...")
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 保存 id
train_id = train['id']
test_id = test['id']

# 目标变量编码
y = (train['Churn'] == 'Yes').astype(int)

# 删除 id 和目标变量
train = train.drop(['id', 'Churn'], axis=1)
test = test.drop(['id'], axis=1)

print(f"训练集: {train.shape}, 流失率: {y.mean():.2%}")
print(f"测试集: {test.shape}")

# ============================================================
# 2. 特征工程
# ============================================================
print("\n" + "=" * 60)
print("特征工程...")

# 2.1 识别类别特征和数值特征
cat_features = train.select_dtypes(include=['object']).columns.tolist()
num_features = train.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"类别特征: {len(cat_features)} 个")
print(f"数值特征: {len(num_features)} 个")

# 2.2 交叉特征 (业务逻辑驱动)
print("\n创建交叉特征...")

def create_interaction_features(df):
    """创建交叉特征"""
    df = df.copy()
    
    # Contract × InternetService (长期合约 + 光纤用户)
    df['Contract_Internet'] = df['Contract'].astype(str) + '_' + df['InternetService'].astype(str)
    
    # tenure × MonthlyCharges (新用户 + 高消费)
    df['tenure_MonthlyCharges'] = df['tenure'] * df['MonthlyCharges']
    
    # SeniorCitizen × TechSupport (老年人 + 技术支持)
    df['Senior_TechSupport'] = df['SeniorCitizen'].astype(str) + '_' + df['TechSupport'].astype(str)
    
    # PaymentMethod × Contract (支付方式 + 合约类型)
    df['Payment_Contract'] = df['PaymentMethod'].astype(str) + '_' + df['Contract'].astype(str)
    
    # OnlineSecurity × OnlineBackup (安全服务组合)
    df['Security_Backup'] = df['OnlineSecurity'].astype(str) + '_' + df['OnlineBackup'].astype(str)
    
    return df

train = create_interaction_features(train)
test = create_interaction_features(test)

# 更新类别特征列表
cat_features = train.select_dtypes(include=['object']).columns.tolist()
print(f"交叉特征后类别特征: {len(cat_features)} 个")

# 2.3 数值特征处理
print("\n数值特征处理...")

# tenure 离散化 (0-6月/6-12月/12-24月/24月+)
def discretize_tenure(df):
    df = df.copy()
    df['tenure_group'] = pd.cut(
        df['tenure'], 
        bins=[0, 6, 12, 24, 100], 
        labels=['0-6m', '6-12m', '12-24m', '24m+']
    ).astype(str)
    return df

train = discretize_tenure(train)
test = discretize_tenure(test)
cat_features.append('tenure_group')

# MonthlyCharges 离散化
def discretize_monthly_charges(df):
    df = df.copy()
    df['MonthlyCharges_group'] = pd.cut(
        df['MonthlyCharges'], 
        bins=[0, 35, 70, 100, 200], 
        labels=['low', 'medium', 'high', 'very_high']
    ).astype(str)
    return df

train = discretize_monthly_charges(train)
test = discretize_monthly_charges(test)
cat_features.append('MonthlyCharges_group')

print(f"最终类别特征: {len(cat_features)} 个")

# ============================================================
# 3. Target Encoding (在 CV 内计算，避免 Leakage)
# ============================================================
print("\n" + "=" * 60)
print("5-Fold Stratified CV + Target Encoding...")

# 准备存储
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
fold_scores = []

# 5-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(train, y), 1):
    print(f"\n{'='*60}")
    print(f"Fold {fold}/5")
    print(f"{'='*60}")
    
    # 划分数据
    X_train_fold = train.iloc[train_idx].copy()
    X_val_fold = train.iloc[val_idx].copy()
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]
    
    # Target Encoding (只在训练集上计算，应用到验证集和测试集)
    target_enc_map = {}
    
    for col in cat_features:
        # 计算每个类别的目标变量均值
        temp_df = X_train_fold[[col]].copy()
        temp_df['target'] = y_train_fold.values
        encoding = temp_df.groupby(col)['target'].agg(['mean', 'count'])
        encoding.columns = ['target_mean', 'count']
        
        # 平滑处理 (Bayesian smoothing)
        global_mean = y_train_fold.mean()
        smoothing = 10  # 平滑参数
        encoding['target_enc'] = (
            (encoding['target_mean'] * encoding['count'] + global_mean * smoothing) / 
            (encoding['count'] + smoothing)
        )
        
        target_enc_map[col] = encoding['target_enc'].to_dict()
        
        # 应用编码
        X_train_fold[col + '_te'] = X_train_fold[col].map(target_enc_map[col]).fillna(global_mean)
        X_val_fold[col + '_te'] = X_val_fold[col].map(target_enc_map[col]).fillna(global_mean)
    
    # 删除原始类别特征
    X_train_fold = X_train_fold.drop(columns=cat_features)
    X_val_fold = X_val_fold.drop(columns=cat_features)
    
    # LightGBM 训练
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
    
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        valid_names=['valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # 验证集预测
    oof_preds[val_idx] = model.predict(X_val_fold, num_iteration=model.best_iteration)
    fold_auc = roc_auc_score(y_val_fold, oof_preds[val_idx])
    fold_scores.append(fold_auc)
    print(f"Fold {fold} AUC: {fold_auc:.6f}")
    
    # 测试集预测 (Target Encoding)
    X_test_fold = test.copy()
    for col in cat_features:
        X_test_fold[col + '_te'] = X_test_fold[col].map(target_enc_map[col]).fillna(y_train_fold.mean())
    X_test_fold = X_test_fold.drop(columns=cat_features)
    
    test_preds += model.predict(X_test_fold, num_iteration=model.best_iteration) / 5

# ============================================================
# 4. 最终评估
# ============================================================
print("\n" + "=" * 60)
print("最终结果")
print("=" * 60)

oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOut-of-Fold AUC: {oof_auc:.6f}")
print(f"各 Fold AUC: {fold_scores}")
print(f"平均 AUC: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")

# ============================================================
# 5. 生成提交文件
# ============================================================
print("\n生成提交文件...")
submission = pd.DataFrame({
    'id': test_id,
    'Churn': test_preds
})

submission.to_csv('submissions/improved_v1_lgb.csv', index=False)
print(f"提交文件已保存: submissions/improved_v1_lgb.csv")
print(f"预测概率范围: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
print("\n" + "=" * 60)
