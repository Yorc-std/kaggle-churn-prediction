"""
Improved Model V2 - Parameter Tuning
测试不同配置找到最优参数:
1. Target Encoding smoothing (1, 3, 5, 10, 20)
2. 是否保留原始数值特征
3. 交叉特征组合
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
import itertools
import sys
warnings.filterwarnings('ignore')

# 强制刷新输出
sys.stdout.flush()

# ============================================================
# 读取数据
# ============================================================
print("读取数据...")
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train_id = train['id']
test_id = test['id']
y = (train['Churn'] == 'Yes').astype(int)

train = train.drop(['id', 'Churn'], axis=1)
test = test.drop(['id'], axis=1)

print(f"训练集: {train.shape}, 流失率: {y.mean():.2%}")

# ============================================================
# 特征工程函数
# ============================================================

def create_interaction_features(df, feature_set='full'):
    """创建交叉特征"""
    df = df.copy()
    
    if feature_set in ['full', 'core']:
        # 核心交叉特征（业务逻辑最强）
        df['Contract_Internet'] = df['Contract'].astype(str) + '_' + df['InternetService'].astype(str)
        df['tenure_MonthlyCharges'] = df['tenure'] * df['MonthlyCharges']
        df['Senior_TechSupport'] = df['SeniorCitizen'].astype(str) + '_' + df['TechSupport'].astype(str)
    
    if feature_set == 'full':
        # 额外交叉特征
        df['Payment_Contract'] = df['PaymentMethod'].astype(str) + '_' + df['Contract'].astype(str)
        df['Security_Backup'] = df['OnlineSecurity'].astype(str) + '_' + df['OnlineBackup'].astype(str)
    
    return df

def discretize_features(df, keep_original=True):
    """数值特征离散化"""
    df = df.copy()
    
    # tenure 离散化
    df['tenure_group'] = pd.cut(
        df['tenure'], 
        bins=[0, 6, 12, 24, 100], 
        labels=['0-6m', '6-12m', '12-24m', '24m+']
    ).astype(str)
    
    # MonthlyCharges 离散化
    df['MonthlyCharges_group'] = pd.cut(
        df['MonthlyCharges'], 
        bins=[0, 35, 70, 100, 200], 
        labels=['low', 'medium', 'high', 'very_high']
    ).astype(str)
    
    # 如果不保留原始特征，删除
    if not keep_original:
        df = df.drop(['tenure', 'MonthlyCharges'], axis=1)
    
    return df

def target_encode_cv(train_df, test_df, y_train, cat_features, smoothing=10):
    """5-Fold CV with Target Encoding"""
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train), 1):
        X_train_fold = train_df.iloc[train_idx].copy()
        X_val_fold = train_df.iloc[val_idx].copy()
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        # Target Encoding
        target_enc_map = {}
        global_mean = y_train_fold.mean()
        
        for col in cat_features:
            temp_df = X_train_fold[[col]].copy()
            temp_df['target'] = y_train_fold.values
            encoding = temp_df.groupby(col)['target'].agg(['mean', 'count'])
            
            # Bayesian smoothing
            encoding['target_enc'] = (
                (encoding['mean'] * encoding['count'] + global_mean * smoothing) / 
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
                lgb.log_evaluation(period=0)  # 静默模式
            ]
        )
        
        # 验证集预测
        oof_preds[val_idx] = model.predict(X_val_fold, num_iteration=model.best_iteration)
        
        # 测试集预测
        X_test_fold = test_df.copy()
        for col in cat_features:
            X_test_fold[col + '_te'] = X_test_fold[col].map(target_enc_map[col]).fillna(global_mean)
        X_test_fold = X_test_fold.drop(columns=cat_features)
        
        test_preds += model.predict(X_test_fold, num_iteration=model.best_iteration) / 5
    
    oof_auc = roc_auc_score(y_train, oof_preds)
    return oof_auc, test_preds

# ============================================================
# 参数搜索
# ============================================================
print("\n" + "=" * 60)
print("参数搜索开始")
print("=" * 60)

# 搜索空间
smoothing_values = [1, 3, 5, 10, 20]
keep_original_values = [True, False]
feature_sets = ['core', 'full']

results = []

total_configs = len(smoothing_values) * len(keep_original_values) * len(feature_sets)
config_num = 0

for smoothing, keep_original, feature_set in itertools.product(
    smoothing_values, keep_original_values, feature_sets
):
    config_num += 1
    print(f"\n[{config_num}/{total_configs}] 测试配置: smoothing={smoothing}, keep_original={keep_original}, features={feature_set}")
    sys.stdout.flush()
    
    # 特征工程
    train_fe = create_interaction_features(train, feature_set=feature_set)
    test_fe = create_interaction_features(test, feature_set=feature_set)
    
    train_fe = discretize_features(train_fe, keep_original=keep_original)
    test_fe = discretize_features(test_fe, keep_original=keep_original)
    
    # 识别类别特征
    cat_features = train_fe.select_dtypes(include=['object']).columns.tolist()
    
    # 训练和评估
    oof_auc, test_preds = target_encode_cv(train_fe, test_fe, y, cat_features, smoothing=smoothing)
    
    print(f"  → OOF AUC: {oof_auc:.6f}")
    sys.stdout.flush()
    
    results.append({
        'smoothing': smoothing,
        'keep_original': keep_original,
        'feature_set': feature_set,
        'oof_auc': oof_auc,
        'test_preds': test_preds
    })

# ============================================================
# 结果汇总
# ============================================================
print("\n" + "=" * 60)
print("搜索结果汇总")
print("=" * 60)

results_df = pd.DataFrame([
    {k: v for k, v in r.items() if k != 'test_preds'} 
    for r in results
])
results_df = results_df.sort_values('oof_auc', ascending=False)

print("\n前 5 名配置:")
print(results_df.head(10).to_string(index=False))

# 最佳配置
best_config = results[results_df.index[0]]
print(f"\n最佳配置:")
print(f"  smoothing: {best_config['smoothing']}")
print(f"  keep_original: {best_config['keep_original']}")
print(f"  feature_set: {best_config['feature_set']}")
print(f"  OOF AUC: {best_config['oof_auc']:.6f}")

# 保存最佳模型预测
submission = pd.DataFrame({
    'id': test_id,
    'Churn': best_config['test_preds']
})
submission.to_csv('submissions/improved_v2_tuned.csv', index=False)
print(f"\n最佳配置提交文件已保存: submissions/improved_v2_tuned.csv")

# 保存搜索结果
results_df.to_csv('tuning_results.csv', index=False)
print(f"搜索结果已保存: tuning_results.csv")
