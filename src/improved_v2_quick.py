"""
Improved Model V2 - Quick Tuning
只测试最有希望的几个配置
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
import sys
warnings.filterwarnings('ignore')

print("读取数据...")
sys.stdout.flush()

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train_id = train['id']
test_id = test['id']
y = (train['Churn'] == 'Yes').astype(int)

train = train.drop(['id', 'Churn'], axis=1)
test = test.drop(['id'], axis=1)

print(f"训练集: {train.shape}, 流失率: {y.mean():.2%}")
sys.stdout.flush()

# 特征工程
def create_features(df, smoothing=3):
    """创建特征（保留原始数值特征）"""
    df = df.copy()
    
    # 核心交叉特征
    df['Contract_Internet'] = df['Contract'].astype(str) + '_' + df['InternetService'].astype(str)
    df['tenure_MonthlyCharges'] = df['tenure'] * df['MonthlyCharges']
    df['Senior_TechSupport'] = df['SeniorCitizen'].astype(str) + '_' + df['TechSupport'].astype(str)
    
    # 数值特征离散化（保留原始）
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 100], labels=['0-6m', '6-12m', '12-24m', '24m+']).astype(str)
    df['MonthlyCharges_group'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 70, 100, 200], labels=['low', 'medium', 'high', 'very_high']).astype(str)
    
    return df

def train_model(train_df, test_df, y_train, smoothing=3):
    """5-Fold CV with Target Encoding"""
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    
    cat_features = train_df.select_dtypes(include=['object']).columns.tolist()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train), 1):
        print(f"  Fold {fold}/5...", end=' ')
        sys.stdout.flush()
        
        X_train_fold = train_df.iloc[train_idx].copy()
        X_val_fold = train_df.iloc[val_idx].copy()
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        # Target Encoding
        global_mean = y_train_fold.mean()
        
        for col in cat_features:
            temp_df = X_train_fold[[col]].copy()
            temp_df['target'] = y_train_fold.values
            encoding = temp_df.groupby(col)['target'].agg(['mean', 'count'])
            
            encoding['target_enc'] = (
                (encoding['mean'] * encoding['count'] + global_mean * smoothing) / 
                (encoding['count'] + smoothing)
            )
            
            enc_map = encoding['target_enc'].to_dict()
            
            X_train_fold[col + '_te'] = X_train_fold[col].map(enc_map).fillna(global_mean)
            X_val_fold[col + '_te'] = X_val_fold[col].map(enc_map).fillna(global_mean)
        
        X_train_fold = X_train_fold.drop(columns=cat_features)
        X_val_fold = X_val_fold.drop(columns=cat_features)
        
        # LightGBM
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
                lgb.log_evaluation(period=0)
            ]
        )
        
        oof_preds[val_idx] = model.predict(X_val_fold, num_iteration=model.best_iteration)
        
        # 测试集（复用训练时的 encoding map）
        X_test_fold = test_df.copy()
        for col in cat_features:
            # 重新计算 encoding（使用完整训练集）
            temp_df = train_df.iloc[train_idx][[col]].copy()
            temp_df['target'] = y_train_fold.values
            encoding = temp_df.groupby(col)['target'].agg(['mean', 'count'])
            encoding['target_enc'] = (
                (encoding['mean'] * encoding['count'] + global_mean * smoothing) / 
                (encoding['count'] + smoothing)
            )
            enc_map = encoding['target_enc'].to_dict()
            X_test_fold[col + '_te'] = X_test_fold[col].map(enc_map).fillna(global_mean)
        
        X_test_fold = X_test_fold.drop(columns=cat_features)
        test_preds += model.predict(X_test_fold, num_iteration=model.best_iteration) / 5
        
        print(f"AUC: {roc_auc_score(y_val_fold, oof_preds[val_idx]):.6f}")
        sys.stdout.flush()
    
    oof_auc = roc_auc_score(y_train, oof_preds)
    return oof_auc, test_preds

# 测试不同 smoothing 值
print("\n" + "="*60)
print("测试不同 smoothing 值")
print("="*60)

smoothing_values = [1, 3, 5, 10]
results = []

for smoothing in smoothing_values:
    print(f"\nsmoothing={smoothing}")
    sys.stdout.flush()
    
    train_fe = create_features(train)
    test_fe = create_features(test)
    
    oof_auc, test_preds = train_model(train_fe, test_fe, y, smoothing=smoothing)
    
    print(f"→ OOF AUC: {oof_auc:.6f}")
    sys.stdout.flush()
    
    results.append({
        'smoothing': smoothing,
        'oof_auc': oof_auc,
        'test_preds': test_preds
    })

# 最佳配置
print("\n" + "="*60)
print("结果汇总")
print("="*60)

for r in results:
    print(f"smoothing={r['smoothing']:2d}: OOF AUC = {r['oof_auc']:.6f}")

best = max(results, key=lambda x: x['oof_auc'])
print(f"\n最佳配置: smoothing={best['smoothing']}, OOF AUC={best['oof_auc']:.6f}")

# 保存提交文件
submission = pd.DataFrame({
    'id': test_id,
    'Churn': best['test_preds']
})
submission.to_csv('submissions/improved_v2_tuned.csv', index=False)
print(f"\n提交文件已保存: submissions/improved_v2_tuned.csv")
