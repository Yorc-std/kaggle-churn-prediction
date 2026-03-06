"""
Pseudo Labels - Threshold Grid Search
测试不同置信度阈值，找到最佳配置
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
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

def create_features(df):
    """V2 的特征工程"""
    df = df.copy()
    
    df['Contract_Internet'] = df['Contract'].astype(str) + '_' + df['InternetService'].astype(str)
    df['tenure_MonthlyCharges'] = df['tenure'] * df['MonthlyCharges']
    df['Senior_TechSupport'] = df['SeniorCitizen'].astype(str) + '_' + df['TechSupport'].astype(str)
    
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 100], 
                                 labels=['0-6m', '6-12m', '12-24m', '24m+']).astype(str)
    df['MonthlyCharges_group'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 70, 100, 200], 
                                        labels=['low', 'medium', 'high', 'very_high']).astype(str)
    
    return df

def train_one_round(train_df, test_df, y_train, cat_features, smoothing=3):
    """训练一轮，返回 OOF 和测试集预测"""
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train), 1):
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
        
        # XGBoost
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 0
        }
        
        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dval = xgb.DMatrix(X_val_fold, label=y_val_fold)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'valid')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        oof_preds[val_idx] = model.predict(dval)
        
        # 测试集预测
        X_test_fold = test_df.copy()
        for col in cat_features:
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
        dtest = xgb.DMatrix(X_test_fold)
        test_preds += model.predict(dtest) / 5
    
    return oof_preds, test_preds

# 特征工程
print("\n创建特征...")
train_fe = create_features(train)
test_fe = create_features(test)
cat_features = train_fe.select_dtypes(include=['object']).columns.tolist()

# 第一轮：获取基础预测
print("\n" + "="*60)
print("第 1 轮：基础模型（无 Pseudo Labels）")
print("="*60)

oof_base, test_base = train_one_round(train_fe, test_fe, y, cat_features)
auc_base = roc_auc_score(y, oof_base)
print(f"基础 OOF AUC: {auc_base:.6f}")

# 网格搜索不同阈值
print("\n" + "="*60)
print("网格搜索：测试不同置信度阈值")
print("="*60)

thresholds = [0.90, 0.92, 0.95, 0.97, 0.99]
results = []

for threshold in thresholds:
    print(f"\n测试阈值: {threshold}")
    sys.stdout.flush()
    
    # 选择高置信度样本
    high_conf_mask = (test_base > threshold) | (test_base < (1 - threshold))
    n_pseudo = high_conf_mask.sum()
    
    if n_pseudo == 0:
        print(f"  没有样本满足阈值 {threshold}，跳过")
        continue
    
    print(f"  选中 {n_pseudo} 个样本（{n_pseudo/len(test_fe):.1%}）")
    
    # 合并训练集和伪标签
    pseudo_X = test_fe[high_conf_mask].copy()
    pseudo_y = (test_base[high_conf_mask] > 0.5).astype(int)
    
    train_fe_aug = pd.concat([train_fe, pseudo_X], axis=0, ignore_index=True)
    y_aug = pd.concat([y, pd.Series(pseudo_y)], axis=0, ignore_index=True)
    
    # 重新训练（只在原始训练集上评估 OOF）
    oof_aug = np.zeros(len(train_fe))
    test_aug = np.zeros(len(test_fe))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_fe, y), 1):
        # 训练集 = 原始训练集的 train_idx + 所有伪标签
        aug_train_idx = list(train_idx) + list(range(len(train_fe), len(train_fe_aug)))
        
        X_train_fold = train_fe_aug.iloc[aug_train_idx].copy()
        X_val_fold = train_fe.iloc[val_idx].copy()
        y_train_fold = y_aug.iloc[aug_train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Target Encoding
        global_mean = y_train_fold.mean()
        
        for col in cat_features:
            temp_df = X_train_fold[[col]].copy()
            temp_df['target'] = y_train_fold.values
            encoding = temp_df.groupby(col)['target'].agg(['mean', 'count'])
            encoding['target_enc'] = (
                (encoding['mean'] * encoding['count'] + global_mean * 3) / 
                (encoding['count'] + 3)
            )
            enc_map = encoding['target_enc'].to_dict()
            
            X_train_fold[col + '_te'] = X_train_fold[col].map(enc_map).fillna(global_mean)
            X_val_fold[col + '_te'] = X_val_fold[col].map(enc_map).fillna(global_mean)
        
        X_train_fold = X_train_fold.drop(columns=cat_features)
        X_val_fold = X_val_fold.drop(columns=cat_features)
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 0
        }
        
        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dval = xgb.DMatrix(X_val_fold, label=y_val_fold)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'valid')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        oof_aug[val_idx] = model.predict(dval)
        
        # 测试集预测
        X_test_fold = test_fe.copy()
        for col in cat_features:
            temp_df = train_fe_aug.iloc[aug_train_idx][[col]].copy()
            temp_df['target'] = y_train_fold.values
            encoding = temp_df.groupby(col)['target'].agg(['mean', 'count'])
            encoding['target_enc'] = (
                (encoding['mean'] * encoding['count'] + global_mean * 3) / 
                (encoding['count'] + 3)
            )
            enc_map = encoding['target_enc'].to_dict()
            X_test_fold[col + '_te'] = X_test_fold[col].map(enc_map).fillna(global_mean)
        
        X_test_fold = X_test_fold.drop(columns=cat_features)
        dtest = xgb.DMatrix(X_test_fold)
        test_aug += model.predict(dtest) / 5
    
    auc_aug = roc_auc_score(y, oof_aug)
    improvement = auc_aug - auc_base
    
    print(f"  OOF AUC: {auc_aug:.6f} ({improvement:+.6f})")
    
    results.append({
        'threshold': threshold,
        'n_pseudo': n_pseudo,
        'pct_pseudo': n_pseudo / len(test_fe),
        'oof_auc': auc_aug,
        'improvement': improvement,
        'test_preds': test_aug
    })

# 结果汇总
print("\n" + "="*60)
print("结果汇总")
print("="*60)

results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'test_preds'} for r in results])
results_df = results_df.sort_values('oof_auc', ascending=False)

print(results_df.to_string(index=False))

# 最佳配置
best = results[results_df.index[0]]
print(f"\n最佳配置:")
print(f"  阈值: {best['threshold']}")
print(f"  伪标签数: {best['n_pseudo']} ({best['pct_pseudo']:.1%})")
print(f"  OOF AUC: {best['oof_auc']:.6f}")
print(f"  提升: {best['improvement']:+.6f}")

# 保存最佳模型预测
submission = pd.DataFrame({
    'id': test_id,
    'Churn': best['test_preds']
})
submission.to_csv('submissions/v2_xgb_pseudo_best.csv', index=False)
print(f"\n最佳配置提交文件已保存: submissions/v2_xgb_pseudo_best.csv")
