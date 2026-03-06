"""
V6: V2 XGBoost + 充分训练
基于 V2 的最佳特征，增加训练轮次和降低学习率
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
    """V2 的特征工程（保持不变）"""
    df = df.copy()
    
    # 核心交叉特征
    df['Contract_Internet'] = df['Contract'].astype(str) + '_' + df['InternetService'].astype(str)
    df['tenure_MonthlyCharges'] = df['tenure'] * df['MonthlyCharges']
    df['Senior_TechSupport'] = df['SeniorCitizen'].astype(str) + '_' + df['TechSupport'].astype(str)
    
    # 数值特征离散化（保留原始）
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 100], 
                                 labels=['0-6m', '6-12m', '12-24m', '24m+']).astype(str)
    df['MonthlyCharges_group'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 70, 100, 200], 
                                        labels=['low', 'medium', 'high', 'very_high']).astype(str)
    
    return df

def train_model(train_df, test_df, y_train, smoothing=3):
    """5-Fold CV with Target Encoding + XGBoost（充分训练）"""
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
        
        # XGBoost（充分训练配置）
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'max_depth': 6,
            'learning_rate': 0.01,  # 降低学习率
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
            num_boost_round=10000,  # 增加上限
            evals=[(dval, 'valid')],
            early_stopping_rounds=200,  # 更稳健的停止条件
            verbose_eval=False
        )
        
        oof_preds[val_idx] = model.predict(dval)
        
        # 测试集
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
        
        # 打印实际训练轮次
        best_iteration = model.best_iteration
        print(f"AUC: {roc_auc_score(y_val_fold, oof_preds[val_idx]):.6f} (trained {best_iteration} rounds)")
        sys.stdout.flush()
    
    oof_auc = roc_auc_score(y_train, oof_preds)
    return oof_auc, test_preds

# 特征工程
print("\n创建特征...")
sys.stdout.flush()

train_fe = create_features(train)
test_fe = create_features(test)

print(f"特征数: {train_fe.shape[1]}")
sys.stdout.flush()

# 训练
print("\n5-Fold CV 训练 (XGBoost 充分训练)...")
sys.stdout.flush()

oof_auc, test_preds = train_model(train_fe, test_fe, y, smoothing=3)

print(f"\n→ OOF AUC: {oof_auc:.6f}")
print(f"   V2 XGBoost (lr=0.05, n=1000): 0.916384")
sys.stdout.flush()

# 保存提交文件
submission = pd.DataFrame({
    'id': test_id,
    'Churn': test_preds
})
submission.to_csv('submissions/v6_full_training.csv', index=False)
print(f"\n提交文件已保存: submissions/v6_full_training.csv")
