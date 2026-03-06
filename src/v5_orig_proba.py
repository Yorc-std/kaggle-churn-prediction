"""
V5: V2 XGBoost + ORIG_proba 特征
关键改进：从原始数据集提取特征值的流失率（Playground 比赛特殊技巧）
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
orig = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')  # 原始数据集

train_id = train['id']
test_id = test['id']
y = (train['Churn'] == 'Yes').astype(int)
y_orig = (orig['Churn'] == 'Yes').astype(int)

train = train.drop(['id', 'Churn'], axis=1)
test = test.drop(['id'], axis=1)
orig = orig.drop(['customerID', 'Churn'], axis=1)

print(f"训练集: {train.shape}, 流失率: {y.mean():.2%}")
print(f"原始数据集: {orig.shape}, 流失率: {y_orig.mean():.2%}")
sys.stdout.flush()

def create_features(train_df, test_df, orig_df, y_orig):
    """V2 特征 + ORIG_proba 特征"""
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # V2 的核心特征
    for df in [train_df, test_df]:
        df['Contract_Internet'] = df['Contract'].astype(str) + '_' + df['InternetService'].astype(str)
        df['tenure_MonthlyCharges'] = df['tenure'] * df['MonthlyCharges']
        df['Senior_TechSupport'] = df['SeniorCitizen'].astype(str) + '_' + df['TechSupport'].astype(str)
        
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 100], 
                                     labels=['0-6m', '6-12m', '12-24m', '24m+']).astype(str)
        df['MonthlyCharges_group'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 70, 100, 200], 
                                            labels=['low', 'medium', 'high', 'very_high']).astype(str)
    
    # 🔥 ORIG_proba 特征：从原始数据集计算每个特征值的流失率
    print("\n添加 ORIG_proba 特征...")
    sys.stdout.flush()
    
    orig_df = orig_df.copy()
    orig_df['Churn'] = y_orig
    
    # 所有类别特征
    cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    # 数值特征
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    orig_proba_count = 0
    for col in cat_cols + num_cols:
        if col in orig_df.columns:
            # 计算原始数据集中每个值的流失率
            proba_map = orig_df.groupby(col)['Churn'].mean().to_dict()
            
            # 映射到训练集和测试集
            train_df[f'ORIG_proba_{col}'] = train_df[col].map(proba_map).fillna(y_orig.mean())
            test_df[f'ORIG_proba_{col}'] = test_df[col].map(proba_map).fillna(y_orig.mean())
            orig_proba_count += 1
    
    print(f"  添加了 {orig_proba_count} 个 ORIG_proba 特征")
    sys.stdout.flush()
    
    return train_df, test_df

def train_model(train_df, test_df, y_train, smoothing=3):
    """5-Fold CV with Target Encoding + XGBoost"""
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
        
        # XGBoost（增加轮次 + 降低学习率）
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'max_depth': 6,
            'learning_rate': 0.01,  # 从 0.05 降到 0.01
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
            num_boost_round=10000,  # 从 1000 增加到 10000
            evals=[(dval, 'valid')],
            early_stopping_rounds=200,  # 从 50 增加到 200
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
        
        print(f"AUC: {roc_auc_score(y_val_fold, oof_preds[val_idx]):.6f}")
        sys.stdout.flush()
    
    oof_auc = roc_auc_score(y_train, oof_preds)
    return oof_auc, test_preds

# 特征工程
print("\n创建特征...")
sys.stdout.flush()

train_fe, test_fe = create_features(train, test, orig, y_orig)

print(f"特征数: {train_fe.shape[1]}")
sys.stdout.flush()

# 训练
print("\n5-Fold CV 训练 (XGBoost + ORIG_proba)...")
sys.stdout.flush()

oof_auc, test_preds = train_model(train_fe, test_fe, y, smoothing=3)

print(f"\n→ OOF AUC: {oof_auc:.6f}")
print(f"   V2 XGBoost + Pseudo Labels: 0.916403")
sys.stdout.flush()

# 保存提交文件
submission = pd.DataFrame({
    'id': test_id,
    'Churn': test_preds
})
submission.to_csv('submissions/v5_orig_proba.csv', index=False)
print(f"\n提交文件已保存: submissions/v5_orig_proba.csv")
