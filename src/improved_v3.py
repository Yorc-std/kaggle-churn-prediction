"""
Improved Model V3 - Advanced Feature Engineering
基于知识库建议的新特征组合
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

def create_advanced_features(df):
    """创建高级特征"""
    df = df.copy()
    
    # === V2 的核心特征（保留）===
    df['Contract_Internet'] = df['Contract'].astype(str) + '_' + df['InternetService'].astype(str)
    df['tenure_MonthlyCharges'] = df['tenure'] * df['MonthlyCharges']
    df['Senior_TechSupport'] = df['SeniorCitizen'].astype(str) + '_' + df['TechSupport'].astype(str)
    
    # 数值特征离散化（保留原始）
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 100], 
                                 labels=['0-6m', '6-12m', '12-24m', '24m+']).astype(str)
    df['MonthlyCharges_group'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 70, 100, 200], 
                                        labels=['low', 'medium', 'high', 'very_high']).astype(str)
    
    # === 新特征 ===
    
    # 1. 支付方式 × 电子账单
    df['Payment_Paperless'] = df['PaymentMethod'].astype(str) + '_' + df['PaperlessBilling'].astype(str)
    
    # 2. 老年人 × 电话服务
    df['Senior_Phone'] = df['SeniorCitizen'].astype(str) + '_' + df['PhoneService'].astype(str)
    
    # 3. 使用时长 × 合约类型
    df['tenure_Contract'] = df['tenure_group'] + '_' + df['Contract'].astype(str)
    
    # 4. 消费强度（月均消费 / 使用时长）
    df['charge_intensity'] = df['MonthlyCharges'] / (df['tenure'] + 1)  # +1 避免除零
    
    # 5. 实际使用月数（总消费 / 月消费）
    df['actual_months'] = df['TotalCharges'] / (df['MonthlyCharges'] + 0.01)  # +0.01 避免除零
    
    # 6. 使用月数与 tenure 的差异
    df['months_diff'] = df['actual_months'] - df['tenure']
    
    # 7. 增值服务总数
    value_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                     'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['value_services_count'] = 0
    for col in value_services:
        df['value_services_count'] += (df[col] == 'Yes').astype(int)
    
    # 8. 家庭结构
    df['family_structure'] = df['Partner'].astype(str) + '_' + df['Dependents'].astype(str)
    
    # 9. 互联网服务 × 流媒体使用
    df['Internet_Streaming'] = (df['InternetService'].astype(str) + '_' + 
                                df['StreamingTV'].astype(str) + '_' + 
                                df['StreamingMovies'].astype(str))
    
    # 10. 多线路 × 互联网服务
    df['MultiLines_Internet'] = df['MultipleLines'].astype(str) + '_' + df['InternetService'].astype(str)
    
    # 11. 人均月消费（考虑家庭成员）
    df['per_capita_charge'] = df['MonthlyCharges'].copy()
    df.loc[df['Partner'] == 'Yes', 'per_capita_charge'] /= 2
    df.loc[df['Dependents'] == 'Yes', 'per_capita_charge'] /= 2
    
    # 12. 消费稳定性（TotalCharges 与预期的偏差）
    df['charge_stability'] = np.abs(df['TotalCharges'] - df['MonthlyCharges'] * df['tenure']) / (df['TotalCharges'] + 1)
    
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
        test_preds += model.predict(X_test_fold, num_iteration=model.best_iteration) / 5
        
        print(f"AUC: {roc_auc_score(y_val_fold, oof_preds[val_idx]):.6f}")
        sys.stdout.flush()
    
    oof_auc = roc_auc_score(y_train, oof_preds)
    return oof_auc, test_preds

# 特征工程
print("\n创建高级特征...")
sys.stdout.flush()

train_fe = create_advanced_features(train)
test_fe = create_advanced_features(test)

print(f"特征数: {train_fe.shape[1]} (原始 19 → 新增 {train_fe.shape[1] - 19})")
sys.stdout.flush()

# 训练
print("\n5-Fold CV 训练...")
sys.stdout.flush()

oof_auc, test_preds = train_model(train_fe, test_fe, y, smoothing=3)

print(f"\n→ OOF AUC: {oof_auc:.6f}")
sys.stdout.flush()

# 保存提交文件
submission = pd.DataFrame({
    'id': test_id,
    'Churn': test_preds
})
submission.to_csv('submissions/improved_v3_advanced.csv', index=False)
print(f"\n提交文件已保存: submissions/improved_v3_advanced.csv")
