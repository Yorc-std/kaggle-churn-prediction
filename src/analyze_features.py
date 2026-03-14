"""
Feature Importance Analysis
分析 V3 模型的特征重要性
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("读取数据...")
train = pd.read_csv('data/train.csv')
y = (train['Churn'] == 'Yes').astype(int)
train = train.drop(['id', 'Churn'], axis=1)

def create_advanced_features(df):
    """创建高级特征"""
    df = df.copy()
    
    df['Contract_Internet'] = df['Contract'].astype(str) + '_' + df['InternetService'].astype(str)
    df['tenure_MonthlyCharges'] = df['tenure'] * df['MonthlyCharges']
    df['Senior_TechSupport'] = df['SeniorCitizen'].astype(str) + '_' + df['TechSupport'].astype(str)
    
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 100], 
                                 labels=['0-6m', '6-12m', '12-24m', '24m+']).astype(str)
    df['MonthlyCharges_group'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 70, 100, 200], 
                                        labels=['low', 'medium', 'high', 'very_high']).astype(str)
    
    df['Payment_Paperless'] = df['PaymentMethod'].astype(str) + '_' + df['PaperlessBilling'].astype(str)
    df['Senior_Phone'] = df['SeniorCitizen'].astype(str) + '_' + df['PhoneService'].astype(str)
    df['tenure_Contract'] = df['tenure_group'] + '_' + df['Contract'].astype(str)
    df['charge_intensity'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['actual_months'] = df['TotalCharges'] / (df['MonthlyCharges'] + 0.01)
    df['months_diff'] = df['actual_months'] - df['tenure']
    
    value_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                     'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['value_services_count'] = 0
    for col in value_services:
        df['value_services_count'] += (df[col] == 'Yes').astype(int)
    
    df['family_structure'] = df['Partner'].astype(str) + '_' + df['Dependents'].astype(str)
    df['Internet_Streaming'] = (df['InternetService'].astype(str) + '_' + 
                                df['StreamingTV'].astype(str) + '_' + 
                                df['StreamingMovies'].astype(str))
    df['MultiLines_Internet'] = df['MultipleLines'].astype(str) + '_' + df['InternetService'].astype(str)
    
    df['per_capita_charge'] = df['MonthlyCharges'].copy()
    df.loc[df['Partner'] == 'Yes', 'per_capita_charge'] /= 2
    df.loc[df['Dependents'] == 'Yes', 'per_capita_charge'] /= 2
    
    df['charge_stability'] = np.abs(df['TotalCharges'] - df['MonthlyCharges'] * df['tenure']) / (df['TotalCharges'] + 1)
    
    return df

train_fe = create_advanced_features(train)

cat_features = train_fe.select_dtypes(include=['object']).columns.tolist()
global_mean = y.mean()

for col in cat_features:
    temp_df = train_fe[[col]].copy()
    temp_df['target'] = y.values
    encoding = temp_df.groupby(col)['target'].agg(['mean', 'count'])
    encoding['target_enc'] = (
        (encoding['mean'] * encoding['count'] + global_mean * 3) / 
        (encoding['count'] + 3)
    )
    enc_map = encoding['target_enc'].to_dict()
    train_fe[col + '_te'] = train_fe[col].map(enc_map).fillna(global_mean)

train_fe = train_fe.drop(columns=cat_features)

remaining_cat = train_fe.select_dtypes(include=['object']).columns.tolist()
if remaining_cat:
    print(f"\n警告: 仍有字符列未处理: {remaining_cat}")
    train_fe = train_fe.drop(columns=remaining_cat)
    print(f"已删除这些列")

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

train_data = lgb.Dataset(train_fe, label=y)
model = lgb.train(params, train_data, num_boost_round=500)

importance_df = pd.DataFrame({
    'feature': train_fe.columns,
    'importance': model.feature_importance(importance_type='gain')
})
importance_df = importance_df.sort_values('importance', ascending=False)

print("\n" + "="*60)
print("Top 30 特征重要性")
print("="*60)
print(importance_df.head(30).to_string(index=False))

new_features = [
    'Payment_Paperless', 'Senior_Phone', 'tenure_Contract', 
    'charge_intensity', 'actual_months', 'months_diff',
    'value_services_count', 'family_structure', 'Internet_Streaming',
    'MultiLines_Internet', 'per_capita_charge', 'charge_stability'
]

print("\n" + "="*60)
print("新特征排名")
print("="*60)
for feat in new_features:
    matching = importance_df[importance_df['feature'].str.contains(feat, regex=False)]
    if len(matching) > 0:
        rank = importance_df.index.get_loc(matching.index[0]) + 1
        imp = matching.iloc[0]['importance']
        print(f"{feat:25s}: 排名 {rank:2d}, 重要性 {imp:8.0f}")
    else:
        print(f"{feat:25s}: 未找到")

importance_df.to_csv('feature_importance_v3.csv', index=False)
print(f"\n完整特征重要性已保存: feature_importance_v3.csv")
