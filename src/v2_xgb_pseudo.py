"""
V2 XGBoost + Pseudo Labels
基于 V2 XGBoost，加入伪标签技术
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

def train_with_pseudo_labels(train_df, test_df, y_train, smoothing=3, 
                             confidence_threshold=0.95, max_iterations=2):
    """
    使用 Pseudo Labels 训练
    
    Args:
        confidence_threshold: 置信度阈值（> threshold 或 < 1-threshold 的样本被选中）
        max_iterations: 最多迭代次数
    """
    
    cat_features = train_df.select_dtypes(include=['object']).columns.tolist()
    
    # 第一轮：正常训练
    print(f"\n{'='*60}")
    print(f"第 1 轮训练（无 Pseudo Labels）")
    print(f"{'='*60}")
    
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    
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
        
        print(f"AUC: {roc_auc_score(y_val_fold, oof_preds[val_idx]):.6f}")
        sys.stdout.flush()
    
    oof_auc_round1 = roc_auc_score(y_train, oof_preds)
    print(f"\n第 1 轮 OOF AUC: {oof_auc_round1:.6f}")
    
    # 选择高置信度的测试集样本作为 Pseudo Labels
    high_conf_mask = (test_preds > confidence_threshold) | (test_preds < (1 - confidence_threshold))
    n_pseudo = high_conf_mask.sum()
    
    if n_pseudo == 0:
        print(f"\n没有找到置信度 > {confidence_threshold} 的样本，停止")
        return oof_auc_round1, test_preds
    
    print(f"\n选中 {n_pseudo} 个高置信度样本（{n_pseudo/len(test_df):.1%}）")
    
    # 第二轮：加入 Pseudo Labels
    print(f"\n{'='*60}")
    print(f"第 2 轮训练（加入 Pseudo Labels）")
    print(f"{'='*60}")
    
    # 合并训练集和伪标签
    pseudo_X = test_df[high_conf_mask].copy()
    pseudo_y = (test_preds[high_conf_mask] > 0.5).astype(int)
    
    train_df_aug = pd.concat([train_df, pseudo_X], axis=0, ignore_index=True)
    y_train_aug = pd.concat([y_train, pd.Series(pseudo_y)], axis=0, ignore_index=True)
    
    print(f"增强后训练集: {len(train_df_aug)} 行（原始 {len(train_df)} + 伪标签 {n_pseudo}）")
    
    # 重新训练
    oof_preds_aug = np.zeros(len(train_df))
    test_preds_aug = np.zeros(len(test_df))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train), 1):
        print(f"  Fold {fold}/5...", end=' ')
        sys.stdout.flush()
        
        # 训练集 = 原始训练集的 train_idx + 所有伪标签
        aug_train_idx = list(train_idx) + list(range(len(train_df), len(train_df_aug)))
        
        X_train_fold = train_df_aug.iloc[aug_train_idx].copy()
        X_val_fold = train_df.iloc[val_idx].copy()
        y_train_fold = y_train_aug.iloc[aug_train_idx]
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
        
        oof_preds_aug[val_idx] = model.predict(dval)
        
        # 测试集预测
        X_test_fold = test_df.copy()
        for col in cat_features:
            temp_df = train_df_aug.iloc[aug_train_idx][[col]].copy()
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
        test_preds_aug += model.predict(dtest) / 5
        
        print(f"AUC: {roc_auc_score(y_val_fold, oof_preds_aug[val_idx]):.6f}")
        sys.stdout.flush()
    
    oof_auc_round2 = roc_auc_score(y_train, oof_preds_aug)
    print(f"\n第 2 轮 OOF AUC: {oof_auc_round2:.6f}")
    print(f"提升: {oof_auc_round2 - oof_auc_round1:+.6f}")
    
    return oof_auc_round2, test_preds_aug

# 特征工程
print("\n创建特征...")
train_fe = create_features(train)
test_fe = create_features(test)

# 训练
oof_auc, test_preds = train_with_pseudo_labels(
    train_fe, test_fe, y, 
    smoothing=3,
    confidence_threshold=0.95
)

print(f"\n{'='*60}")
print(f"最终结果")
print(f"{'='*60}")
print(f"OOF AUC: {oof_auc:.6f}")
print(f"V2 XGBoost (无 Pseudo Labels): 0.916384")

# 保存提交文件
submission = pd.DataFrame({
    'id': test_id,
    'Churn': test_preds
})
submission.to_csv('submissions/v2_xgb_pseudo.csv', index=False)
print(f"\n提交文件已保存: submissions/v2_xgb_pseudo.csv")
