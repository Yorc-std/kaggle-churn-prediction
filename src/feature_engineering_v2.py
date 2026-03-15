"""
特征工程脚本V2 - 批量生成100+特征
运行此脚本生成:
  - data/train_features_v2.csv (训练集特征)
  - data/test_features_v2.csv (测试集特征)
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

print("特征工程 V2 - 批量生成100+特征")

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train_ids = train["id"].copy()
test_ids = test["id"].copy()
y = train["Churn"].map({"Yes": 1, "No": 0})

X = train.drop(["id", "Churn"], axis=1)
X_test = test.drop(["id"], axis=1)

print(f"\n原始数据: 训练集 {X.shape}, 测试集 {X_test.shape}")
print(f"流失率: {y.mean():.4f}")

CAT_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

df = pd.concat([X, X_test], axis=0, ignore_index=True)
n_train = len(X)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

feature_count = 0

print("\n[1] 服务统计特征")

df["service_count"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
df["service_count_no"] = (df[SERVICE_COLS] == "No").sum(axis=1)
df["service_count_no_internet"] = (df[SERVICE_COLS] == "No internet service").sum(
    axis=1
)
df["has_internet"] = (df["InternetService"] != "No").astype(int)
df["has_phone"] = (df["PhoneService"] == "Yes").astype(int)
df["has_multiple_lines"] = (df["MultipleLines"] == "Yes").astype(int)
df["has_partner"] = (df["Partner"] == "Yes").astype(int)
df["has_dependents"] = (df["Dependents"] == "Yes").astype(int)
df["is_senior"] = (df["SeniorCitizen"] == 1).astype(int)
df["is_male"] = (df["gender"] == "Male").astype(int)
df["paperless"] = (df["PaperlessBilling"] == "Yes").astype(int)

df["streaming_count"] = (
    (df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")
).astype(int)
df["streaming_both"] = (
    (df["StreamingTV"] == "Yes") & (df["StreamingMovies"] == "Yes")
).astype(int)

df["security_count"] = (
    (df["OnlineSecurity"] == "Yes")
    | (df["OnlineBackup"] == "Yes")
    | (df["DeviceProtection"] == "Yes")
    | (df["TechSupport"] == "Yes")
).astype(int)
df["security_all"] = (
    (df["OnlineSecurity"] == "Yes")
    & (df["OnlineBackup"] == "Yes")
    & (df["DeviceProtection"] == "Yes")
    & (df["TechSupport"] == "Yes")
).astype(int)

df["support_services"] = (
    (df["OnlineSecurity"] == "Yes") + (df["TechSupport"] == "Yes")
).astype(int)
df["backup_services"] = (
    (df["OnlineBackup"] == "Yes") + (df["DeviceProtection"] == "Yes")
).astype(int)

df["service_diversity"] = df["service_count"] / 6.0
df["internet_service_type"] = (
    df["InternetService"].map({"No": 0, "DSL": 1, "Fiber optic": 2}).fillna(0)
)
df["contract_type"] = (
    df["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2}).fillna(0)
)
df["payment_type"] = (
    df["PaymentMethod"]
    .map(
        {
            "Electronic check": 0,
            "Mailed check": 1,
            "Bank transfer (automatic)": 2,
            "Credit card (automatic)": 3,
        }
    )
    .fillna(0)
)

feature_count += 20
print(f"  生成 {20} 个服务统计特征")

print("\n[2] 数值变换特征 - 已删除(根据无用特征.txt)")

feature_count += 0
print(f"  生成 {0} 个数值变换特征")

print("\n[3] 比例与交互特征")

df["avg_charges"] = df["TotalCharges"] / (df["tenure"] + 1)
df["charges_per_service"] = df["MonthlyCharges"] / (df["service_count"] + 1)
df["total_per_service"] = df["TotalCharges"] / (df["service_count"] + 1)
df["monthly_per_internet"] = df["MonthlyCharges"] / (df["has_internet"] + 1)

feature_count += 4
print(f"  生成 {4} 个比例与交互特征")

print("\n[4] 分箱特征")

tenure_bins = [0, 6, 12, 24, 36, 48, 60, 72, 100]
tenure_labels = [0, 1, 2, 3, 4, 5, 6, 7]
df["tenure_bin"] = pd.cut(
    df["tenure"], bins=tenure_bins, labels=tenure_labels, include_lowest=True
).astype(int)

df["tenure_new"] = (df["tenure"] <= 12).astype(int)
df["tenure_long"] = (df["tenure"] > 36).astype(int)

monthly_bins = [0, 30, 50, 70, 90, 120, 200]
monthly_labels = [0, 1, 2, 3, 4, 5]
df["monthly_bin"] = pd.cut(
    df["MonthlyCharges"], bins=monthly_bins, labels=monthly_labels, include_lowest=True
).astype(int)

df["monthly_high"] = (df["MonthlyCharges"] > 70).astype(int)

total_bins = [0, 500, 1000, 2000, 4000, 8000, 10000]
total_labels = [0, 1, 2, 3, 4, 5]
df["total_bin"] = pd.cut(
    df["TotalCharges"], bins=total_bins, labels=total_labels, include_lowest=True
).astype(int)

df["total_high"] = (df["TotalCharges"] > 4000).astype(int)

feature_count += 8
print(f"  生成 {8} 个分箱特征")

print("\n[5] 聚合统计特征")

for cat_col in ["Contract", "InternetService", "PaymentMethod"]:
    for num_col in NUM_COLS:
        agg_mean = df.groupby(cat_col)[num_col].transform("mean")
        agg_std = df.groupby(cat_col)[num_col].transform("std").fillna(0)
        df[f"{cat_col}_{num_col}_mean"] = agg_mean
        df[f"{cat_col}_{num_col}_diff"] = df[num_col] - agg_mean
        df[f"{cat_col}_{num_col}_norm"] = (df[num_col] - agg_mean) / (agg_std + 1e-6)

feature_count += 3 * 3 * 3
print(f"  生成 {3 * 3 * 3} 个聚合统计特征")

print("\n[6] 风险因子特征")

df["risk_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
df["risk_fiber_optic"] = (df["InternetService"] == "Fiber optic").astype(int)
df["risk_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)

df["risk_count"] = (
    df["risk_month_to_month"] + df["risk_fiber_optic"] + df["risk_electronic_check"]
)

df["risk_high_tenure"] = (
    (df["tenure"] <= 12) & (df["Contract"] == "Month-to-month")
).astype(int)
df["risk_no_security"] = (
    (df["OnlineSecurity"] == "No") & (df["has_internet"] == 1)
).astype(int)

df["protection_score"] = (
    (df["OnlineSecurity"] == "Yes").astype(int)
    + (df["OnlineBackup"] == "Yes").astype(int)
    + (df["DeviceProtection"] == "Yes").astype(int)
    + (df["TechSupport"] == "Yes").astype(int)
)

df["loyalty_score"] = df["contract_type"] * 10 + df["tenure"] / 6.0

feature_count += 9
print(f"  生成 {9} 个风险因子特征")

print("\n[7] 多项式与交叉特征")

df["contract_tenure"] = df["contract_type"] * df["tenure"]
df["contract_monthly"] = df["contract_type"] * df["MonthlyCharges"]
df["internet_tenure"] = df["internet_service_type"] * df["tenure"]
df["internet_monthly"] = df["internet_service_type"] * df["MonthlyCharges"]
df["service_tenure"] = df["service_count"] * df["tenure"]
df["service_monthly"] = df["service_count"] * df["MonthlyCharges"]
df["risk_tenure"] = df["risk_count"] * df["tenure"]

feature_count += 7
print(f"  生成 {7} 个多项式与交叉特征")

print("\n[8] 统计排名特征 - 已删除(根据无用特征.txt)")

feature_count += 0
print(f"  生成 {0} 个统计排名特征")

print("\n[9] 组合分类特征 - 已删除(根据无用特征.txt)")

feature_count += 0
print(f"  生成 {0} 个组合分类特征")

print("\n[10] 高级数学特征 - 已删除(对树模型无用的线性变换)")

feature_count += 0
print(f"  生成 {0} 个高级数学特征")

print("\n[11] 时间价值特征 - 已删除(与前面特征高度重复)")

feature_count += 0
print(f"  生成 {0} 个时间价值特征")

X_transformed = df.iloc[:n_train].copy()
X_test_transformed = df.iloc[n_train:].copy()

X_transformed["id"] = train_ids.values
X_transformed["Churn"] = y.values
X_test_transformed["id"] = test_ids.values

final_feature_count = X_transformed.shape[1] - 2

print("\n特征工程汇总")
print(f"原始特征数: {X.shape[1]}")
print(f"新增特征数: {feature_count}")
print(f"最终特征数: {final_feature_count}")
print(f"训练集形状: {X_transformed.shape}")
print(f"测试集形状: {X_test_transformed.shape}")

print("\n[12] 保存特征文件...")
X_transformed.to_csv("data/train_features_v2.csv", index=False)
X_test_transformed.to_csv("data/test_features_v2.csv", index=False)

print(f"  训练集特征: data/train_features_v2.csv")
print(f"  测试集特征: data/test_features_v2.csv")

print("\n特征工程V2完成!")
print("\n接下来运行训练脚本:")
print("  python src/v9_ensemble.py")
