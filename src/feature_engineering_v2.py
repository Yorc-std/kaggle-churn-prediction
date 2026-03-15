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

print("\n[2] 数值变换特征")

for col in NUM_COLS:
    freq = df[col].value_counts(normalize=True)
    df[f"FREQ_{col}"] = df[col].map(freq).fillna(0)
    df[f"LOG1P_{col}"] = np.log1p(df[col])
    df[f"SQRT_{col}"] = np.sqrt(df[col])
    df[f"RANK_{col}"] = df[col].rank(pct=True)
    df[f"SQUARE_{col}"] = df[col] ** 2
    df[f"INV_{col}"] = 1.0 / (df[col] + 1)

feature_count += len(NUM_COLS) * 6
print(f"  生成 {len(NUM_COLS) * 6} 个数值变换特征")

print("\n[3] 比例与交互特征")

df["avg_charges"] = df["TotalCharges"] / (df["tenure"] + 1)
df["charges_diff"] = df["TotalCharges"] - df["MonthlyCharges"] * df["tenure"]
df["charges_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
df["tenure_charges"] = df["tenure"] * df["MonthlyCharges"]
df["tenure_total"] = df["tenure"] * df["TotalCharges"]
df["monthly_total_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

df["charges_per_service"] = df["MonthlyCharges"] / (df["service_count"] + 1)
df["total_per_service"] = df["TotalCharges"] / (df["service_count"] + 1)
df["tenure_per_service"] = df["tenure"] / (df["service_count"] + 1)

df["monthly_per_internet"] = df["MonthlyCharges"] / (df["has_internet"] + 1)
df["total_per_internet"] = df["TotalCharges"] / (df["has_internet"] + 1)

df["log_tenure_monthly"] = np.log1p(df["tenure"] * df["MonthlyCharges"])
df["log_tenure_total"] = np.log1p(df["tenure"] * df["TotalCharges"])
df["sqrt_tenure_monthly"] = np.sqrt(df["tenure"] * df["MonthlyCharges"])

df["monthly_minus_avg"] = df["MonthlyCharges"] - df["avg_charges"]
df["total_div_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)

feature_count += 15
print(f"  生成 {15} 个比例与交互特征")

print("\n[4] 分箱特征")

tenure_bins = [0, 6, 12, 24, 36, 48, 60, 72, 100]
tenure_labels = [0, 1, 2, 3, 4, 5, 6, 7]
df["tenure_bin"] = pd.cut(
    df["tenure"], bins=tenure_bins, labels=tenure_labels, include_lowest=True
).astype(int)

df["tenure_bin_6"] = (df["tenure"] <= 6).astype(int)
df["tenure_bin_12"] = ((df["tenure"] > 6) & (df["tenure"] <= 12)).astype(int)
df["tenure_bin_24"] = ((df["tenure"] > 12) & (df["tenure"] <= 24)).astype(int)
df["tenure_bin_48"] = ((df["tenure"] > 24) & (df["tenure"] <= 48)).astype(int)
df["tenure_bin_72"] = (df["tenure"] > 48).astype(int)

df["tenure_new"] = (df["tenure"] <= 12).astype(int)
df["tenure_medium"] = ((df["tenure"] > 12) & (df["tenure"] <= 36)).astype(int)
df["tenure_long"] = (df["tenure"] > 36).astype(int)

monthly_bins = [0, 30, 50, 70, 90, 120, 200]
monthly_labels = [0, 1, 2, 3, 4, 5]
df["monthly_bin"] = pd.cut(
    df["MonthlyCharges"], bins=monthly_bins, labels=monthly_labels, include_lowest=True
).astype(int)

df["monthly_low"] = (df["MonthlyCharges"] <= 30).astype(int)
df["monthly_medium"] = (
    (df["MonthlyCharges"] > 30) & (df["MonthlyCharges"] <= 70)
).astype(int)
df["monthly_high"] = (df["MonthlyCharges"] > 70).astype(int)

total_bins = [0, 500, 1000, 2000, 4000, 8000, 10000]
total_labels = [0, 1, 2, 3, 4, 5]
df["total_bin"] = pd.cut(
    df["TotalCharges"], bins=total_bins, labels=total_labels, include_lowest=True
).astype(int)

df["total_low"] = (df["TotalCharges"] <= 1000).astype(int)
df["total_medium"] = (
    (df["TotalCharges"] > 1000) & (df["TotalCharges"] <= 4000)
).astype(int)
df["total_high"] = (df["TotalCharges"] > 4000).astype(int)

feature_count += 18
print(f"  生成 {18} 个分箱特征")

print("\n[5] 聚合统计特征")

for cat_col in ["Contract", "InternetService", "PaymentMethod"]:
    for num_col in NUM_COLS:
        agg_mean = df.groupby(cat_col)[num_col].transform("mean")
        agg_std = df.groupby(cat_col)[num_col].transform("std").fillna(0)
        agg_median = df.groupby(cat_col)[num_col].transform("median")
        agg_max = df.groupby(cat_col)[num_col].transform("max")
        agg_min = df.groupby(cat_col)[num_col].transform("min")

        df[f"{cat_col}_{num_col}_mean"] = agg_mean
        df[f"{cat_col}_{num_col}_std"] = agg_std
        df[f"{cat_col}_{num_col}_median"] = agg_median
        df[f"{cat_col}_{num_col}_diff"] = df[num_col] - agg_mean
        df[f"{cat_col}_{num_col}_norm"] = (df[num_col] - agg_mean) / (agg_std + 1e-6)
        df[f"{cat_col}_{num_col}_range"] = agg_max - agg_min
        df[f"{cat_col}_{num_col}_rank"] = df.groupby(cat_col)[num_col].rank(pct=True)

feature_count += 3 * 3 * 7
print(f"  生成 {3 * 3 * 7} 个聚合统计特征")

print("\n[6] 风险因子特征")

df["risk_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
df["risk_fiber_optic"] = (df["InternetService"] == "Fiber optic").astype(int)
df["risk_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
df["risk_paperless"] = (df["PaperlessBilling"] == "Yes").astype(int)
df["risk_senior"] = (df["SeniorCitizen"] == 1).astype(int)
df["risk_no_partner"] = (df["Partner"] == "No").astype(int)
df["risk_no_dependents"] = (df["Dependents"] == "No").astype(int)

df["risk_count"] = (
    df["risk_month_to_month"]
    + df["risk_fiber_optic"]
    + df["risk_electronic_check"]
    + df["risk_paperless"]
    + df["risk_senior"]
    + df["risk_no_partner"]
    + df["risk_no_dependents"]
)

df["risk_high_tenure"] = (
    (df["tenure"] <= 12) & (df["Contract"] == "Month-to-month")
).astype(int)
df["risk_high_monthly"] = (
    (df["MonthlyCharges"] > 70) & (df["Contract"] == "Month-to-month")
).astype(int)
df["risk_no_security"] = (
    (df["OnlineSecurity"] == "No") & (df["has_internet"] == 1)
).astype(int)
df["risk_no_support"] = (
    (df["TechSupport"] == "No") & (df["has_internet"] == 1)
).astype(int)

df["protection_score"] = (
    (df["OnlineSecurity"] == "Yes").astype(int)
    + (df["OnlineBackup"] == "Yes").astype(int)
    + (df["DeviceProtection"] == "Yes").astype(int)
    + (df["TechSupport"] == "Yes").astype(int)
)

df["loyalty_score"] = df["contract_type"] * 10 + df["tenure"] / 6.0
df["value_score"] = df["TotalCharges"] / (df["MonthlyCharges"] + 1)

feature_count += 15
print(f"  生成 {15} 个风险因子特征")

print("\n[7] 多项式与交叉特征")

df["tenure_monthly_sq"] = df["tenure"] * df["MonthlyCharges"] ** 2
df["tenure_sq_monthly"] = df["tenure"] ** 2 * df["MonthlyCharges"]
df["tenure_monthly_cube"] = df["tenure"] * df["MonthlyCharges"] ** 3

df["log_tenure_log_monthly"] = np.log1p(df["tenure"]) * np.log1p(df["MonthlyCharges"])
df["sqrt_tenure_sqrt_monthly"] = np.sqrt(df["tenure"]) * np.sqrt(df["MonthlyCharges"])

df["tenure_monthly_total"] = df["tenure"] * df["MonthlyCharges"] * df["TotalCharges"]
df["tenure_plus_monthly"] = df["tenure"] + df["MonthlyCharges"]
df["tenure_minus_monthly"] = df["tenure"] - df["MonthlyCharges"]

df["contract_tenure"] = df["contract_type"] * df["tenure"]
df["contract_monthly"] = df["contract_type"] * df["MonthlyCharges"]
df["contract_total"] = df["contract_type"] * df["TotalCharges"]

df["internet_tenure"] = df["internet_service_type"] * df["tenure"]
df["internet_monthly"] = df["internet_service_type"] * df["MonthlyCharges"]
df["internet_total"] = df["internet_service_type"] * df["TotalCharges"]

df["service_tenure"] = df["service_count"] * df["tenure"]
df["service_monthly"] = df["service_count"] * df["MonthlyCharges"]
df["service_total"] = df["service_count"] * df["TotalCharges"]

df["risk_tenure"] = df["risk_count"] * df["tenure"]
df["risk_monthly"] = df["risk_count"] * df["MonthlyCharges"]

feature_count += 18
print(f"  生成 {18} 个多项式与交叉特征")

print("\n[8] 统计排名特征")

for col in NUM_COLS:
    df[f"DECILE_{col}"] = pd.qcut(df[col], q=10, labels=False, duplicates="drop")
    df[f"QUARTILE_{col}"] = pd.qcut(df[col], q=4, labels=False, duplicates="drop")

df["tenure_rank"] = df["tenure"].rank(pct=True)
df["monthly_rank"] = df["MonthlyCharges"].rank(pct=True)
df["total_rank"] = df["TotalCharges"].rank(pct=True)

df["tenure_monthly_rank_sum"] = df["tenure_rank"] + df["monthly_rank"]
df["tenure_total_rank_sum"] = df["tenure_rank"] + df["total_rank"]
df["monthly_total_rank_sum"] = df["monthly_rank"] + df["total_rank"]

df["tenure_monthly_rank_prod"] = df["tenure_rank"] * df["monthly_rank"]
df["tenure_total_rank_prod"] = df["tenure_rank"] * df["total_rank"]
df["monthly_total_rank_prod"] = df["monthly_rank"] * df["total_rank"]

feature_count += 15
print(f"  生成 {15} 个统计排名特征")

print("\n[9] 组合分类特征")

df["contract_internet"] = (
    df["Contract"].astype(str) + "_" + df["InternetService"].astype(str)
)
df["contract_payment"] = (
    df["Contract"].astype(str) + "_" + df["PaymentMethod"].astype(str)
)
df["internet_payment"] = (
    df["InternetService"].astype(str) + "_" + df["PaymentMethod"].astype(str)
)

df["senior_partner"] = df["SeniorCitizen"].astype(str) + "_" + df["Partner"].astype(str)
df["senior_dependents"] = (
    df["SeniorCitizen"].astype(str) + "_" + df["Dependents"].astype(str)
)
df["partner_dependents"] = (
    df["Partner"].astype(str) + "_" + df["Dependents"].astype(str)
)

df["phone_internet"] = (
    df["PhoneService"].astype(str) + "_" + df["InternetService"].astype(str)
)
df["phone_multiple"] = (
    df["PhoneService"].astype(str) + "_" + df["MultipleLines"].astype(str)
)

for col in [
    "contract_internet",
    "contract_payment",
    "internet_payment",
    "senior_partner",
    "senior_dependents",
    "partner_dependents",
    "phone_internet",
    "phone_multiple",
]:
    freq = df[col].value_counts(normalize=True)
    df[f"FREQ_{col}"] = df[col].map(freq).fillna(0)
    df[f"COUNT_{col}"] = df[col].map(df[col].value_counts()).fillna(0)

feature_count += 8 * 2
print(f"  生成 {8 * 2} 个组合分类特征")

print("\n[10] 高级数学特征")

df["tenure_zscore"] = (df["tenure"] - df["tenure"].mean()) / df["tenure"].std()
df["monthly_zscore"] = (df["MonthlyCharges"] - df["MonthlyCharges"].mean()) / df[
    "MonthlyCharges"
].std()
df["total_zscore"] = (df["TotalCharges"] - df["TotalCharges"].mean()) / df[
    "TotalCharges"
].std()

df["tenure_minmax"] = (df["tenure"] - df["tenure"].min()) / (
    df["tenure"].max() - df["tenure"].min()
)
df["monthly_minmax"] = (df["MonthlyCharges"] - df["MonthlyCharges"].min()) / (
    df["MonthlyCharges"].max() - df["MonthlyCharges"].min()
)
df["total_minmax"] = (df["TotalCharges"] - df["TotalCharges"].min()) / (
    df["TotalCharges"].max() - df["TotalCharges"].min()
)

df["tenure_clipped"] = df["tenure"].clip(
    lower=df["tenure"].quantile(0.01), upper=df["tenure"].quantile(0.99)
)
df["monthly_clipped"] = df["MonthlyCharges"].clip(
    lower=df["MonthlyCharges"].quantile(0.01), upper=df["MonthlyCharges"].quantile(0.99)
)
df["total_clipped"] = df["TotalCharges"].clip(
    lower=df["TotalCharges"].quantile(0.01), upper=df["TotalCharges"].quantile(0.99)
)

df["zscore_sum"] = df["tenure_zscore"] + df["monthly_zscore"] + df["total_zscore"]
df["zscore_prod"] = df["tenure_zscore"] * df["monthly_zscore"] * df["total_zscore"]
df["minmax_sum"] = df["tenure_minmax"] + df["monthly_minmax"] + df["total_minmax"]
df["minmax_prod"] = df["tenure_minmax"] * df["monthly_minmax"] * df["total_minmax"]

feature_count += 16
print(f"  生成 {16} 个高级数学特征")

print("\n[11] 时间价值特征")

df["monthly_tenure_ratio"] = df["MonthlyCharges"] / (df["tenure"] + 1)
df["total_tenure_ratio"] = df["TotalCharges"] / (df["tenure"] + 1)
df["monthly_total_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

df["expected_total"] = df["MonthlyCharges"] * df["tenure"]
df["actual_expected_ratio"] = df["TotalCharges"] / (df["expected_total"] + 1)
df["charge_deviation"] = df["TotalCharges"] - df["expected_total"]

df["monthly_growth"] = df["MonthlyCharges"] / (df["avg_charges"] + 1)
df["payment_efficiency"] = df["TotalCharges"] / (
    df["MonthlyCharges"] * df["tenure"] + 1
)

df["value_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)
df["cost_per_service"] = df["MonthlyCharges"] / (df["service_count"] + 1)
df["value_per_service"] = df["TotalCharges"] / (df["service_count"] + 1)

feature_count += 10
print(f"  生成 {10} 个时间价值特征")

drop_cols = [
    "contract_internet",
    "contract_payment",
    "internet_payment",
    "senior_partner",
    "senior_dependents",
    "partner_dependents",
    "phone_internet",
    "phone_multiple",
]
df = df.drop(columns=drop_cols)

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
