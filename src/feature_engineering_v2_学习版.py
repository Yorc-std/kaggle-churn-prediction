"""
================================================================================
特征工程脚本V2 - 批量生成100+特征 (学习版)
================================================================================

运行此脚本生成:
  - data/train_features_v2.csv (训练集特征)
  - data/test_features_v2.csv (测试集特征)

================================================================================
【什么是特征工程？】
================================================================================
特征工程是机器学习中最重要的一步。简单来说，就是把原始数据转换成模型更容易理解的形式。

举个例子：
- 原始数据中有"tenure"(在网月数)，这是一个数值
- 我们可以创建新特征"是否新用户"(tenure <= 12)，这是一个布尔值
- 还可以创建"用户生命周期阶段"(0-6月/6-12月/1-2年/2年以上)

好的特征工程可以显著提升模型效果，甚至比选择更复杂的模型更有效！
业界有句话："特征工程决定了模型的上限，算法只是逼近这个上限"

================================================================================
【这个脚本做什么？】
================================================================================
1. 读取原始训练集和测试集数据
2. 创建各种新特征：
   - 服务统计特征：用户订阅了多少服务
   - 比例特征：平均每月消费、每个服务的费用等
   - 分箱特征：把连续数值分成几个区间
   - 聚合特征：按类别计算统计量
   - 风险因子特征：已知的流失风险因素
   - 交叉特征：两个特征的组合
3. 保存处理后的特征文件，供后续模型训练使用
================================================================================
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

print("特征工程 V2")

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
print("  【什么是服务统计特征？】")
print("  统计用户订阅了多少服务、有什么类型的订阅")
print("  这些特征反映了用户的参与度和忠诚度")

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
print("  【什么是数值变换？】")
print("  对数值特征进行数学变换，如log、平方根等")
print("  这里根据实验结果，这些特征对模型没有帮助，已删除")

feature_count += 0
print(f"  生成 {0} 个数值变换特征")

print("\n[3] 比例与交互特征")
print("  【什么是比例特征？】")
print("  比例特征可以揭示数据中的隐藏模式")
print("  例如：平均每月消费 = 总消费 / 在网月数")

df["avg_charges"] = df["TotalCharges"] / (df["tenure"] + 1)
df["charges_per_service"] = df["MonthlyCharges"] / (df["service_count"] + 1)
df["total_per_service"] = df["TotalCharges"] / (df["service_count"] + 1)
df["monthly_per_internet"] = df["MonthlyCharges"] / (df["has_internet"] + 1)

feature_count += 4
print(f"  生成 {4} 个比例与交互特征")

print("\n[4] 分箱特征")
print("  【什么是分箱？】")
print("  把连续数值分成几个区间，转换为离散类别")
print("  例如：年龄分成[0-18, 18-35, 35-60, 60+]")
print("  好处：")
print("    1. 降低噪声，使模型更稳定")
print("    2. 捕捉非线性关系")
print("    3. 处理异常值")

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
print("  【什么是聚合特征？】")
print("  按某个类别分组，计算每组的统计量（均值、标准差等）")
print("  例如：计算每种合同类型的平均月费")
print("  然后比较用户与同组平均值的差异")

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
print("  【什么是风险因子？】")
print("  根据业务知识，已知的流失风险因素")
print("  例如：月付合同、光纤网络、电子支票支付的用户更容易流失")

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
print("  【什么是交叉特征？】")
print("  两个特征的乘积，捕捉它们之间的交互作用")
print("  例如：合同类型 × 在网时长")
print("  长期合同的老用户 vs 短期合同的新用户，流失风险完全不同")

df["contract_tenure"] = df["contract_type"] * df["tenure"]
df["contract_monthly"] = df["contract_type"] * df["MonthlyCharges"]
df["internet_tenure"] = df["internet_service_type"] * df["tenure"]
df["internet_monthly"] = df["internet_service_type"] * df["MonthlyCharges"]
df["service_tenure"] = df["service_count"] * df["tenure"]
df["service_monthly"] = df["service_count"] * df["MonthlyCharges"]
df["risk_tenure"] = df["risk_count"] * df["tenure"]

feature_count += 7
print(f"  生成 {7} 个多项式与交叉特征")

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
