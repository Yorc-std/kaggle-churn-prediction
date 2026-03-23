"""
特征工程脚本 - 一次性计算所有特征
运行此脚本生成:
  - data/train_features.csv (训练集特征)
  - data/test_features.csv (测试集特征)
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

print("特征工程 - 一次性计算")

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train_ids = train["id"].copy()
test_ids = test["id"].copy()
y = train["Churn"].map({"Yes": 1, "No": 0})

X = train.drop(["id", "Churn"], axis=1)
X_test = test.drop(["id"], axis=1)

print(f"训练集: {X.shape}, 测试集: {X_test.shape}")
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

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("\n[1] 服务统计特征...")
df["service_count"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
df["has_internet"] = (df["InternetService"] != "No").astype(int)
df["has_phone"] = (df["PhoneService"] == "Yes").astype(int)
print(f"  service_count: mean={df['service_count'].mean():.2f}")
print(f"  has_internet: {df['has_internet'].mean():.2%}")

print("\n[2] 数值变换...")
for col in NUM_COLS:
    freq = df[col].value_counts(normalize=True)
    df[f"FREQ_{col}"] = df[col].map(freq).fillna(0)

for col in NUM_COLS:
    df[f"LOG1P_{col}"] = np.log1p(df[col])

for col in NUM_COLS:
    df[f"SQRT_{col}"] = np.sqrt(df[col])

for col in NUM_COLS:
    df[f"RANK_{col}"] = df[col].rank(pct=True)

print(f"  数值变换特征: {len(NUM_COLS) * 4} 个")

X_transformed = df.iloc[: len(X)].copy()
X_test_transformed = df.iloc[len(X) :].copy()

X_transformed["id"] = train_ids.values
X_transformed["Churn"] = y.values
X_test_transformed["id"] = test_ids.values

print(f"\n变换后特征数: {X_transformed.shape[1] - 2}")

print("\n[3] 保存特征文件...")
X_transformed.to_csv("data/train_features.csv", index=False)
X_test_transformed.to_csv("data/test_features.csv", index=False)

print(f"  训练集特征: data/train_features.csv ({X_transformed.shape})")
print(f"  测试集特征: data/test_features.csv ({X_test_transformed.shape})")

print("\n特征工程完成!")
print("\n接下来运行训练脚本:")
print("  python src/v9_ensemble.py")
