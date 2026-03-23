"""
特征工程脚本V4 - 整合高分Notebook优秀特征 (优化版)
运行此脚本生成:
  - data/train_features_v4.csv (训练集特征)
  - data/test_features_v4.csv (测试集特征)
"""

import pandas as pd
import numpy as np
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

print("特征工程 V4 - 整合高分Notebook优秀特征 (优化版)")

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
original = pd.read_csv("data/original.csv")

train_ids = train["id"].copy()
test_ids = test["id"].copy()
y = train["Churn"].map({"Yes": 1, "No": 0})

X = train.drop(["id", "Churn"], axis=1)
X_test = test.drop(["id"], axis=1)

print(f"\n原始数据: 训练集 {X.shape}, 测试集 {X_test.shape}")
print(f"原始IBM数据: {original.shape}")
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
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

original["Churn"] = original["Churn"].map({"Yes": 1, "No": 0})
original["TotalCharges"] = pd.to_numeric(original["TotalCharges"], errors="coerce")
original["TotalCharges"] = original["TotalCharges"].fillna(
    original["TotalCharges"].median()
)
if "customerID" in original.columns:
    original.drop(columns=["customerID"], inplace=True)

df = pd.concat([X, X_test], axis=0, ignore_index=True)
n_train = len(X)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

feature_count = 0

print("\n[1] 服务统计特征 (来自高分Notebook)")

df["service_yes_count"] = (df[SERVICE_COLS] == "Yes").sum(axis=1).astype("float32")
df["service_no_count"] = (df[SERVICE_COLS] == "No").sum(axis=1).astype("float32")
df["service_other_count"] = (
    (df[SERVICE_COLS].isin(["No phone service", "No internet service"]))
    .sum(axis=1)
    .astype("float32")
)
df["has_internet"] = (df["InternetService"] != "No").astype("float32")
df["has_phone"] = (df["PhoneService"] == "Yes").astype("float32")

df["has_partner"] = (df["Partner"] == "Yes").astype(int)
df["has_dependents"] = (df["Dependents"] == "Yes").astype(int)
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

df["service_diversity"] = (df["service_yes_count"] / 6.0).astype("float32")
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

feature_count += 17
print(f"  生成 17 个服务统计特征")

print("\n[2] 数值变换特征 (频率/排名/数学变换)")

for col in NUM_COLS:
    freq = pd.concat([X[col], X_test[col]]).value_counts(normalize=True)
    df[f"FREQ_{col}"] = df[col].map(freq).fillna(0).astype("float32")

for col in NUM_COLS:
    v = pd.to_numeric(df[col], errors="coerce")
    df[f"LOG1P_{col}"] = np.log1p(v.clip(lower=0)).astype("float32")
    df[f"SQRT_{col}"] = np.sqrt(v.clip(lower=0)).astype("float32")
    df[f"INV1P_{col}"] = (1.0 / (1.0 + v.clip(lower=0))).astype("float32")

_all_num = pd.concat([X[NUM_COLS], X_test[NUM_COLS]], axis=0, ignore_index=True)
for col in NUM_COLS:
    r = _all_num[col].rank(method="average", pct=True)
    df[f"RANK_{col}"] = r.values.astype("float32")

feature_count += 15
print(f"  生成 15 个数值变换特征")

print("\n[3] 费用偏差与比例特征 (高分Notebook核心特征)")

df["charges_deviation"] = (
    df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]
).astype("float32")
df["abs_charges_dev"] = np.abs(df["charges_deviation"]).astype("float32")
df["monthly_to_total_ratio"] = (df["MonthlyCharges"] / (df["TotalCharges"] + 1)).astype(
    "float32"
)
df["avg_monthly_charges"] = (df["TotalCharges"] / (df["tenure"] + 1)).astype("float32")
df["charges_per_service"] = (
    df["MonthlyCharges"] / (df["service_yes_count"] + 1)
).astype("float32")
df["monthly_per_internet"] = (df["MonthlyCharges"] / (df["has_internet"] + 1)).astype(
    "float32"
)

df["is_first_month"] = (df["tenure"] == 1).astype("float32")
df["dev_is_zero"] = (df["charges_deviation"] == 0).astype("float32")
df["dev_sign"] = np.sign(df["charges_deviation"]).astype("float32")
df["total_to_monthly_ratio"] = (df["TotalCharges"] / (df["MonthlyCharges"] + 1)).astype(
    "float32"
)
df["tenure_x_monthly"] = (df["tenure"] * df["MonthlyCharges"]).astype("float32")
df["tenure_x_total"] = (df["tenure"] * df["TotalCharges"]).astype("float32")

feature_count += 12
print(f"  生成 12 个费用偏差与比例特征")

print("\n[4] 分箱特征 (高分Notebook)")

TENURE_BINS = [0, 1, 3, 6, 12, 24, 36, 48, 60, 72, 10000]
df["tenure_bin"] = pd.cut(
    df["tenure"], bins=TENURE_BINS, include_lowest=True
).cat.codes.astype("float32")

mc_bins = pd.qcut(
    pd.concat([X["MonthlyCharges"], X_test["MonthlyCharges"]]),
    q=40,
    retbins=True,
    duplicates="drop",
)[1]
df["MonthlyCharges_bin"] = pd.cut(
    df["MonthlyCharges"], bins=mc_bins, include_lowest=True
).cat.codes.astype("float32")

tc_bins = pd.qcut(
    pd.concat([X["TotalCharges"], X_test["TotalCharges"]]),
    q=60,
    retbins=True,
    duplicates="drop",
)[1]
df["TotalCharges_bin"] = pd.cut(
    df["TotalCharges"], bins=tc_bins, include_lowest=True
).cat.codes.astype("float32")

df["tenure_new"] = (df["tenure"] <= 12).astype(int)
df["tenure_long"] = (df["tenure"] > 36).astype(int)
df["monthly_high"] = (df["MonthlyCharges"] > 70).astype(int)
df["total_high"] = (df["TotalCharges"] > 4000).astype(int)

feature_count += 7
print(f"  生成 7 个分箱特征")

print("\n[5] 数字/模数特征 (来自0.919 Notebook - 关键发现)")

df["tenure_mod10"] = (df["tenure"] % 10).astype("float32")
df["tenure_mod12"] = (df["tenure"] % 12).astype("float32")
df["tenure_years"] = (df["tenure"] // 12).astype("float32")
df["tenure_is_multiple_12"] = ((df["tenure"] % 12) == 0).astype("float32")
df["tenure_first_digit"] = (
    df["tenure"].astype(str).str[0].astype(float).astype("float32")
)
df["tenure_last_digit"] = (
    df["tenure"].astype(str).str[-1].astype(float).astype("float32")
)

mc = df["MonthlyCharges"].values
df["mc_fractional"] = (mc - np.floor(mc)).astype("float32")
df["mc_rounded_10"] = (np.round(mc / 10) * 10).astype("float32")
df["mc_dev_from_round10"] = np.abs(mc - df["mc_rounded_10"]).astype("float32")
df["mc_is_multiple_10"] = ((np.floor(mc) % 10) == 0).astype("float32")

tc = df["TotalCharges"].values
df["tc_fractional"] = (tc - np.floor(tc)).astype("float32")
df["tc_rounded_100"] = (np.round(tc / 100) * 100).astype("float32")
df["tc_is_multiple_100"] = ((np.floor(tc) % 100) == 0).astype("float32")

feature_count += 14
print(f"  生成 14 个数字/模数特征")

print("\n[6] 交叉特征 (高分Notebook核心)")

CROSS_PAIRS = [
    ("Contract", "InternetService"),
    ("PaymentMethod", "Contract"),
    ("InternetService", "OnlineSecurity"),
    ("PaymentMethod", "PaperlessBilling"),
    ("Contract", "PaperlessBilling"),
    ("InternetService", "TechSupport"),
]

cross_features = []
for a, b in CROSS_PAIRS:
    if a in df.columns and b in df.columns:
        name = f"{a}__{b}"
        df[name] = (
            (df[a].astype(str) + "|" + df[b].astype(str))
            .astype("category")
            .cat.codes.astype("float32")
        )
        cross_features.append(name)

TRIPLE = [("Contract", "InternetService", "PaymentMethod")]
for a, b, c in TRIPLE:
    if a in df.columns and b in df.columns and c in df.columns:
        name = f"{a}__{b}__{c}"
        df[name] = (
            (df[a].astype(str) + "|" + df[b].astype(str) + "|" + df[c].astype(str))
            .astype("category")
            .cat.codes.astype("float32")
        )
        cross_features.append(name)

feature_count += len(cross_features)
print(f"  生成 {len(cross_features)} 个交叉特征: {cross_features}")

print("\n[7] ISYES/ISNO/ISOTHER二值化特征")

YN_COLS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "MultipleLines",
]

isyes_features = []
isno_features = []
isother_features = []
for c in YN_COLS:
    if c in df.columns:
        s = df[c].astype(str)
        df[f"ISYES_{c}"] = (s == "Yes").astype("float32")
        df[f"ISNO_{c}"] = (s == "No").astype("float32")
        df[f"ISOTHER_{c}"] = (~s.isin(["Yes", "No"])).astype("float32")
        isyes_features.append(f"ISYES_{c}")
        isno_features.append(f"ISNO_{c}")
        isother_features.append(f"ISOTHER_{c}")

feature_count += len(isyes_features) + len(isno_features) + len(isother_features)
print(f"  生成 {len(isyes_features)} 个ISYES特征")
print(f"  生成 {len(isno_features)} 个ISNO特征")
print(f"  生成 {len(isother_features)} 个ISOTHER特征")

print("\n[8] 类别计数和稀有度特征")

ALL_CATS_FOR_COUNT = CAT_COLS + cross_features

cat_count_features = []
for c in ALL_CATS_FOR_COUNT:
    if c in df.columns:
        vc = df[c].value_counts(dropna=False)
        df[f"CAT_CNT_{c}"] = df[c].astype(str).map(vc).fillna(0).astype("float32")
        df[f"CAT_RARE_{c}"] = (df[f"CAT_CNT_{c}"] <= 50).astype("float32")
        cat_count_features.extend([f"CAT_CNT_{c}", f"CAT_RARE_{c}"])

feature_count += len(cat_count_features)
print(f"  生成 {len(cat_count_features)} 个类别计数特征")

print("\n[9] 原始数据目标编码特征 (核心)")

orig_proba_features = []
for col in CAT_COLS + NUM_COLS:
    tmp = original.groupby(col, observed=False)["Churn"].mean()
    feature_name = f"ORIG_proba_{col}"
    df[feature_name] = df[col].map(tmp).fillna(0.5).astype("float32")
    orig_proba_features.append(feature_name)

feature_count += len(orig_proba_features)
print(f"  生成 {len(orig_proba_features)} 个原始数据目标编码特征")

print("\n[10] ORIG_proba交叉特征 - 已移除 (原始数据无交互效应)")
print("  原因: original.csv是IBM合成数据，特征完全独立")
print("  train.csv基于original再合成，有交互效应，交叉编码会引入错误信息")
orig_proba_cross_features = []

print("\n[11] 百分位排名特征 (RealMLP Notebook)")


def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype("float32")


def zscore_against(values, reference):
    mu, sigma = np.mean(reference), np.std(reference)
    return (
        np.zeros(len(values), dtype="float32")
        if sigma == 0
        else ((values - mu) / sigma).astype("float32")
    )


orig_churner_tc = original.loc[original["Churn"] == 1, "TotalCharges"].values
orig_nonchurner_tc = original.loc[original["Churn"] == 0, "TotalCharges"].values
orig_tc = original["TotalCharges"].values

tc = df["TotalCharges"].values
df["_pctrank_nonchurner_TC"] = pctrank_against(tc, orig_nonchurner_tc)
df["_pctrank_churner_TC"] = pctrank_against(tc, orig_churner_tc)
df["_pctrank_orig_TC"] = pctrank_against(tc, orig_tc)
df["_zscore_churn_gap_TC"] = (
    np.abs(zscore_against(tc, orig_churner_tc))
    - np.abs(zscore_against(tc, orig_nonchurner_tc))
).astype("float32")
df["_zscore_nonchurner_TC"] = zscore_against(tc, orig_nonchurner_tc)
df["_pctrank_churn_gap_TC"] = (
    pctrank_against(tc, orig_churner_tc) - pctrank_against(tc, orig_nonchurner_tc)
).astype("float32")

for q_label, q_val in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
    ch_q = np.quantile(orig_churner_tc, q_val)
    nc_q = np.quantile(orig_nonchurner_tc, q_val)
    df[f"_dist_To_ch_{q_label}"] = np.abs(df["TotalCharges"] - ch_q).astype("float32")
    df[f"_dist_To_nc_{q_label}"] = np.abs(df["TotalCharges"] - nc_q).astype("float32")
    df[f"_qdist_gap_To_{q_label}"] = (
        df[f"_dist_To_nc_{q_label}"] - df[f"_dist_To_ch_{q_label}"]
    ).astype("float32")

feature_count += 15
print(f"  生成 15 个百分位排名特征")

print("\n[12] 条件百分位排名特征 (按InternetService分组)")

cond_pctrank_features = []
for is_val in df["InternetService"].unique():
    mask = df["InternetService"] == is_val
    ref = original.loc[original["InternetService"] == is_val, "TotalCharges"].values
    if len(ref) > 0:
        df.loc[mask, f"cond_pctrank_TC_IS_{is_val}"] = pctrank_against(
            df.loc[mask, "TotalCharges"].values, ref
        )
        cond_pctrank_features.append(f"cond_pctrank_TC_IS_{is_val}")

orig_is_mc_mean = original.groupby("InternetService")["MonthlyCharges"].mean()
df["resid_IS_MC"] = (
    df["MonthlyCharges"] - df["InternetService"].map(orig_is_mc_mean).fillna(0)
).astype("float32")
cond_pctrank_features.append("resid_IS_MC")

feature_count += len(cond_pctrank_features)
print(f"  生成 {len(cond_pctrank_features)} 个条件百分位排名特征")

print("\n[13] 风险因子特征")

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

df["fiber_no_support"] = (
    (df["InternetService"] == "Fiber optic") & (df["TechSupport"] == "No")
).astype(int)
df["is_new_customer"] = (df["tenure"] <= 6).astype(int)

feature_count += 11
print(f"  生成 11 个风险因子特征")

print("\n[14] 多项式与交叉特征")

df["contract_tenure"] = df["contract_type"] * df["tenure"]
df["contract_monthly"] = df["contract_type"] * df["MonthlyCharges"]
df["internet_tenure"] = df["internet_service_type"] * df["tenure"]
df["internet_monthly"] = df["internet_service_type"] * df["MonthlyCharges"]
df["service_tenure"] = df["service_yes_count"] * df["tenure"]
df["service_monthly"] = df["service_yes_count"] * df["MonthlyCharges"]
df["risk_tenure"] = df["risk_count"] * df["tenure"]

feature_count += 7
print(f"  生成 7 个多项式与交叉特征")

print("\n[15] 数值列作为类别")

NUM_AS_CAT = []
for col in NUM_COLS:
    _new = f"CAT_{col}"
    NUM_AS_CAT.append(_new)
    df[_new] = df[col].astype(str).astype("category").cat.codes.astype("float32")

feature_count += len(NUM_AS_CAT)
print(f"  生成 {len(NUM_AS_CAT)} 个数值类别特征: {NUM_AS_CAT}")

print("\n[16] 缺失值标记和平方特征")

missing_features = []
for col in NUM_COLS:
    df[f"MISS_{col}"] = pd.to_numeric(df[col], errors="coerce").isna().astype("int8")
    missing_features.append(f"MISS_{col}")

df["tenure_sq"] = (df["tenure"] ** 2).astype("float32")
df["MonthlyCharges_sq"] = (df["MonthlyCharges"] ** 2).astype("float32")
df["TotalCharges_sq"] = (df["TotalCharges"] ** 2).astype("float32")

feature_count += len(missing_features) + 3
print(f"  生成 {len(missing_features)} 个缺失值标记特征")
print(f"  生成 3 个平方特征")

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

print("\n[17] 保存特征文件...")
X_transformed.to_csv("data/train_features_v4.csv", index=False)
X_test_transformed.to_csv("data/test_features_v4.csv", index=False)

print(f"  训练集特征: data/train_features_v4.csv")
print(f"  测试集特征: data/test_features_v4.csv")

print("\n特征工程V4完成!")
print("\n优化说明:")

print("\n接下来运行训练脚本:")
print("  python src/v14_train_v4.py")
