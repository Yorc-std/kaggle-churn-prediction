"""
================================================================================
V13: LightGBM + XGBoost 集成模型训练脚本
================================================================================

【这个脚本做什么？】
1. 加载特征工程处理后的数据
2. 使用 Target Encoding 处理分类特征
3. 使用 6 折交叉验证训练 XGBoost 和 LightGBM 模型
4. 融合两个模型的预测结果
5. 生成提交文件

================================================================================
【什么是集成学习？】
================================================================================
集成学习就是"三个臭皮匠顶个诸葛亮"的思想。

单个模型可能有偏差，但多个不同模型的组合往往更稳定、更准确。

这里我们使用 XGBoost 和 LightGBM 两种梯度提升树模型，它们各有特点：
- XGBoost: 更保守，正则化更强，不容易过拟合
- LightGBM: 训练更快，能处理更大的数据集

================================================================================
【什么是交叉验证？】
================================================================================
交叉验证是把训练数据分成多份（这里分6份），每次用其中一份做验证，其余做训练。

这样可以：
1. 充分利用数据（每条数据都会被验证一次）
2. 得到更可靠的模型评估
3. 避免偶然性（单次划分可能运气好或运气差）

================================================================================
【什么是梯度提升树？】
================================================================================
梯度提升树（Gradient Boosting Decision Tree, GBDT）是一种强大的机器学习算法。

核心思想：
1. 训练第一棵树，预测结果
2. 计算预测误差（残差）
3. 训练第二棵树，预测残差
4. 重复...直到达到指定数量或误差不再下降

最终预测 = 所有树的预测之和

XGBoost 和 LightGBM 都是 GBDT 的高效实现。

================================================================================
"""

# ==================== 导入库 ====================
# pandas: 数据处理库
import pandas as pd

# numpy: 数值计算库
import numpy as np

# sklearn: 机器学习工具库
# StratifiedKFold: 分层交叉验证，保证每折中各类别比例与整体一致
from sklearn.model_selection import StratifiedKFold

# roc_auc_score: AUC评估指标，用于衡量二分类模型性能
from sklearn.metrics import roc_auc_score

# xgboost: 极端梯度提升库
import xgboost as xgb

# lightgbm: 轻量级梯度提升库
import lightgbm as lgb

# warnings: 警告控制
import warnings

# math: 数学函数库
import math

# os: 操作系统接口
import os

# pickle: Python对象序列化库，用于保存和加载Python对象
import pickle

# hashlib: 哈希函数库，用于生成唯一标识
import hashlib

# glob: 文件路径匹配库，用于查找文件
import glob

# 忽略警告信息
warnings.filterwarnings("ignore")

# ==================== 全局参数配置 ====================
# 这些参数控制学习率衰减策略

# XGBoost 学习率参数
XGB_LR_START = 0.1  # 初始学习率：开始时较大，让模型快速学习
XGB_LR_END = 0.01  # 最终学习率：后期较小，让模型精细调整
XGB_LR_DECAY_ITER = 1500  # 衰减完成的迭代次数

# LightGBM 学习率参数
LGB_LR_START = 0.03
LGB_LR_END = 0.005
LGB_LR_DECAY_ITER = 1000

# 训练参数
NUM_BOOST_ROUND = 3000  # 最大迭代次数（最多训练3000棵树）
EARLY_STOPPING_ROUNDS = 200  # 早停轮数（如果200轮没有提升就停止）


# ==================== 学习率衰减函数 ====================
def get_learning_rate(current_iter, lr_start, lr_end, lr_decay_iter, mode="cosine"):
    """
    计算当前迭代的学习率（学习率衰减策略）

    【什么是学习率衰减？】
    学习率控制模型每一步更新的幅度。
    - 开始时用较大的学习率，让模型快速学习大方向
    - 后期用较小的学习率，让模型精细调整，找到最优点

    这就像下山：开始大步快走，接近山脚时小步慢走找最优点。

    【三种衰减模式】
    1. cosine（余弦衰减）: 学习率平滑下降，开始慢、中间快、最后慢
    2. linear（线性衰减）: 学习率均匀下降
    3. exp（指数衰减）: 学习率按指数下降

    参数说明:
        current_iter: 当前迭代次数（第几轮训练）
        lr_start: 初始学习率
        lr_end: 最终学习率
        lr_decay_iter: 衰减完成的迭代次数（之后保持lr_end）
        mode: 衰减模式 ("cosine"余弦/"linear"线性/"exp"指数)

    返回:
        当前迭代应该使用的学习率
    """
    # 计算训练进度（0到1之间）
    # min确保进度不超过1
    progress = min(current_iter / lr_decay_iter, 1.0)

    if mode == "cosine":
        # 余弦衰减：学习率按余弦曲线下降
        # 公式：lr = lr_end + 0.5 * (lr_start - lr_end) * (1 + cos(π * progress))
        # 特点：开始下降慢，中间下降快，最后下降慢
        lr = lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * progress))
    elif mode == "linear":
        # 线性衰减：学习率均匀下降
        # 公式：lr = lr_start - (lr_start - lr_end) * progress
        lr = lr_start - (lr_start - lr_end) * progress
    elif mode == "exp":
        # 指数衰减：学习率按指数下降
        # 公式：lr = lr_start * (lr_end / lr_start) ^ progress
        lr = lr_start * (lr_end / lr_start) ** progress
    else:
        # 不衰减，保持初始学习率
        lr = lr_start
    return lr


# ==================== Target Encoding 函数 ====================
def target_encode_cv(X_train, y_train, X_test, cat_cols, n_splits=6, smoothing=5):
    """
    Target Encoding（目标编码）- 将分类特征转换为数值

    【什么是 Target Encoding？】
    分类特征（如"Contract"有"月付/年付/两年"三个值）不能直接输入模型。
    Target Encoding 用每个类别对应的目标变量均值来替代类别值。

    例如：
    - 月付用户的流失率是40%
    - 年付用户的流失率是15%
    - 两年付用户的流失率是5%

    那么把"月付"编码为0.4，"年付"编码为0.15，"两年付"编码为0.05

    【为什么要用交叉验证？】
    如果直接用全部数据计算编码，会造成"数据泄露"——模型看到了答案。
    用交叉验证，每次只用训练集计算编码，验证集用训练集的编码结果。

    【smoothing 是什么？】
    平滑参数，用于处理样本量少的类别。
    如果某类别只有几个样本，直接用均值不可靠，需要向全局均值靠近。

    公式: encoded_value = (count * mean + smoothing * global_mean) / (count + smoothing)
    - count越大，越相信该类别的均值
    - count越小，越相信全局均值

    参数说明:
        X_train: 训练集特征
        y_train: 训练集标签
        X_test: 测试集特征
        cat_cols: 需要编码的分类特征列表
        n_splits: 交叉验证折数
        smoothing: 平滑系数，越大越向全局均值靠近

    返回:
        编码后的训练集和测试集
    """
    # 复制数据，避免修改原数据
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    # 计算全局均值（所有样本的标签均值）
    global_mean = y_train.mean()

    # 对每个分类特征进行编码
    for col in cat_cols:
        # 初始化out-of-fold编码数组
        # OOF(Out of Fold): 每折验证集的预测结果
        oof_encoding = np.zeros(len(X_train))

        # 创建分层交叉验证器
        # StratifiedKFold 保证每折中正负样本比例与整体一致
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # 对每折进行编码
        for train_idx, val_idx in skf.split(X_train, y_train):
            # 获取当前折的训练集
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]

            # 计算每个类别的编码值
            encoding_map = {}
            for cat in X_train[col].unique():
                # 找出该类别在训练集中的样本
                mask = X_tr[col] == cat
                count = mask.sum()  # 该类别的样本数
                mean = y_tr[mask].mean() if count > 0 else global_mean  # 该类别的流失率

                # 应用平滑公式
                encoding_map[cat] = (count * mean + smoothing * global_mean) / (
                    count + smoothing
                )

            # 对验证集进行编码
            oof_encoding[val_idx] = (
                X_train.iloc[val_idx][col].map(encoding_map).fillna(global_mean)
            )

        # 将编码结果添加为新特征（前缀TE_表示Target Encoding）
        X_train_encoded[f"TE_{col}"] = oof_encoding

        # 对测试集进行编码（使用全部训练数据计算编码）
        encoding_map = {}
        for cat in X_train[col].unique():
            mask = X_train[col] == cat
            count = mask.sum()
            mean = y_train[mask].mean() if count > 0 else global_mean
            encoding_map[cat] = (count * mean + smoothing * global_mean) / (
                count + smoothing
            )
        X_test_encoded[f"TE_{col}"] = X_test[col].map(encoding_map).fillna(global_mean)

    # 删除原始分类特征（已经编码过了）
    X_train_encoded = X_train_encoded.drop(cat_cols, axis=1)
    X_test_encoded = X_test_encoded.drop(cat_cols, axis=1)

    return X_train_encoded, X_test_encoded


# ==================== 预计算编码函数（带缓存）====================
def precompute_fold_encodings(X, y, X_test, cat_cols, n_splits=6, smoothing=5):
    """
    预计算每折的 Target Encoding（带缓存）

    【为什么要预计算？】
    在交叉验证中，每折都需要重新计算编码，这很耗时。
    预计算一次并缓存，后续训练时直接读取，大大加快速度。

    【缓存机制】
    使用 pickle 将编码结果保存到文件。
    文件名包含特征的哈希值，确保特征变化时缓存会更新。

    参数说明:
        X: 训练集特征
        y: 训练集标签
        X_test: 测试集特征
        cat_cols: 需要编码的分类特征列表
        n_splits: 交叉验证折数
        smoothing: 平滑系数

    返回:
        包含每折编码结果和测试集编码的字典
    """
    # 删除旧的缓存文件
    # glob.glob 返回匹配的文件列表
    for old_cache in glob.glob("cache/te_encodings*.pkl"):
        os.remove(old_cache)
        print(f"已删除旧缓存: {old_cache}")

    # 生成缓存文件名（使用特征列表的哈希值）
    # 这样当特征列表变化时，会生成新的缓存文件
    cols_hash = hashlib.md5(str(sorted(cat_cols)).encode()).hexdigest()[:8]
    cache_path = f"cache/te_encodings_{cols_hash}.pkl"

    print("计算Target Encoding...")

    # 创建缓存目录
    os.makedirs("cache", exist_ok=True)

    # 计算全局均值
    global_mean = y.mean()

    # 创建分层交叉验证器
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 存储每折的编码结果
    fold_encodings = {}

    # 对每折进行编码
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        # 获取当前折的训练集和验证集
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        X_tr_enc, X_val_enc = X_tr.copy(), X_val.copy()

        # 对每个分类特征进行编码
        for col in cat_cols:
            encoding_map = {}
            for cat in X[col].unique():
                mask = X_tr[col] == cat
                count = mask.sum()
                mean = y_tr[mask].mean() if count > 0 else global_mean
                encoding_map[cat] = (count * mean + smoothing * global_mean) / (
                    count + smoothing
                )
            # 对训练集和验证集进行编码
            X_tr_enc[f"TE_{col}"] = X_tr[col].map(encoding_map).fillna(global_mean)
            X_val_enc[f"TE_{col}"] = X_val[col].map(encoding_map).fillna(global_mean)

        # 保存当前折的编码结果
        fold_encodings[fold] = {
            "train_idx": train_idx,  # 训练集索引
            "val_idx": val_idx,  # 验证集索引
            "X_tr_enc": X_tr_enc.drop(cat_cols, axis=1),  # 编码后的训练集
            "X_val_enc": X_val_enc.drop(cat_cols, axis=1),  # 编码后的验证集
        }

    # 对测试集进行编码（使用全部训练数据）
    X_test_enc = X_test.copy()
    for col in cat_cols:
        encoding_map = {}
        for cat in X[col].unique():
            mask = X[col] == cat
            count = mask.sum()
            mean = y[mask].mean() if count > 0 else global_mean
            encoding_map[cat] = (count * mean + smoothing * global_mean) / (
                count + smoothing
            )
        X_test_enc[f"TE_{col}"] = X_test[col].map(encoding_map).fillna(global_mean)

    # 整理结果
    result = {
        "fold_encodings": fold_encodings,  # 每折的编码结果
        "X_test_enc": X_test_enc.drop(cat_cols, axis=1),  # 测试集编码结果
    }

    # 保存到缓存文件
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"已缓存: {cache_path}")

    return result


# ==================== XGBoost 学习率回调类 ====================
class XgbLearningRateCallback(xgb.callback.TrainingCallback):
    """
    XGBoost 学习率回调函数

    【什么是回调函数？】
    回调函数是在训练过程中自动执行的代码。
    这里我们在每次迭代后自动调整学习率，实现学习率衰减。

    【为什么要自定义回调？】
    XGBoost 原生不支持复杂的学习率衰减策略，
    通过自定义回调函数可以实现余弦衰减等高级策略。

    【如何使用？】
    在 xgb.train() 的 callbacks 参数中传入实例：
    callbacks=[XgbLearningRateCallback()]
    """

    def __init__(
        self,
        lr_start=XGB_LR_START,
        lr_end=XGB_LR_END,
        lr_decay_iter=XGB_LR_DECAY_ITER,
        mode="cosine",
    ):
        """
        初始化回调函数

        参数:
            lr_start: 初始学习率
            lr_end: 最终学习率
            lr_decay_iter: 衰减完成的迭代次数
            mode: 衰减模式
        """
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.lr_decay_iter = lr_decay_iter
        self.mode = mode

    def after_iteration(self, model, epoch, evals_log):
        """
        每次迭代后调用，更新学习率

        这是 TrainingCallback 的核心方法，XGBoost 会在每轮训练后自动调用。

        参数:
            model: XGBoost 模型
            epoch: 当前迭代次数（从0开始）
            evals_log: 评估日志（包含每轮的评估结果）

        返回:
            False 表示继续训练，True 表示停止训练
        """
        # 计算新的学习率
        new_lr = get_learning_rate(
            epoch, self.lr_start, self.lr_end, self.lr_decay_iter, self.mode
        )
        # 更新模型的学习率参数
        model.set_param("learning_rate", new_lr)
        # 返回 False 继续训练
        return False


# ==================== LightGBM 学习率回调函数 ====================
def lgb_learning_rate_callback(env):
    """
    LightGBM 学习率回调函数

    【与 XGBoost 的区别】
    LightGBM 使用不同的回调机制，通过 env 对象访问模型和迭代信息。

    env 对象包含:
    - env.iteration: 当前迭代次数
    - env.model: LightGBM 模型
    - env.evaluation_result_list: 评估结果列表

    【如何使用？】
    在 lgb.train() 的 callbacks 参数中传入函数：
    callbacks=[lgb_learning_rate_callback]
    """
    # 计算新的学习率
    new_lr = get_learning_rate(
        env.iteration, LGB_LR_START, LGB_LR_END, LGB_LR_DECAY_ITER, mode="cosine"
    )
    # 更新模型的学习率
    env.model.reset_parameter({"learning_rate": new_lr})


# ==================== 主程序开始 ====================
print("V13: LightGBM + XGBoost Ensemble (双重信号理论优化)")

# ==================== 1. 加载数据 ====================
print("\n加载预处理特征...")
# 读取特征工程处理后的数据
train = pd.read_csv("data/train_features_v2.csv")
test = pd.read_csv("data/test_features_v2.csv")

# 分离标签和特征
y = train["Churn"]  # 标签：是否流失
test_ids = test["id"]  # 测试集ID，用于提交
X = train.drop(["id", "Churn"], axis=1)  # 训练集特征
X_test = test.drop(["id"], axis=1)  # 测试集特征

# 打印数据信息
print(f"训练集: {X.shape}, 测试集: {X_test.shape}")
print(f"流失率: {y.mean():.4f}")
print(f"\n特征列表 ({len(X.columns)} 个):")
print(f"  {list(X.columns[:10])} ... (前10个)")

# ==================== 2. 定义分类特征列表 ====================
# 这些特征需要进行 Target Encoding
ALL_CAT_COLS = [
    "gender",  # 性别
    "SeniorCitizen",  # 是否老年人
    "Partner",  # 是否有伴侣
    "Dependents",  # 是否有受抚养人
    "PhoneService",  # 是否有电话服务
    "MultipleLines",  # 是否有多条线路
    "InternetService",  # 互联网服务类型
    "OnlineSecurity",  # 在线安全服务
    "OnlineBackup",  # 在线备份服务
    "DeviceProtection",  # 设备保护服务
    "TechSupport",  # 技术支持服务
    "StreamingTV",  # 电视流媒体服务
    "StreamingMovies",  # 电影流媒体服务
    "Contract",  # 合同类型
    "PaperlessBilling",  # 是否无纸化账单
    "PaymentMethod",  # 付款方式
]

# ==================== 3. 定义模型参数 ====================
# XGBoost 参数
xgb_params = {
    # 目标函数：二分类逻辑回归
    "objective": "binary:logistic",
    # 评估指标：AUC（Area Under Curve）
    # AUC 越大越好，范围[0,1]，0.5表示随机猜测
    "eval_metric": "auc",
    # 学习率：控制每棵树的贡献
    "learning_rate": XGB_LR_START,
    # 最大深度：树的最大层数
    # 较小的深度防止过拟合
    "max_depth": 4,
    # 最小叶子权重：叶子节点最小样本权重和
    # 较大的值防止过拟合
    "min_child_weight": 5,
    # 行采样比例：每棵树使用的样本比例
    "subsample": 0.7,
    # 列采样比例（按树）：每棵树使用的特征比例
    "colsample_bytree": 0.5,
    # 列采样比例（按层）：每层使用的特征比例
    "colsample_bylevel": 0.6,
    # L1正则化参数：控制模型复杂度
    "reg_alpha": 1.0,
    # L2正则化参数：控制模型复杂度
    "reg_lambda": 5.0,
    # 最小分裂增益：分裂的最小损失下降
    "gamma": 0.5,
    # 随机种子：保证结果可复现
    "random_state": 42,
    # 树构建方法：hist（直方图方法，更快）
    "tree_method": "hist",
    # 设备：使用GPU加速
    "device": "cuda",
    # 并行线程数：-1表示使用所有CPU核心
    "n_jobs": -1,
    # 正样本权重：处理类别不平衡
    # 流失样本（正样本）的权重是负样本的3倍
    "scale_pos_weight": 3.0,
}

# LightGBM 参数
lgb_params = {
    # 目标函数：二分类
    "objective": "binary",
    # 评估指标：AUC
    "metric": "auc",
    # 学习率
    "learning_rate": LGB_LR_START,
    # 最大深度
    "max_depth": 12,
    # 叶子节点数：LightGBM特有的参数
    # 控制模型复杂度，值越大模型越复杂
    "num_leaves": 127,
    # 叶子节点最小样本数
    "min_child_samples": 10,
    # 行采样比例
    "subsample": 0.9,
    # 列采样比例
    "colsample_bytree": 0.3,
    # L1正则化
    "reg_alpha": 0.01,
    # L2正则化
    "reg_lambda": 0.1,
    # 最小分裂增益
    "min_split_gain": 0,
    # 随机种子
    "random_state": 123,
    # 并行线程数
    "n_jobs": -1,
    # 设备：CPU（LightGBM的GPU支持需要额外安装）
    "device": "cpu",
    # 日志级别：-1表示不输出
    "verbose": -1,
    # 正样本权重
    "scale_pos_weight": 4.0,
}

# 打印关键参数
print("\n参数配置:")
print(
    f"  XGBoost: depth={xgb_params['max_depth']}, colsample={xgb_params['colsample_bytree']}, "
    f"alpha={xgb_params['reg_alpha']}, lambda={xgb_params['reg_lambda']}"
)
print(
    f"  LightGBM: depth={lgb_params['max_depth']}, leaves={lgb_params['num_leaves']}, "
    f"colsample={lgb_params['colsample_bytree']}"
)

# ==================== 4. 预计算 Target Encoding ====================
print("\n预计算Target Encoding...")
encodings = precompute_fold_encodings(
    X, y, X_test, ALL_CAT_COLS, n_splits=6, smoothing=5
)

# ==================== 5. 初始化预测数组 ====================
# OOF(Out of Fold)预测：每折验证集的预测结果
# 用于评估模型性能和寻找最优融合权重
oof_xgb = np.zeros(len(X))  # XGBoost的OOF预测
oof_lgb = np.zeros(len(X))  # LightGBM的OOF预测

# 测试集预测：最终提交用的预测结果
test_xgb = np.zeros(len(X_test))  # XGBoost的测试集预测
test_lgb = np.zeros(len(X_test))  # LightGBM的测试集预测

# ==================== 6. 交叉验证训练 ====================
# 6折交叉验证
for fold in range(1, 7):
    print(f"\nFold {fold}")

    # 获取当前折的数据
    fold_data = encodings["fold_encodings"][fold]
    train_idx = fold_data["train_idx"]  # 训练集索引
    val_idx = fold_data["val_idx"]  # 验证集索引
    X_tr_enc = fold_data["X_tr_enc"]  # 编码后的训练集
    X_val_enc = fold_data["X_val_enc"]  # 编码后的验证集
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]  # 训练集和验证集标签

    # ==================== 6.1 训练 XGBoost ====================
    print("  [XGBoost] 训练中...")

    # 创建 DMatrix（XGBoost 的数据结构）
    # DMatrix 是 XGBoost 的内部数据格式，比 numpy 数组更高效
    dtrain = xgb.DMatrix(X_tr_enc, label=y_tr)
    dval = xgb.DMatrix(X_val_enc, label=y_val)

    # 训练模型
    xgb_model = xgb.train(
        xgb_params,  # 模型参数
        dtrain,  # 训练数据
        num_boost_round=NUM_BOOST_ROUND,  # 最大迭代次数
        evals=[(dval, "val")],  # 验证集（用于监控性能）
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,  # 早停轮数
        verbose_eval=300,  # 每300轮打印一次评估结果
        custom_metric=lambda pred, dtrain: (
            "auc",
            roc_auc_score(dtrain.get_label(), pred),
        ),  # 自定义评估指标
        callbacks=[XgbLearningRateCallback()],  # 学习率回调
    )

    # 对验证集进行预测
    oof_xgb[val_idx] = xgb_model.predict(dval)
    print(
        f"  [XGBoost] Fold {fold} AUC: {roc_auc_score(y_val, oof_xgb[val_idx]):.6f} (iter: {xgb_model.best_iteration})"
    )

    # ==================== 6.2 训练 LightGBM ====================
    print("  [LightGBM] 训练中...")

    # 创建 Dataset（LightGBM 的数据结构）
    lgb_train = lgb.Dataset(X_tr_enc, label=y_tr)
    lgb_val = lgb.Dataset(X_val_enc, label=y_val, reference=lgb_train)

    # 训练模型
    lgb_model = lgb.train(
        lgb_params,  # 模型参数
        lgb_train,  # 训练数据
        num_boost_round=NUM_BOOST_ROUND,  # 最大迭代次数
        valid_sets=[lgb_val],  # 验证集
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True
            ),  # 早停
            lgb.log_evaluation(period=300),  # 每300轮打印评估结果
            lgb_learning_rate_callback,  # 学习率回调
        ],
    )

    # 对验证集进行预测
    oof_lgb[val_idx] = lgb_model.predict(X_val_enc)
    print(
        f"  [LightGBM] Fold {fold} AUC: {roc_auc_score(y_val, oof_lgb[val_idx]):.6f} (iter: {lgb_model.best_iteration})"
    )

    # ==================== 6.3 对测试集进行预测 ====================
    X_test_enc = encodings["X_test_enc"]
    dtest = xgb.DMatrix(X_test_enc)

    # 累加每折的预测结果，最后除以折数取平均
    test_xgb += xgb_model.predict(dtest) / 6
    test_lgb += lgb_model.predict(X_test_enc) / 6

# ==================== 7. 模型评估 ====================
print("\n模型评估")

# 计算 OOF AUC（整体验证集的 AUC）
xgb_oof_auc = roc_auc_score(y, oof_xgb)
lgb_oof_auc = roc_auc_score(y, oof_lgb)
print(f"XGBoost  OOF AUC: {xgb_oof_auc:.6f}")
print(f"LightGBM OOF AUC: {lgb_oof_auc:.6f}")

# ==================== 8. 模型诊断 ====================
# 计算两个模型预测结果的相关性
correlation = np.corrcoef(oof_xgb, oof_lgb)[0, 1]
print(f"\n模型诊断:")
print(f"  XGB与LGB OOF相关性: {correlation:.4f}")

# 相关性分析
if correlation > 0.9:
    print("  警告: 相关性过高(>0.9), 两模型趋同, 融合价值有限")
elif correlation < 0.7:
    print("  警告: 相关性过低(<0.7), 可能存在模型不稳定")
else:
    print("  理想状态: 相关性在0.7-0.85范围, 模型互补性好")

# ==================== 9. 预测错误分析 ====================
print("\n预测错误分析:")

# 计算预测错误（阈值0.5）
xgb_error = (oof_xgb > 0.5) != y.values  # XGBoost预测错误
lgb_error = (oof_lgb > 0.5) != y.values  # LightGBM预测错误

# 统计各种情况
both_error = (xgb_error & lgb_error).sum()  # 两模型都错误
only_xgb_error = (xgb_error & ~lgb_error).sum()  # 仅XGBoost错误
only_lgb_error = (~xgb_error & lgb_error).sum()  # 仅LightGBM错误
both_correct = (~xgb_error & ~lgb_error).sum()  # 两模型都正确
total = len(y)

print(f"  两模型都正确: {both_correct} ({both_correct/total*100:.1f}%)")
print(f"  两模型都错误: {both_error} ({both_error/total*100:.1f}%)")
print(f"  仅XGB错误: {only_xgb_error} ({only_xgb_error/total*100:.1f}%)")
print(f"  仅LGB错误: {only_lgb_error} ({only_lgb_error/total*100:.1f}%)")

# 计算准确率
xgb_correct_rate = (~xgb_error).sum() / total * 100
lgb_correct_rate = (~lgb_error).sum() / total * 100
print(f"  XGB准确率: {xgb_correct_rate:.2f}%")
print(f"  LGB准确率: {lgb_correct_rate:.2f}%")

# 计算互补性指标
# 互补性 = 单模型错误数 / 总错误数
# 表示单模型错误中有多少可以通过融合修正
complementarity = (only_xgb_error + only_lgb_error) / (
    both_error + only_xgb_error + only_lgb_error + 1e-6
)
print(f"  互补性指标: {complementarity:.2%} (单模型错误中可被融合修正的比例)")

# ==================== 10. 寻找最优融合权重 ====================
print("\n寻找最优融合权重...")

best_auc = 0
best_weight = 0.5

# 遍历不同的权重组合
# w 是 XGBoost 的权重，(1-w) 是 LightGBM 的权重
for w in np.arange(0.3, 0.8, 0.05):
    # 加权融合
    ensemble_oof = w * oof_xgb + (1 - w) * oof_lgb

    # 计算融合后的 AUC
    auc = roc_auc_score(y, ensemble_oof)

    # 更新最优权重
    if auc > best_auc:
        best_auc = auc
        best_weight = w

    print(f"  XGB权重={w:.2f}: AUC={auc:.6f}")

# 打印最优结果
print(f"\n最优权重: XGB={best_weight:.2f}, LGB={1-best_weight:.2f}")
print(f"融合后 OOF AUC: {best_auc:.6f}")

# ==================== 11. 生成提交文件 ====================
# 使用最优权重融合测试集预测
test_preds = best_weight * test_xgb + (1 - best_weight) * test_lgb

# 创建提交文件
submission = pd.DataFrame({"id": test_ids, "Churn": test_preds})
submission.to_csv("submissions/v13_ensemble.csv", index=False)

print(f"\n提交文件已保存: submissions/v13_ensemble.csv")
print(
    f"预测分布: min={test_preds.min():.5f}, max={test_preds.max():.5f}, mean={test_preds.mean():.5f}"
)

# ==================== 12. 清理缓存 ====================
# 删除临时缓存文件
for cache_file in glob.glob("cache/te_encodings*.pkl"):
    os.remove(cache_file)
    print(f"\n已删除缓存文件: {cache_file}")
