# 项目进度

## ✅ Phase 1: 环境搭建和数据探索（已完成）
- [x] 创建项目目录结构
- [x] 安装 Kaggle CLI 和数据科学库
- [x] 配置 Kaggle API 认证
- [x] 下载比赛数据（train.csv, test.csv, sample_submission.csv）
- [x] 初步数据探索
  - 训练集：594,194 行 × 21 列
  - 测试集：254,655 行 × 20 列
  - 无缺失值
  - 流失率：22.52%
  - 15 个类别特征 + 4 个数值特征

## ✅ Phase 2: Baseline 模型（已完成）
- [x] 简单特征工程（LabelEncoder 编码类别特征）
- [x] LightGBM 分类器
- [x] 验证集 AUC: **0.9164**
- [x] 首次提交
  - Public Score: **0.91316**
  - 排名: **577 / 827**（前 70%）
  - 提交时间: 2026-03-05 18:06

## 🔄 Phase 3: 特征工程（进行中）
### 待尝试的方向：
1. **数值特征变换**
   - tenure 分箱（新客户/老客户）
   - MonthlyCharges / TotalCharges 比值
   - 标准化/归一化

2. **类别特征优化**
   - Target Encoding（用目标变量编码）
   - Frequency Encoding（频率编码）
   - 交叉特征（如 Contract × InternetService）

3. **业务特征**
   - 服务数量统计（有多少增值服务）
   - 客户价值分层（RFM 模型思路）
   - 合约类型 × 付款方式交互

4. **原始数据集融合**
   - 比赛说明提到可以用原始 IBM 数据集
   - 探索合成数据和原始数据的差异

## ⏳ Phase 4: 模型优化（待开始）
1. **超参数调优**
   - Optuna / GridSearch
   - 学习率、树深度、叶子数等

2. **模型融合**
   - XGBoost + LightGBM + CatBoost
   - Stacking / Blending

3. **交叉验证**
   - 5-Fold CV 确保稳定性

## ⏳ Phase 5: 高分方案学习（待开始）
- 研究 Code 页面的高分 Notebook
- 学习 Discussion 中的技巧分享

## 目标
- 短期：进入前 50%（AUC > 0.916）
- 中期：进入前 20%（AUC > 0.917）
- 长期：冲击前 10（AUC > 0.9171）

## 参考资源
- 比赛链接: https://www.kaggle.com/competitions/playground-series-s6e3
- 原始数据集: https://www.kaggle.com/datasets/thedrzee/customer-churn-in-telecom-sample-dataset-by-ibm
