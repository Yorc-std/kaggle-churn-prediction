# Kaggle Playground Series S6E3 - Customer Churn Prediction

A machine learning solution for predicting customer churn in the telecommunications industry.

## 📊 Leaderboard Progress

| Version | Features | Model | CV AUC | Public LB | Improvement | Date |
|---------|----------|-------|--------|-----------|-------------|------|
| Baseline | LabelEncoder | LightGBM | 0.9164 | 0.91316 | - | 2026-03-05 |
| V1 | Target Encoding + 5 interactions | LightGBM | 0.91603 | 0.91359 | +0.00043 | 2026-03-06 |
| V2 | Keep original numerical + core features | LightGBM | 0.91612 | 0.91370 | +0.00011 | 2026-03-06 |
| V2 XGBoost | Same as V2, switch to XGBoost | XGBoost | 0.91638 | 0.91393 | +0.00023 | 2026-03-06 |
| **V2 XGBoost + Pseudo** | **+ Pseudo Labels (threshold=0.95)** | **XGBoost** | **0.91640** | **0.91401** | **+0.00008** | **2026-03-06** |

**Current Rank**: ~540 / 827+ (Top 65%)  
**Gap to 1st Place**: 0.00318 (0.35%)  
**Total Improvement**: +0.00085 (0.09%)

### Failed Experiments
- V3: 17 advanced features → LB 0.91335 (overfitting)
- V4: Top 8 features only → LB 0.91331 (still worse)
- V5: ORIG_proba features → LB 0.91381 (original dataset too small)
- V6: Full training (3000+ rounds) → LB 0.91393 (no improvement)

## 🎯 Competition Info

- **Platform**: [Kaggle Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3)
- **Task**: Binary Classification (Customer Churn Prediction)
- **Metric**: ROC AUC
- **Deadline**: 2026-03-31
- **Dataset**: 594K training samples, 255K test samples, 19 features

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt
```

### Download Data

```bash
# Using Kaggle API
kaggle competitions download -c playground-series-s6e3 -p data/
cd data && unzip playground-series-s6e3.zip && cd ..
```

### Run Baseline

```bash
python src/baseline.py
```

## 📁 Project Structure

```
kaggle-churn-s6e3/
├── README.md                 # Project overview
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── data/                    # Data files (not tracked)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── src/                     # Source code
│   ├── baseline.py          # Baseline model
│   ├── feature_engineering.py
│   ├── models.py
│   └── utils.py
├── notebooks/               # Jupyter notebooks
│   └── 01_eda.ipynb
├── experiments/             # Experiment scripts
├── submissions/             # Submission files
└── docs/                    # Documentation
    └── experiment_log.md
```

## 🔍 Dataset Overview

**Features:**
- **Categorical (15)**: gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaymentMethod, PaperlessBilling
- **Numerical (4)**: SeniorCitizen, tenure, MonthlyCharges, TotalCharges
- **Target**: Churn (Yes/No)

**Class Distribution:**
- Churn Rate: 22.52%
- No Missing Values

## 📚 Learning Resources

This project is built upon insights from:
- **Feature Engineering for Machine Learning** by Alice Zheng & Amanda Casari
- **Hands-On Machine Learning** by Aurélien Géron
- **Kaggle竞赛宝典** (Kaggle Competition Guide)

## 🔑 Key Insights

### What Worked ✅
1. **Simplicity > Complexity**: Core features (24) outperformed advanced features (36+)
2. **XGBoost > LightGBM**: +0.00023 improvement just by switching models
3. **Pseudo Labels**: Effective with threshold=0.95 (47.9% of test set)
4. **Target Encoding**: More effective than One-Hot for categorical features
5. **Keep Original Numerical Features**: Discretization loses information

### What Didn't Work ❌
1. **ORIG_proba Features**: Original dataset too small (7K vs 594K), distribution mismatch
2. **Over-Engineering**: Adding 17 advanced features introduced noise
3. **Over-Training**: 3000+ rounds improved CV but not LB (overfitting)
4. **Lower Pseudo Threshold**: 0.90 had best CV but worse LB than 0.95

### Lessons Learned 📚
1. **Don't over-optimize CV scores**: Best CV ≠ Best LB (balance is key)
2. **Feature quality > quantity**: 24 good features > 36 noisy features
3. **Early stopping is crucial**: Prevents overfitting while allowing sufficient training
4. **Playground-specific tricks don't always work**: ORIG_proba failed due to dataset size
5. **Stratified K-Fold CV is reliable**: OOF and LB gap stable at ~-0.0024

## 🎯 Goals

- ✅ **Short-term**: Top 70% (AUC > 0.914) - **Achieved: 0.91401**
- ⏳ **Mid-term**: Top 50% (AUC > 0.916) - **Current: Top 65%**
- 🎯 **Long-term**: Top 30% (AUC > 0.920)

## 📈 Next Steps (If Continuing)

### Potential Improvements
1. **Model Ensemble**: LightGBM + XGBoost weighted average
2. **More Feature Interactions**: All feature pairs (systematic approach)
3. **Numerical Transformations**: Log1p, Sqrt, Rank (from top notebooks)
4. **10-Fold CV**: More stable evaluation (but 2x training time)
5. **Inner KFold Target Encoding**: Prevent leakage more rigorously

### Analysis from Top Performers (CV 0.917+)
- **Key difference**: 60+ features vs our 24
- **Numerical transformations**: Log1p, Sqrt, Rank, Frequency encoding
- **GPU acceleration**: 10-100x faster training
- **Extreme patience**: 50,000 n_estimators with early_stopping=500
- **Lower learning rate**: 0.005 vs our 0.01

## 📝 Experiment Log

See [docs/experiment_log.md](docs/experiment_log.md) for detailed experiment records.

## 🤝 Contributing

This is a personal learning project, but suggestions and discussions are welcome!

## 📄 License

MIT License - Feel free to use this code for learning purposes.

## 🙏 Acknowledgments

- Kaggle for hosting the competition
- The authors of the learning resources mentioned above
- The Kaggle community for sharing insights

---

**Author**: York  
**Started**: March 5, 2026  
**Status**: Active Development
