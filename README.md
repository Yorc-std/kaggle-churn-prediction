# Kaggle Playground Series S6E3 - Customer Churn Prediction

A machine learning solution for predicting customer churn in the telecommunications industry.

## 📊 Leaderboard Progress

| Version | Features | Model | CV AUC | Public LB | Rank | Date |
|---------|----------|-------|--------|-----------|------|------|
| Baseline | LabelEncoder | LightGBM | 0.9164 | 0.91316 | 577/827 (70%) | 2026-03-05 |
| v1 | TBD | TBD | TBD | TBD | TBD | TBD |

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

1. **Feature Engineering > Model Selection**: 80% of improvement comes from better features
2. **Target Encoding**: More effective than One-Hot for high-cardinality categorical features
3. **Cross Features**: Business logic often hides in feature interactions
4. **Stratified K-Fold CV**: Essential for reliable validation
5. **Leakage Prevention**: Always compute statistics within CV folds

## 📈 Roadmap

### Phase 1: Feature Engineering
- [ ] Implement Target Encoding (avoid leakage)
- [ ] Create interaction features (Contract × InternetService, etc.)
- [ ] Binning numerical features (tenure, MonthlyCharges)
- [ ] Feature selection

### Phase 2: Validation Strategy
- [ ] Switch to 5-fold Stratified CV
- [ ] Compare CV scores with LB scores
- [ ] Detect overfitting

### Phase 3: Model Ensemble
- [ ] Train multiple models (LightGBM, XGBoost, CatBoost)
- [ ] Implement proper stacking (OOF predictions)
- [ ] Hyperparameter tuning

### Phase 4: Final Push
- [ ] Feature importance analysis
- [ ] Model interpretation
- [ ] Documentation

## 🎯 Goals

- **Short-term**: Top 50% (AUC > 0.916)
- **Mid-term**: Top 30% (AUC > 0.920)
- **Long-term**: Top 10% (AUC > 0.925)

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
