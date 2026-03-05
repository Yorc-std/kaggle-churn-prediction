# Experiment Log

## Baseline (2026-03-05)

### Configuration
- **Model**: LightGBM Classifier
- **Features**: 
  - Categorical: LabelEncoder (15 features)
  - Numerical: Raw values (4 features)
- **Validation**: Single 80/20 train-test split
- **Hyperparameters**: Default LightGBM settings

### Results
- **Validation AUC**: 0.9164
- **Public LB AUC**: 0.91316
- **Rank**: 577 / 827 (Top 70%)
- **Training Rounds**: 657 (early stopping)

### Observations
- Simple LabelEncoder works but leaves room for improvement
- No cross-validation - validation score might be unstable
- Gap to 1st place: 0.004 (very small, achievable with better features)

### Next Steps
1. Implement Target Encoding for categorical features
2. Add interaction features (Contract × InternetService, etc.)
3. Switch to 5-fold Stratified CV for stable validation
4. Feature binning for numerical features

---

## v1: Target Encoding (TBD)

### Plan
- Replace LabelEncoder with Target Encoding
- Implement proper CV-based encoding (avoid leakage)
- Expected improvement: +0.005-0.01 AUC

### Configuration
- TBD

### Results
- TBD

---

## v2: Cross Features (TBD)

### Plan
- Add interaction features based on business logic
- Bin numerical features and create cross features
- Expected improvement: +0.003-0.008 AUC

### Configuration
- TBD

### Results
- TBD

---

## v3: Model Stacking (TBD)

### Plan
- Train multiple models (LightGBM, XGBoost, CatBoost)
- Implement proper stacking with OOF predictions
- Expected improvement: +0.003-0.008 AUC

### Configuration
- TBD

### Results
- TBD
