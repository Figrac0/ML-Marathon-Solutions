# Task 1: Christmas Tree Survival Prediction

<div align="center">
  <a href="https://colab.research.google.com/drive/1V_3PW32s4i6m9AQcWBq-M5eouJZ9zA3A?usp=sharing" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab">
    <br>
    <strong>Click to open in Google Colab</strong>
  </a>
</div>

## Problem Statement

The task is to predict the probability that a Christmas tree in an apartment will survive until January 18th. The prediction is based on apartment characteristics and tree maintenance conditions. The dataset contains information from 30,000 apartments for training and 18,000 apartments for testing.

### Objective
Predict probability `survived_to_18jan = 1` (float between 0.0 and 1.0) for each apartment in the test set.

### Evaluation Metric
ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

## Dataset Overview

### Data Dimensions
- Training set: 30,000 samples × 30 features
- Test set: 18,000 samples × 29 features (no target)
- Target variable: Binary (0 = did not survive, 1 = survived)

### Key Features
- **Apartment characteristics**: building age, wing, floor, area, ceiling height, window quality, heating type
- **Environmental factors**: room temperature, humidity, ventilation frequency, radiator distance
- **Household factors**: children count, cat presence, robot vacuum
- **Tree characteristics**: species, height, form, stand type, cutting date, potting status
- **Maintenance factors**: watering frequency, mist spraying, garland type and hours, ornaments weight

### Data Quality
- Target distribution: 48.06% survived, 51.94% did not survive
- Missing values present (5-7% in humidity, watering frequency, ornaments weight, garland hours)
- Categorical features: 8 columns (wing, window_quality, heating_type, tree_species, tree_form, stand_type, tinsel_level)

## Methodology

### 1. Baseline Modeling
- **Algorithm**: CatBoostClassifier (gradient boosting optimized for categorical data)
- **Validation**: 5-fold Stratified Cross-Validation
- **Configuration**:
  - Iterations: 6000
  - Learning rate: 0.03
  - Depth: 7
  - L2 regularization: 4.0
  - Early stopping with 300 iterations patience

### 2. Model Comparison
Tested four different configurations:
- **A_baseline**: Standard configuration (best performer)
- **C_deeper_reg**: Deeper trees with stronger regularization
- **D_shallower_fast**: Shallower trees with faster learning
- **E_balanced**: Balanced configuration with Bayesian bootstrapping

### 3. Ensemble Approach
- **Linear blending**: Weighted combination of Model A and Model B
- **Optimal weight search**: Grid search over 101 weights (0.0 to 1.0)
- **Best blend**: 77% Model A + 23% Model B

### 4. Feature Engineering
Created 17 new features:
- **Binning**: Floor, apartment area, temperature, humidity, tree height, garland hours, cutting days, watering frequency
- **Categorical interactions**: Wing × window quality, heating × window quality, stand type × potted tree, species × form, tinsel × garland, cat × children
- **Numerical transformations**: Ornaments weight per height, inverse radiator distance, temperature minus humidity ratio

## Results

### Cross-Validation Performance

| Model | OOF AUC | Mean Fold AUC | Std Dev |
|-------|---------|---------------|---------|
| Baseline CatBoost | 0.67202 | 0.67212 | 0.00331 |
| Model B (Regularized) | 0.67038 | 0.67052 | 0.00311 |
| Best Blend (77%A + 23%B) | 0.67211 | - | - |
| With Feature Engineering | 0.67271 | 0.67282 | 0.00323 |

### Key Findings
1. Feature engineering provided a modest but consistent improvement (+0.00069 AUC)
2. The baseline configuration performed best among individual models
3. Ensemble blending offered marginal improvement over single models
4. Model stability was good with standard deviation < 0.0035 across folds

## Technical Implementation

### Dependencies
- Python 3.7+
- catboost 1.2.8
- pandas 2.2.2
- numpy 1.26.4
- scikit-learn 1.4.2

### Code Structure
1. **Data Loading & Validation**: Load datasets and verify integrity
2. **Preprocessing**: Handle categorical features and missing values
3. **Cross-Validation Setup**: 5-fold stratified splitting
4. **Model Training**: CatBoost with early stopping
5. **Prediction & Submission**: Generate probability predictions and format output

### Validation Strategy
- Fixed random seed (42) for reproducibility
- Stratified K-Fold to preserve class distribution
- Out-of-Fold predictions for reliable performance estimation
- Assertions to ensure data integrity throughout pipeline

## Files

### Input Data
- `train.csv`: Training data with target
- `test.csv`: Test data without target
- `sample_submission.csv`: Submission format template

### Output Files
- `submission_catboost_baseline.csv`: Baseline model predictions
- `submission_best_blend.csv`: Ensemble model predictions
- `submission_best_single_model.csv`: Best single model predictions
- `submission_catboost_stable_fe.csv`: Best overall predictions with feature engineering

### Code Files
- `Codemrock_1.ipynb`: Jupyter notebook with complete analysis
- `codemrock_1.py`: Python script version

- ### Current Limitations

1. **Feature Engineering Scope**  
   Feature engineering was applied globally across the entire dataset rather than within cross-validation folds. This approach may introduce subtle data leakage, as feature transformations use information from both training and validation sets during cross-validation.

2. **Hyperparameter Optimization**  
   Limited exploration of hyperparameter space. The current configuration was selected based on initial experimentation rather than systematic optimization, potentially leaving performance improvements unexplored.

3. **Algorithm Diversity**  
   The solution relies exclusively on CatBoost algorithm. While CatBoost is well-suited for tabular data with categorical features, ensemble approaches combining multiple algorithms could provide better generalization and robustness.

4. **Missing Value Handling**  
   Missing values are handled implicitly by CatBoost's internal mechanisms. More sophisticated imputation strategies (multiple imputation, predictive modeling of missing values) were not explored.

5. **Categorical Feature Encoding**  
   Categorical features are processed using CatBoost's native handling. Alternative encoding methods (target encoding, frequency encoding) with proper cross-validation were not implemented.
