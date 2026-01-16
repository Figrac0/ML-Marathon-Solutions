# Text Classification with TF-IDF and Logistic Regression

## Overview
A machine learning pipeline for topic classification of chat messages using TF-IDF text vectorization and Logistic Regression.

## Data Processing
- **Dataset Structure**: Two datasets (train/test) containing chat messages with metadata
- **Feature Engineering**: 
  - Combined raw text with temporal features (day and hour)
  - Created `full_text` feature format: `"text <DAY_X> <HOUR_Y>"`
  - Handled missing values in text fields

## Model Architecture
### Text Vectorization
- **TF-IDF Vectorizer** with:
  - N-gram range: (1, 2) - includes unigrams and bigrams
  - Minimum document frequency: 3 terms
  - Maximum document frequency: 90%
  - Sublinear TF scaling for better weighting

### Classification Model
- **Logistic Regression** with:
  - Regularization parameter: C=6
  - Maximum iterations: 3000
  - Balanced class weighting for imbalanced data
  - Parallel processing enabled (n_jobs=-1)

### Pipeline Design
- Sequential pipeline combining TF-IDF transformation and classification
- Ensures consistent preprocessing during training and inference

## Training Strategy
- **Cross-Validation**: 5-fold stratified K-Fold
- **Random State**: 42 for reproducibility
- **Evaluation Metric**: Macro-averaged F1-score
- **Validation**: Each fold reports individual and mean performance

## Results
- Model performance measured via cross-validation F1 scores
- Final model trained on entire training dataset
- Predictions generated for test dataset

## Output
- Predictions saved in submission format with `message_id` and `topic` columns
- CSV file compatible with competition submission requirements

## Key Features
- Temporal feature integration with text data
- Robust cross-validation scheme
- Handles class imbalance through weighting
- Production-ready pipeline for consistent predictions
