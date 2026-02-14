# Credit Card Fraud Detection (Imbalanced) — Random Forest (PR-AUC)

## Overview
This project detects fraudulent credit card transactions in a highly imbalanced dataset.  
The primary evaluation metric is **PR-AUC (Average Precision)**, which is more informative than accuracy for rare-event detection.

## Key points
- Model: **Random Forest**
- Imbalance handling: **SMOTE** (compared against `class_weight="balanced"`)
- Validation: **Stratified 5-Fold Cross-Validation**
- Metrics: **PR-AUC (primary)** and **ROC-AUC (secondary)**

## Repository structure
- `src/` — Python script to train and evaluate the model
- `notebooks/` — Kaggle notebook with EDA and experiments

## Dataset
Kaggle: **Credit Card Fraud Detection** (mlg-ulb/creditcardfraud)  
> Dataset files are not included in this repository.

## How to run
### Option A — Kaggle (recommended)
1. Create a Kaggle Notebook and add the dataset `mlg-ulb/creditcardfraud`.
2. Run the notebook in `notebooks/`.

### Option B — Local (if you have the dataset file)
1. Place `creditcard.csv` in the project root (or adjust the path in the script).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
