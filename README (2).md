# Telco Customer Churn Prediction

Predicting customer churn for a telecom company using classical ML, with an integrated SQL analysis layer for business-driven insights.

## Overview

This project analyzes the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle) to identify customers likely to churn and surface the business drivers behind churn. It combines:

- **Data cleaning & EDA** to understand churn patterns
- **SQL analysis (SQLite)** for business-style querying — churn rate by segment, revenue at risk, customer risk profiling
- **Feature engineering** for model-ready data
- **Model comparison** across Logistic Regression and Random Forest

## Dataset

7,043 customers, 21 features covering demographics, account information, and subscribed services. Target variable: `Churn` (Yes/No).

## Project Structure

```
telco-churn-prediction/
├── data/
│   ├── raw/                          # original Kaggle CSV
│   └── telco_churn_cleaned.csv
├── notebooks/
│   ├── 01_eda_cleaning.ipynb
│   ├── 02_sql_practice.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
├── telco_churn.db
├── model_comparison.csv
├── README.md
└── requirements.txt
```

## Approach

**1. Data Cleaning**
- Fixed `TotalCharges` dtype (stored as object due to blank strings for zero-tenure customers)
- Handled missing values, checked/dropped duplicates
- Verified class distribution (~73% no-churn / 27% churn — moderately imbalanced)

**2. Exploratory Data Analysis**
- Univariate and bivariate analysis of numeric features (tenure, MonthlyCharges, TotalCharges) against churn
- Categorical breakdowns (Contract, InternetService, PaymentMethod) against churn
- Correlation and outlier checks

**3. SQL Analysis (SQLite)**
Practiced and applied production-style SQL directly on the cleaned dataset, including:
- Aggregate churn rate by Contract type, PaymentMethod, tenure bucket
- Revenue-at-risk calculation (monthly/annual charges tied to churned customers)
- Window functions: `RANK()`, `NTILE()`, `LAG()`, running totals via `SUM() OVER()`
- CTEs (single and chained) for readable multi-step logic
- Subqueries and `HAVING` for segment-level filtering
- High-risk customer segmentation (month-to-month + electronic check + low tenure)

**4. Feature Engineering**
- Encoded binary and multi-category categorical variables (label mapping + one-hot encoding)
- Created derived features: `AvgMonthlySpend`, `TenureGroup` buckets
- Prepared separate scaled/unscaled feature sets for linear vs. tree-based models

**5. Modeling**
- Stratified train/test split to preserve class balance
- Logistic Regression with `class_weight='balanced'` (no SMOTE needed)
- Random Forest with `class_weight='balanced'`
- Evaluated via ROC-AUC, precision/recall, and confusion matrices

## Results

| Model               | ROC-AUC |
|---------------------|---------|
| Logistic Regression | 0.8417  |
| Model               | ROC-AUC |
|---------------------|---------|
| Random Forest        | 0.8427  |
| Logistic Regression | 0.8425  |

Both models converge around **0.84 ROC-AUC** — the ~0.0002 gap between them is negligible, well within normal run-to-run variance. Added model complexity (Random Forest) doesn't meaningfully improve performance over a linear model. **Logistic Regression was selected as the final model** for its comparable performance, interpretability, and ease of explaining to business stakeholders.

## Key Insights

- Month-to-month contracts have substantially higher churn rates than one- or two-year contracts
- Customers paying via electronic check churn more than those on automatic payment methods
- Churn risk is highest in the first 12 months of tenure and drops sharply after
- A combined risk profile (month-to-month + electronic check + tenure < 12 months) identifies a small but high-value at-risk segment worth targeted retention efforts

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `SQLite` · `matplotlib` / `seaborn`

## Setup

```bash
pip install -r requirements.txt
```

Run notebooks in order: `01_eda_cleaning.ipynb` → `02_sql_practice.ipynb` → `03_feature_engineering.ipynb` → `04_modeling.ipynb`
