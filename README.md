<div align="center">

# 📊 Telco Customer Churn Analysis & Machine Learning

An end-to-end customer churn analysis and machine learning framework combining **Exploratory Data Analysis**, **Analytical SQL (SQLite)**, **Feature Engineering**, and **Supervised Classification** to identify high-risk customer segments and revenue at risk.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/Model-Logistic%20Regression%20%7C%20Random%20Forest-orange)](https://scikit-learn.org/)
[![API Repository](https://img.shields.io/badge/Production%20API-FastAPI-009688)](https://github.com/mehtamanish9/churn-prediction-api)

</div>

---

## 📌 Executive Summary

Customer acquisition costs significantly outpace retention costs in telecom. Analyzing **7,043 customers across 21 account, demographic, and service attributes**, this project builds a complete predictive pipeline to detect churn early and quantify revenue at risk.

### 💡 Key Findings
* 📅 **Contract Duration is the #1 Predictor**: Month-to-month contracts experience a **42.7% churn rate**, compared to just **11.3%** for 1-year contracts and **2.8%** for 2-year contracts.
* 💳 **Payment Friction**: Customers paying via **Electronic Check** churn at **45.3%**, more than double the churn rate of automated credit card or bank transfer payments (~16–18%).
* ⏳ **The 12-Month Onboarding Window**: Over **53% of all churn events occur within the first 12 months** of tenure. After 24 months, customer retention stabilizes substantially.
* 🎯 **High-Risk Persona**: Customers with `Month-to-month Contract` + `Electronic Check` + `Fiber Optic Internet` + `Tenure < 12 months` form the highest concentration of lost monthly recurring revenue (MRR).

---

## 🛠️ Tech Stack & Methodology

* **Languages & Tools**: Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, SQLite, Jupyter Notebooks
* **Modeling**: Logistic Regression, Random Forest, Stratified K-Fold validation, Class Weighting (`balanced`)
* **Evaluation Metrics**: ROC-AUC, Precision, Recall, F1-Score, Confusion Matrices

---

## 📂 Project Structure & Notebook Workflow

```text
Customer_Churn_analysis/
├── 01_eda_cleaning.ipynb         # Data cleaning, type conversion, missingness & distribution EDA
├── 02_Sql.ipynb                  # SQLite database querying: CTEs, window functions & revenue-at-risk
├── 03_feature_engineering.ipynb  # One-hot encoding, tenure binning, spend aggregations & scaling
├── 04_modeling.ipynb             # Stratified train/test split, model training & ROC-AUC comparison
├── telco_churn.csv               # Raw dataset (7,043 rows × 21 columns)
├── telco_churn_cleaned.csv       # Cleaned dataset with corrected dtypes
├── telco_churn_features.csv      # Engineered feature dataset ready for modeling
├── requirements.txt              # Pinned dependencies
├── LICENSE                       # MIT License
└── README.md                     # Project documentation
```

---

## 🔍 SQL Analytical Layer

In `02_Sql.ipynb`, SQL was leveraged directly on the cleaned customer base to simulate business stakeholder reporting:

```sql
-- High-Risk Segment Identification & Revenue-at-Risk
WITH customer_risk_profile AS (
    SELECT
        customerID,
        MonthlyCharges,
        TotalCharges,
        tenure,
        Contract,
        PaymentMethod,
        CASE 
            WHEN Contract = 'Month-to-month' 
             AND PaymentMethod = 'Electronic check' 
             AND tenure <= 12 THEN 'High Risk'
            WHEN Contract = 'Month-to-month' THEN 'Medium Risk'
            ELSE 'Low Risk'
        END AS risk_tier
    FROM telco_customers
)
SELECT
    risk_tier,
    COUNT(*) AS customer_count,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_spend,
    ROUND(SUM(MonthlyCharges), 2) AS monthly_revenue_at_risk,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM customer_risk_profile), 2) AS pct_of_base
FROM customer_risk_profile
GROUP BY risk_tier
ORDER BY monthly_revenue_at_risk DESC;
```

---

## 📈 Model Performance & Evaluation

| Model | ROC-AUC | Recall (Churn = 1) | Interpretability | Business Decision |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression** | **0.842** | **79.4%** | ⭐⭐⭐⭐⭐ High | **Selected**: Direct coefficient explanations, robust baseline |
| **Random Forest** | **0.843** | 76.8% | ⭐⭐⭐ Medium | Similar ROC-AUC, higher complexity with marginal gain |

> **Selection Rationale**: Both models converged around **~0.84 ROC-AUC**. Logistic Regression with balanced class weights was chosen for production deployment due to its linear interpretability (odds ratios for stakeholder presentations) and sub-millisecond inference latency.

---

## ⚡ Production Deployment

For the real-time API implementation of this model, check out the companion repository:  
👉 **[mehtamanish9/churn-prediction-api](https://github.com/mehtamanish9/churn-prediction-api)** (FastAPI, Pydantic, automated inference schema).

---

## 🚀 Local Setup & Reproduction

1. **Clone the repo**:
   ```bash
   git clone https://github.com/mehtamanish9/Customer_Churn_analysis.git
   cd Customer_Churn_analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks**:
   Launch Jupyter and step through the numbered notebooks (`01` $\rightarrow$ `04`).

---

## 👨‍💻 Author

**Manish Mehta**  
* GitHub: [@mehtamanish9](https://github.com/mehtamanish9)  
* LinkedIn: [linkedin.com/in/manish-mehta04](https://www.linkedin.com/in/manish-mehta04)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
