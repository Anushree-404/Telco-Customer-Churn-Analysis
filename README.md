# 📊 Customer Churn Prediction & Insight Generation

A complete end-to-end analytics project that predicts customer churn for a telecom business and translates model outputs into **actionable business recommendations** — not just a prediction score.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Table of Contents
- [Problem Statement](#-problem-statement)
- [Business Objective](#-business-objective)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Tech Stack](#-tech-stack)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Modeling Approach](#-modeling-approach)
- [Results](#-results)
- [Key Business Insights](#-key-business-insights)
- [QC / Data Validation Checks](#-qc--data-validation-checks)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🎯 Problem Statement

Customer churn — when a customer stops using a company's service — directly hurts recurring revenue. Acquiring a new customer costs **5–25x more** than retaining an existing one, yet most businesses only find out a customer is unhappy *after* they've already left.

This project builds a model that identifies **which customers are likely to churn, and why**, so a retention team can act early with targeted offers instead of expensive blanket discounts.

## 💼 Business Objective

> Predict customers at high risk of churn and surface the top drivers behind churn, enabling data-driven retention strategy instead of guesswork.

This mirrors real analytics-consulting deliverables — the kind of insight-to-action pipeline used by firms working with telecom, SaaS, and BFSI clients.

---

## 🗂 Dataset

- **Source:** [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** ~7,043 customer records, 21 features
- **Target variable:** `Churn` (Yes/No)
- **Key features:** `tenure`, `Contract`, `MonthlyCharges`, `TotalCharges`, `PaymentMethod`, `InternetService`, `OnlineSecurity`, `TechSupport`, `StreamingTV`, `StreamingMovies`, etc.

---

## 🔄 Project Workflow

```
Raw Data → Data Cleaning → EDA (Business Insights) → Feature Engineering
→ Model Training (Logistic Regression + Random Forest) → Evaluation
→ QC Validation → Business Insight Report
```

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Modeling | Scikit-learn (Logistic Regression, Random Forest) |
| Environment | Jupyter Notebook |

---

## 🔍 Exploratory Data Analysis

Key questions explored during EDA, each converted into a plain-language business insight:

| Question | Insight |
|---|---|
| Does contract type affect churn? | Month-to-month customers churn ~3x more than annual-contract customers |
| Does tenure affect churn? | Customers in their first 12 months churn the most |
| Does tech support reduce churn? | Customers without tech support churn ~2x more |
| Does payment method matter? | Electronic check users show higher churn than auto-pay users |

### 📸 EDA Visuals

| Chart | Description |
|---|---|
| ![Churn by Contract](screenshots/eda_01_contract.png) | Churn rate by contract type |
| ![Churn by Tenure](screenshots/eda_02_tenure.png) | Churn rate by tenure |
| ![Churn by Internet Service](screenshots/eda_03_internet.png) | Churn rate by internet service type |
| ![Churn by Payment Method](screenshots/eda_04_payment.png) | Churn rate by payment method |
| ![Churn by Add-on Services](screenshots/eda_05_addons.png) | Churn rate by add-on services (tech support, security, etc.) |
| ![Churn by Demographics](screenshots/eda_06_demographics.png) | Churn rate by demographic segments |
| ![Churn by Services Count](screenshots/eda_07_services_count.png) | Churn rate by total number of subscribed services |
| ![Correlation Heatmap](screenshots/eda_08_heatmap.png) | Feature correlation heatmap |
| ![Churn by Monthly Charges](screenshots/eda_09_charges.png) | Distribution of monthly charges by churn status |
| ![Feature Correlation with Churn](screenshots/eda_10_correlation.png) | Ranked correlation of features with churn |
| ![Numeric Feature Distributions](screenshots/dist_numeric.png) | Distribution plots for numeric features |

---

## 🤖 Modeling Approach

Three models were trained and compared, each chosen for a specific reason:

| Model | Why used |
|---|---|
| **Logistic Regression** | Interpretable coefficients — shows *how much* each factor increases/decreases churn odds, useful for explaining results to non-technical stakeholders |
| **Random Forest** | Higher predictive performance + native feature importance ranking to identify top churn drivers |
| **XGBoost** | Gradient-boosted model for a further performance benchmark, paired with SHAP for granular explainability |

**Class imbalance handling:** Used `class_weight='balanced'` (and equivalent scale-weighting for XGBoost) since churners are a minority class, and missing a real churner (false negative) is costlier to the business than a false alarm.

**Evaluation metrics used:** Precision, Recall, F1-score, and ROC-AUC were prioritized over raw accuracy, since accuracy alone is misleading on imbalanced data.

**Threshold tuning:** Instead of defaulting to a 0.5 classification threshold, a threshold sweep was performed to find the cutoff that best balances precision and recall for the business use case (see `threshold_sweep.png`).

**Explainability (SHAP):** SHAP values were used on both Random Forest and XGBoost to explain not just *which* features matter globally, but *why* any individual customer was flagged as high-risk — this is the layer that turns a prediction into a client-ready explanation.

### 📊 Model Diagnostics

![Logistic Regression Coefficients](screenshots/lr_coefs.png)
![Random Forest Feature Importance](screenshots/rf_importance.png)
![Feature Importance Comparison](screenshots/feature_importance_comparison.png)
![Confusion Matrices (All Models)](screenshots/model_confusion_matrices.png)
![ROC & PR Curves](screenshots/model_roc_pr.png)
![ROC Comparison](screenshots/roc_comparison.png)
![Threshold Sweep](screenshots/threshold_sweep.png)

### 🔬 SHAP Explainability

![SHAP - Random Forest (Bar)](screenshots/shap_rf_bar.png)
![SHAP - XGBoost (Bar)](screenshots/shap_xgb_bar.png)
![SHAP - XGBoost (Beeswarm)](screenshots/shap_xgb_beeswarm.png)
![SHAP - XGBoost (Waterfall - single prediction)](screenshots/shap_xgb_waterfall.png)

---

## 📈 Results

| Metric | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|
| Accuracy | _add your value_ | _add your value_ | _add your value_ |
| Precision | _add your value_ | _add your value_ | _add your value_ |
| Recall | _add your value_ | _add your value_ | _add your value_ |
| F1-Score | _add your value_ | _add your value_ | _add your value_ |
| ROC-AUC | _add your value_ | _add your value_ | _add your value_ |

**Top churn-driving features (via Random Forest + SHAP):**
1. Contract type
2. Tenure
3. Monthly charges
4. Tech support subscription
5. Payment method

> 📄 A full auto-generated report with all metrics and charts is available at [`telco_churn_report.html`](telco_churn_report.html).

---

## 💡 Key Business Insights

1. **Month-to-month customers churn ~3x more** than annual-contract customers → recommend incentivizing annual plan upgrades.
2. **First-year customers are highest risk** → recommend onboarding check-ins and loyalty perks in months 1–3.
3. **No tech support = ~2x higher churn** → recommend bundling free tech support for new customers.
4. **Electronic check payers churn more** → recommend nudging customers toward auto-pay.

---

## ✅ QC / Data Validation Checks

Since this project is designed to mirror a real analytics-QC workflow, the following checks were built in before finalizing results:

- ✔️ No duplicate customer records
- ✔️ No missing values remaining after cleaning
- ✔️ Churn ratio consistent across train and test splits (no sampling bias)
- ✔️ No data leakage — verified no post-outcome fields were used as features

---

## 📁 Project Structure

```
Churner/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
│
├── telco_churn_analysis.ipynb             # Main analysis notebook (EDA → modeling → SHAP)
├── run_analysis.py                        # Script version of the pipeline
├── telco_churn_report.html                # Auto-generated HTML report
│
├── screenshots/                           # All exported charts (or place in repo root)
│   ├── eda_01_contract.png
│   ├── eda_02_tenure.png
│   ├── eda_03_internet.png
│   ├── eda_04_payment.png
│   ├── eda_05_addons.png
│   ├── eda_06_demographics.png
│   ├── eda_07_services_count.png
│   ├── eda_08_heatmap.png
│   ├── eda_09_charges.png
│   ├── eda_10_correlation.png
│   ├── dist_numeric.png
│   ├── lr_coefs.png
│   ├── rf_importance.png
│   ├── feature_importance_comparison.png
│   ├── model_confusion_matrices.png
│   ├── model_roc_pr.png
│   ├── roc_comparison.png
│   ├── threshold_sweep.png
│   ├── shap_rf_bar.png
│   ├── shap_xgb_bar.png
│   ├── shap_xgb_beeswarm.png
│   └── shap_xgb_waterfall.png
│
├── requirements.txt
└── README.md
```

> 💡 Tip: create a `screenshots/` folder in your repo and move all the `.png` files into it (they're currently at the project root in your local setup) so the image links above resolve correctly on GitHub.

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/<your-username>/churn-prediction-project.git
cd churn-prediction-project

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook telco_churn_analysis.ipynb
```

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
jupyter
```

Alternatively, run the script version directly:
```bash
python run_analysis.py
```

---

## 🚀 Future Improvements

- Add behavioral data (support call frequency, app usage) — often stronger churn predictors than static demographics
- Hyperparameter tuning (GridSearchCV/Optuna) for further performance gains on XGBoost
- Deploy as a simple Streamlit dashboard so non-technical stakeholders can explore predictions and SHAP explanations interactively
- Set up model monitoring for drift once deployed, since churn patterns shift over time (pricing changes, new competitors, etc.)

---

## 👤 Author

Anushree Dhanashetti  anushreedhanashetti@gmail.com

---

⭐ If you found this project useful, consider giving it a star!
