# 🏦 Loan Approval Prediction — Industry-Level Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-orange?style=flat&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=flat&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-90.2%25-brightgreen?style=flat)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9665-brightgreen?style=flat)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat)

> An end-to-end machine learning pipeline to predict loan approval status — from raw data cleaning to a live deployed web application.

🔗 **Live App:** [Click here to try the app](https://loan-approval-app-8tmlykwxxpesbv4knsykbt.streamlit.app/)

---

## 📌 Problem Statement

Financial institutions process thousands of loan applications every day. Manual evaluation is slow, inconsistent, and prone to human bias.

**Goal:** Build a robust, interpretable ML pipeline to predict whether a loan application will be **Approved ✅** or **Rejected ❌** based on applicant demographics, financial history, and loan details.

**Target Variable:** `loan_status` → `1 = Approved` | `0 = Rejected`

---

## 🎯 Project Highlights

| Metric | Value |
|--------|-------|
| 🎯 Model Accuracy | **90.16%** |
| 📈 ROC-AUC Score | **0.9665** |
| 🗃️ Dataset Size | **45,000 records** |
| 🤖 Models Compared | **5 models** |
| 🌐 Deployment | **Live on Streamlit** |

---


## 📁 Project Structure
```
loan_app/
├── app.py                  ← Streamlit web app
├── requirements.txt        ← Python dependencies
├── saved_model/
│   ├── best_model.pkl      ← Trained ML model
│   ├── scaler.pkl          ← Feature scaler
│   ├── feature_names.pkl   ← Feature list
│   └── label_mappings.pkl  ← Categorical encodings
```
---

## 🔄 End-to-End Workflow

| Step | What Was Done |
|------|---------------|
| 1. Data Loading | Loaded CSV, inspected shape, dtypes, and statistics |
| 2. Data Cleaning | Fixed missing values, removed duplicates, capped outliers (IQR) |
| 3. EDA | Univariate, bivariate, multivariate analysis + correlation heatmap |
| 4. Feature Engineering | Created 5 domain-driven features |
| 5. Encoding | Label encoding for all categorical variables |
| 6. Log Transform | Applied log1p to skewed financial features |
| 7. Scaling | StandardScaler on all numeric features |
| 8. Model Training | Trained and compared 5 classification models |
| 9. Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| 10. Hyperparameter Tuning | RandomizedSearchCV + GridSearchCV |
| 11. Feature Importance | Identified top business-relevant predictors |
| 12. Deployment | Shipped as a live Streamlit web app |

---

## 📊 Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| `person_age` | Numeric | Age of the applicant |
| `person_gender` | Categorical | Gender |
| `person_education` | Categorical | Highest education level |
| `person_income` | Numeric | Annual income ($) |
| `loan_amnt` | Numeric | Requested loan amount ($) |
| `loan_intent` | Categorical | Purpose of the loan |
| `loan_int_rate` | Numeric | Interest rate (%) |
| `loan_percent_income` | Numeric | Loan as % of annual income |
| `cb_person_cred_hist_length` | Numeric | Credit history length (years) |
| `credit_score` | Numeric | Applicant credit score |
| `previous_loan_defaults_on_file` | Categorical | Prior loan default history |

---

## ⚙️ Feature Engineering

Five new features were created based on domain knowledge:

| New Feature | Formula | Why It Matters |
|-------------|---------|----------------|
| `debt_to_income_ratio` | `loan_amnt / (income + 1)` | Higher = more financial burden = higher risk |
| `income_per_age` | `income / (age + 1)` | Measures financial maturity relative to age |
| `loan_to_credit_hist` | `loan_amnt / (credit_hist + 1)` | Loan size vs credit experience |
| `is_high_loan` | `1 if loan > 75th percentile` | Flags large loan requests |
| `is_high_income` | `1 if income > median` | Flags above-median earners |

---

## 🤖 Models Trained & Compared

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline linear model |
| Decision Tree | Simple interpretable model |
| ✅ **Random Forest** | **Best model — selected for deployment** |
| Gradient Boosting | Sequential boosting model |
| XGBoost | Optimized gradient boosting |

All models trained with `class_weight='balanced'` to handle class imbalance (22% approval rate).

---

## 📈 Final Model Results — Random Forest

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | **90.16%** |
| **ROC-AUC** | **0.9665** |
| F1-Score (Approved) | 0.80 |
| F1-Score (Rejected) | 0.93 |
| Precision (Approved) | 0.73 |
| Recall (Approved) | 0.89 |

### Classification Report
precision    recall  f1-score   support

       0       0.97      0.90      0.93      7000
       1       0.73      0.89      0.80      2000

accuracy                           0.90      9000

---

## 💡 Business Insights

| # | Insight | Business Impact |
|---|---------|----------------|
| 1 | **Previous loan default** is the strongest predictor | Prior defaulters are rejected at a very high rate |
| 2 | **Loan-to-income ratio** is the top numeric signal | Borrowing >50% of annual income = high risk |
| 3 | **Credit score** clearly separates approvals from rejections | Higher score = significantly better approval odds |
| 4 | **Credit history length** correlates with approval | Longer history = more reliable repayment patterns |
| 5 | **Loan intent** influences approval rates | Education and medical loans carry different risk profiles |

---

## 🚀 Run This Project Locally

**Step 1 — Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/loan-approval-prediction.git
cd loan-approval-prediction
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Launch the app**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` ✅

---

## 🌐 Deploy on Streamlit (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **New app** → select your repo → set main file as `app.py`
5. Click **Deploy** — you get a free public URL instantly ✅

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | ML models, preprocessing, evaluation |
| XGBoost | Gradient boosting model |
| Matplotlib & Seaborn | Visualization |
| Joblib | Model saving and loading |
| Streamlit | Web app deployment |

---

## 📦 requirements.txt


---

## 🔮 Future Improvements

| Enhancement | Description |
|-------------|-------------|
| SMOTE | Oversample minority class for better balance |
| SHAP Values | Add local + global model explainability |
| MLflow | Track experiments and model versions |
| FastAPI | Build a REST API for banking system integration |
| Docker | Containerize the app for cloud deployment |
| Drift Monitoring | Detect data drift in production with Evidently AI |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋 Author

**Your Name**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/sunilkulali)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/SunilKulali)
[![Live App](https://img.shields.io/badge/Live_App-Try_Now-red?style=flat&logo=streamlit)](https://loan-approval-app-8tmlykwxxpesbv4knsykbt.streamlit.app/)

---

⭐ **If you found this project helpful, please give it a star!** ⭐

## 🚀 Deploy to Streamlit Community Cloud (Free)

1. Upload this entire folder to a **GitHub repository**
2. Go to https://share.streamlit.io
3. Sign in with GitHub
4. Click **New app** → select your repo → select `app.py`
5. Click **Deploy**

Your app will be live at: `https://yourname-appname.streamlit.app`

## 💻 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
