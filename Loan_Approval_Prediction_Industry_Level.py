#!/usr/bin/env python
# coding: utf-8

# # 🏦 Loan Approval Prediction Industry-Level Machine Learning Project
# ---
# ### 📌 Problem Statement
# Financial institutions receive thousands of loan applications daily. Manual evaluation is slow, inconsistent, and biased.
# 
# **Goal:** Build a robust, interpretable ML pipeline to **predict loan approval status** based on applicant demographics, financial history, and loan details — enabling faster, fairer, data-driven lending decisions.
# 
# ### 📋 Project Workflow
# | # | Section |
# |---|---------|
# | 1 | Data Loading & Overview |
# | 2 | Data Cleaning |
# | 3 | Exploratory Data Analysis (EDA) |
# | 4 | Feature Engineering |
# | 5 | Model Building |
# | 6 | Model Evaluation |
# | 7 | Hyperparameter Tuning |
# | 8 | Feature Importance |
# | 9 | Model Comparison |
# | 10 | Business Insights |
# | 11 | Model Saving |
# | 12 | Deployment (Streamlit) |
# 
# **Target Variable:** `loan_status` — 1 = Approved, 0 = Rejected

# ##  Import Libraries & Configuration

# In[1]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
sns.set_style('whitegrid')
sns.set_palette('Set2')

# Preprocessing & Modelling
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, GridSearchCV, RandomizedSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, classification_report,
                              roc_auc_score, roc_curve, ConfusionMatrixDisplay)
import joblib, pickle, os

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    print("XGBoost loaded successfully")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed — skipping. Install with: pip install xgboost")

print("All libraries imported successfully")


# ## Data Loading & Overview
# > Loading the dataset and inspecting its shape, column types, and basic statistics to understand what we are working with.

# In[2]:


df = pd.read_csv("loan_dataset.csv")
print(f"Dataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
df.head()


# In[3]:


df.info()


# In[4]:


# Statistical summary — numeric features
df.describe().T.round(2)


# In[5]:


# Categorical features summary
df.describe(include='object').T


# In[6]:


# Target class distribution
counts = df['loan_status'].value_counts()
print(f"Approved (1): {counts.get(1, 0):,}  |  Rejected (0): {counts.get(0, 0):,}")
print(f"Approval Rate: {df['loan_status'].mean()*100:.1f}%")


# ##  Data Cleaning
# > Clean data is the foundation of every good model. We handle missing values, duplicates, irrelevant columns, and outliers.

# ### 3.1 Missing Values

# In[7]:


missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Count': missing, 'Percentage': missing_pct})
missing_df = missing_df[missing_df['Count'] > 0].sort_values('Percentage', ascending=False)

if missing_df.empty:
    print("No missing values found!")
else:
    print(missing_df)
    # Fill numerics with median (robust to skew/outliers)
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)
    # Fill categoricals with mode
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    print("Missing values imputed.")


# ### 3.2 Duplicate Records

# In[8]:


dupes = df.duplicated().sum()
print(f"Duplicate rows: {dupes}")
df.drop_duplicates(inplace=True)
print(f"Shape after removing duplicates: {df.shape}")


# ### 3.3 Drop Irrelevant Columns

# In[9]:


# person_home_ownership: low signal for approval compared to financial ratios
# person_emp_exp: highly correlated with credit_hist_length — causes multicollinearity
cols_to_drop = [c for c in ['person_home_ownership', 'person_emp_exp'] if c in df.columns]
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped: {cols_to_drop}")
print(f"Remaining features: {list(df.columns)}")


# ### 3.4 Outlier Detection & Treatment (IQR Capping / Winsorization)

# In[10]:


num_features = [c for c in df.select_dtypes(include=np.number).columns if c != 'loan_status']

# Plot before treatment
fig, axes = plt.subplots(2, (len(num_features)+1)//2, figsize=(16, 7))
axes = axes.flatten()
for i, col in enumerate(num_features):
    axes[i].boxplot(df[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor='#74b9ff', alpha=0.8))
    axes[i].set_title(col, fontsize=9, fontweight='bold')
    axes[i].tick_params(labelbottom=False)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Outlier Detection — Before Treatment', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()


# In[11]:


# Winsorize outliers (cap rather than remove — preserves data size)
def cap_outliers(data, col):
    Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    n = ((data[col] < lower) | (data[col] > upper)).sum()
    data[col] = data[col].clip(lower, upper)
    return n

print(f"{'Feature':<30} {'Outliers Capped':>15}")
print("-" * 47)
for col in num_features:
    n = cap_outliers(df, col)
    print(f"{col:<30} {n:>15,}")
print(f"\nDataset shape after cleaning: {df.shape}")


# ### 3.5 Class Imbalance Check

# In[12]:


fig, axes = plt.subplots(1, 2, figsize=(11, 4))
counts = df['loan_status'].value_counts()
labels = ['Rejected (0)', 'Approved (1)']
colors = ['#e17055', '#00b894']

axes[0].bar(labels, counts.values, color=colors, width=0.5, edgecolor='white')
axes[0].set_title('Loan Status Distribution', fontweight='bold')
axes[0].set_ylabel('Count')
for bar, val in zip(axes[0].patches, counts.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                 f'{val:,}', ha='center', fontsize=11, fontweight='bold')

axes[1].pie(counts.values, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
axes[1].set_title('Approval Rate', fontweight='bold')

plt.suptitle('Class Balance Analysis', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()

imbalance_ratio = counts.min() / counts.max()
print(f"Imbalance ratio: {imbalance_ratio:.2f}")
if imbalance_ratio < 0.5:
    print("Significant imbalance detected — consider SMOTE or class_weight='balanced'")
else:
    print("Class distribution is acceptable. Using class_weight='balanced' as a precaution.")


# ## Exploratory Data Analysis (EDA)
# > We analyse distributions, relationships, and patterns across all features to extract business-relevant insights before modelling.

# ### 4.1 Univariate Analysis — Numeric Features

# In[13]:


fig, axes = plt.subplots(3, (len(num_features)+2)//3, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(num_features):
    axes[i].hist(df[col], bins=35, color='#0984e3', edgecolor='white', alpha=0.85)
    axes[i].axvline(df[col].mean(), color='red', linestyle='--', linewidth=1.2, label='Mean')
    axes[i].axvline(df[col].median(), color='orange', linestyle='-', linewidth=1.2, label='Median')
    axes[i].set_title(col, fontweight='bold', fontsize=9)
    axes[i].legend(fontsize=7)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Univariate Distribution — Numeric Features', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()


# ### 4.2 Univariate Analysis — Categorical Features
# > Analyzing one variable/column at a time.

# In[14]:


cat_features = df.select_dtypes(include='object').columns.tolist()
fig, axes = plt.subplots(1, len(cat_features), figsize=(14, 5))
if len(cat_features) == 1:
    axes = [axes]
for ax, col in zip(axes, cat_features):
    counts = df[col].value_counts()
    ax.bar(counts.index, counts.values, color=sns.color_palette('Set2', len(counts)),
           edgecolor='white')
    ax.set_title(col, fontweight='bold')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=30)
plt.suptitle('Univariate Analysis — Categorical Features', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()


# ### 4.3 Bivariate Analysis — Feature vs Loan Status
# > Analyzing the relationship between two variables

# In[15]:


# Numeric features vs target
fig, axes = plt.subplots(2, (len(num_features)+1)//2, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(num_features):
    approved = df[df['loan_status']==1][col]
    rejected = df[df['loan_status']==0][col]
    axes[i].hist(rejected, bins=30, alpha=0.6, color='#e17055', label='Rejected', density=True)
    axes[i].hist(approved, bins=30, alpha=0.6, color='#00b894', label='Approved', density=True)
    axes[i].set_title(col, fontweight='bold', fontsize=9)
    axes[i].legend(fontsize=7)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Bivariate Analysis — Numeric Features vs Loan Status', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()


# In[16]:


# Box plots — numeric features split by loan status
fig, axes = plt.subplots(2, (len(num_features)+1)//2, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(num_features):
    data_plot = [df[df['loan_status']==0][col], df[df['loan_status']==1][col]]
    bp = axes[i].boxplot(data_plot, patch_artist=True,
                         boxprops=dict(alpha=0.8), notch=False)
    bp['boxes'][0].set_facecolor('#e17055')
    bp['boxes'][1].set_facecolor('#00b894')
    axes[i].set_xticklabels(['Rejected', 'Approved'])
    axes[i].set_title(col, fontweight='bold', fontsize=9)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Box Plots — Feature Distribution by Loan Status', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()


# ### 4.4 Categorical Features vs Loan Status

# In[17]:


for col in cat_features:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Count plot
    order = df[col].value_counts().index
    sns.countplot(data=df, x=col, hue='loan_status', order=order, ax=axes[0],
                  palette={'0':'#e17055','1':'#00b894'} if df['loan_status'].dtype==object
                  else {0:'#e17055',1:'#00b894'})
    axes[0].set_title(f'{col} — Count by Loan Status', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].legend(title='Loan Status', labels=['Rejected','Approved'])
    # Approval rate
    approval_rate = df.groupby(col)['loan_status'].mean().sort_values(ascending=False)
    axes[1].bar(approval_rate.index, approval_rate.values*100,
                color=sns.color_palette('Set2', len(approval_rate)), edgecolor='white')
    axes[1].set_title(f'{col} — Approval Rate (%)', fontweight='bold')
    axes[1].set_ylabel('Approval Rate (%)')
    axes[1].tick_params(axis='x', rotation=30)
    for bar in axes[1].patches:
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     f'{bar.get_height():.1f}%', ha='center', fontsize=9)
    plt.tight_layout(); plt.show()


# ### 4.5 Correlation Heatmap

# In[18]:


# Encode categoricals temporarily for correlation
df_enc = df.copy()
le = LabelEncoder()
for col in df_enc.select_dtypes(include='object').columns:
    df_enc[col] = le.fit_transform(df_enc[col])

corr = df_enc.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(13, 9))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, linecolor='white', vmin=-1, vmax=1,
            annot_kws={'size': 8})
plt.title('Correlation Matrix — All Features', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout(); plt.show()

# Top correlations with target
target_corr = corr['loan_status'].drop('loan_status').abs().sort_values(ascending=False)
print("\nTop Features Correlated with Loan Status:")
print(target_corr.round(3).to_string())


# ### 4.6 Multivariate Analysis — Pair Plot

# In[19]:


# Pair plot on top 5 correlated features
top5 = target_corr.head(5).index.tolist() + ['loan_status']
sample = df_enc[top5].sample(min(1500, len(df_enc)), random_state=42)
g = sns.pairplot(sample, hue='loan_status', diag_kind='kde',
                 palette={0:'#e17055', 1:'#00b894'}, plot_kws={'alpha':0.4, 's':15})
g.fig.suptitle('Pair Plot — Top 5 Features vs Loan Status', y=1.01, fontsize=13, fontweight='bold')
plt.show()


# ##  Feature Engineering
# > Create domain-driven features that improve model signal, encode categoricals, and scale numeric features.

# ### 5.1 New Feature Creation

# In[20]:


# --- Domain-based engineered features ---

# 1. Debt-to-Income ratio: higher = more financial burden = higher default risk
df['debt_to_income_ratio'] = df['loan_amnt'] / (df['person_income'] + 1)

# 2. Income per age: proxy for financial maturity relative to age
df['income_per_age'] = df['person_income'] / (df['person_age'] + 1)

# 3. Loan amount to credit history ratio: credit experience vs loan size
if 'cb_person_cred_hist_length' in df.columns:
    df['loan_to_credit_hist'] = df['loan_amnt'] / (df['cb_person_cred_hist_length'] + 1)

# 4. High loan flag: flag applicants borrowing above 75th percentile
loan_75 = df['loan_amnt'].quantile(0.75)
df['is_high_loan'] = (df['loan_amnt'] > loan_75).astype(int)

# 5. High income flag: applicant earns above median
df['is_high_income'] = (df['person_income'] > df['person_income'].median()).astype(int)

print("New features added:")
new_feats = ['debt_to_income_ratio','income_per_age','loan_to_credit_hist','is_high_loan','is_high_income']
print(df[[f for f in new_feats if f in df.columns]].describe().T.round(3))


# ### 5.2 Encode Categorical Variables

# In[21]:


cat_cols = df.select_dtypes(include='object').columns.tolist()
print(f"Categorical columns to encode: {cat_cols}")

le = LabelEncoder()
label_mappings = {}

for col in cat_cols:
    df[col] = le.fit_transform(df[col])
    label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  {col}: {label_mappings[col]}")

print("\nEncoding complete. All features are now numeric.")
df.head(3)


# ### 5.3 Log Transformation — Skewed Features

# In[22]:


# Apply log1p to heavily right-skewed financial features
skewed_cols = ['person_income', 'loan_amnt', 'loan_percent_income']
skewed_cols = [c for c in skewed_cols if c in df.columns]

for col in skewed_cols:
    df[col] = np.log1p(df[col])

print(f"Log1p transformation applied to: {skewed_cols}")


# ### 5.4 Feature Scaling

# In[23]:


X = df.drop(columns=['loan_status'])
y = df['loan_status']

feature_names = X.columns.tolist()
print(f"Total features for modelling: {len(feature_names)}")

# Train-test split (stratified to preserve class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=feature_names)

print(f"Train set: {X_train_scaled.shape} | Test set: {X_test_scaled.shape}")
print(f"Train approval rate: {y_train.mean()*100:.1f}% | Test approval rate: {y_test.mean()*100:.1f}%")


# ##  Model Building
# > We train five classification models and evaluate them systematically using a consistent evaluation framework.

# In[24]:


# Define all models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000,
                                               class_weight='balanced'),
    'Decision Tree':       DecisionTreeClassifier(random_state=42, max_depth=10,
                                                   class_weight='balanced'),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42,
                                                   max_depth=12, class_weight='balanced'),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=42,
                                                       learning_rate=0.1, max_depth=5),
}
if XGBOOST_AVAILABLE:
    models['XGBoost'] = XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss',
                                       scale_pos_weight=(y_train==0).sum()/(y_train==1).sum())

print(f"Models to train: {list(models.keys())}")


# In[25]:


# Train all models and collect results
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    print(f"Training {name}...", end=' ')
    model.fit(X_train_scaled, y_train)
    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv,
                                scoring='f1', n_jobs=-1)
    results[name] = {
        'model':     model,
        'y_pred':    y_pred,
        'y_proba':   y_proba,
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall':    recall_score(y_test, y_pred),
        'F1 Score':  f1_score(y_test, y_pred),
        'ROC-AUC':   roc_auc_score(y_test, y_proba),
        'CV F1 Mean':cv_scores.mean(),
        'CV F1 Std': cv_scores.std(),
    }
    print(f"Accuracy={results[name]['Accuracy']:.4f} | F1={results[name]['F1 Score']:.4f} | AUC={results[name]['ROC-AUC']:.4f}")

print("\nAll models trained successfully!")


# In[26]:


#!pip install --upgrade xgboost
#!pip install -U xgboost
#!pip install scikit-learn==1.5.2


# ## Model Evaluation
# > Evaluate each model with confusion matrices, classification reports, and ROC curves.

# In[27]:


# Confusion Matrices — all models
n = len(models)
fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
if n == 1: axes = [axes]
for ax, (name, res) in zip(axes, results.items()):
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, res['y_pred']),
        display_labels=['Rejected','Approved']
    ).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{name}\nF1={res["F1 Score"]:.3f}', fontweight='bold', fontsize=9)
plt.suptitle('Confusion Matrices', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()


# In[28]:


# ROC Curves — all models
plt.figure(figsize=(9, 6))
colors = ['#0984e3','#6c5ce7','#00b894','#e17055','#fdcb6e']
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['ROC-AUC']:.3f})",
             color=color, linewidth=2)
plt.plot([0,1],[0,1],'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate', fontsize=11)
plt.title('ROC Curves — All Models', fontsize=13, fontweight='bold')
plt.legend(loc='lower right', fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()


# In[29]:


# Classification report for each model
for name, res in results.items():
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(classification_report(y_test, res['y_pred'],
                                 target_names=['Rejected','Approved']))


# ## Model Comparison
# > Side-by-side comparison of all models across all key metrics.

# In[30]:


metrics = ['Accuracy','Precision','Recall','F1 Score','ROC-AUC','CV F1 Mean','CV F1 Std']
comparison_df = pd.DataFrame(
    {name: {m: res[m] for m in metrics} for name, res in results.items()}
).T.round(4)

# Highlight best values
styled = comparison_df.style.highlight_max(
    subset=['Accuracy','Precision','Recall','F1 Score','ROC-AUC','CV F1 Mean'],
    color='#b2f2bb'
).highlight_min(
    subset=['CV F1 Std'],
    color='#b2f2bb'
).format(precision=4)
print("Model Comparison Table:")
display(styled) if 'display' in dir() else print(comparison_df.to_string())


# In[31]:


# Visual comparison — grouped bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bar_metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
x = np.arange(len(bar_metrics))
width = 0.15
colors = ['#0984e3','#6c5ce7','#00b894','#e17055','#fdcb6e']

for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in bar_metrics]
    axes[0].bar(x + i*width, vals, width, label=name, color=colors[i], alpha=0.85)
axes[0].set_xticks(x + width*(len(results)-1)/2)
axes[0].set_xticklabels(bar_metrics, rotation=15)
axes[0].set_ylim(0.5, 1.0)
axes[0].set_title('Metric Comparison — All Models', fontweight='bold')
axes[0].legend(fontsize=7)
axes[0].set_ylabel('Score')

# CV F1 with error bars
names = list(results.keys())
cv_means = [results[n]['CV F1 Mean'] for n in names]
cv_stds  = [results[n]['CV F1 Std']  for n in names]
axes[1].barh(names, cv_means, xerr=cv_stds, color=colors[:len(names)],
             alpha=0.85, edgecolor='white', capsize=5)
axes[1].set_title('Cross-Validation F1 Score (5-Fold)', fontweight='bold')
axes[1].set_xlabel('F1 Score')
axes[1].set_xlim(0.5, 1.0)
for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
    axes[1].text(m + s + 0.005, i, f'{m:.3f}', va='center', fontsize=9)

plt.tight_layout(); plt.show()


# ## Hyperparameter Tuning
# > Fine-tune the best performing model using GridSearchCV and RandomizedSearchCV.

# In[32]:


# Identify best model by F1 Score
best_model_name = max(results, key=lambda k: results[k]['F1 Score'])
print(f"Best model selected for tuning: {best_model_name}")
print(f"Baseline F1: {results[best_model_name]['F1 Score']:.4f}")


# In[33]:


# --- RandomizedSearchCV (fast broad search) ---
print("Running RandomizedSearchCV...")

if 'Random Forest' in best_model_name:
    param_dist = {
        'n_estimators':     [100, 200, 300, 500],
        'max_depth':        [8, 10, 12, 15, None],
        'min_samples_split':[2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features':     ['sqrt', 'log2'],
    }
    base_estimator = RandomForestClassifier(random_state=42, class_weight='balanced')
elif 'Gradient' in best_model_name:
    param_dist = {
        'n_estimators':  [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth':     [3, 5, 7],
        'subsample':     [0.7, 0.8, 1.0],
    }
    base_estimator = GradientBoostingClassifier(random_state=42)
else:
    param_dist = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1','l2'], 'solver': ['liblinear']}
    base_estimator = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

random_search = RandomizedSearchCV(
    estimator=base_estimator, param_distributions=param_dist,
    n_iter=30, cv=5, scoring='f1', n_jobs=-1, random_state=42, verbose=0
)
random_search.fit(X_train_scaled, y_train)
print(f"RandomizedSearchCV best params: {random_search.best_params_}")
print(f"RandomizedSearchCV best CV F1:  {random_search.best_score_:.4f}")


# In[34]:


# --- GridSearchCV (fine-grained search around best params) ---
print("Running GridSearchCV (refined)...")

best_p = random_search.best_params_

if 'Random Forest' in best_model_name:
    grid = {
        'n_estimators': [max(50, best_p.get('n_estimators',200)-50),
                         best_p.get('n_estimators',200),
                         best_p.get('n_estimators',200)+50],
        'max_depth':    [best_p.get('max_depth',10)],
        'max_features': [best_p.get('max_features','sqrt')],
    }
    grid_estimator = RandomForestClassifier(random_state=42, class_weight='balanced',
                                             min_samples_split=best_p.get('min_samples_split',2),
                                             min_samples_leaf=best_p.get('min_samples_leaf',1))
elif 'Gradient' in best_model_name:
    grid = {
        'n_estimators':  [best_p.get('n_estimators',200)],
        'learning_rate': [best_p.get('learning_rate',0.1)],
        'max_depth':     [best_p.get('max_depth',5)],
    }
    grid_estimator = GradientBoostingClassifier(random_state=42)
else:
    grid = {'C': [best_p.get('C',1)]}
    grid_estimator = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

grid_search = GridSearchCV(grid_estimator, grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
grid_search.fit(X_train_scaled, y_train)
print(f"GridSearchCV best params: {grid_search.best_params_}")
print(f"GridSearchCV best CV F1:  {grid_search.best_score_:.4f}")


# In[35]:


# Evaluate tuned model
best_tuned_model = grid_search.best_estimator_
y_pred_tuned  = best_tuned_model.predict(X_test_scaled)
y_proba_tuned = best_tuned_model.predict_proba(X_test_scaled)[:,1]

print("\nTuned Model Performance:")
print(f"  Accuracy : {accuracy_score(y_test, y_pred_tuned):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_tuned):.4f}")
print(f"  Recall   : {recall_score(y_test, y_pred_tuned):.4f}")
print(f"  F1 Score : {f1_score(y_test, y_pred_tuned):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_test, y_proba_tuned):.4f}")
print(f"\nBaseline F1 : {results[best_model_name]['F1 Score']:.4f}")
print(f"Tuned F1    : {f1_score(y_test, y_pred_tuned):.4f}")
print(f"Improvement : +{(f1_score(y_test, y_pred_tuned) - results[best_model_name]['F1 Score']):.4f}")


# ## Feature Importance
# > Understand which features drive loan approval decisions the most.

# In[36]:


# Feature importance from the best tree-based model
if hasattr(best_tuned_model, 'feature_importances_'):
    importances = best_tuned_model.feature_importances_
elif hasattr(results[best_model_name]['model'], 'feature_importances_'):
    importances = results[best_model_name]['model'].feature_importances_
else:
    # Use logistic regression coefficients
    importances = np.abs(results['Logistic Regression']['model'].coef_[0])

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart — top 15
top_n = min(15, len(feat_imp))
feat_imp.head(top_n).plot(kind='barh', ax=axes[0], color='#0984e3', edgecolor='white')
axes[0].invert_yaxis()
axes[0].set_title(f'Top {top_n} Feature Importances', fontweight='bold')
axes[0].set_xlabel('Importance Score')

# Cumulative importance
cumulative = feat_imp.cumsum() / feat_imp.sum() * 100
axes[1].plot(range(1, len(cumulative)+1), cumulative.values, marker='o',
             color='#6c5ce7', linewidth=2, markersize=4)
axes[1].axhline(80, color='red', linestyle='--', linewidth=1, label='80% threshold')
axes[1].axhline(95, color='orange', linestyle='--', linewidth=1, label='95% threshold')
axes[1].set_xlabel('Number of Features')
axes[1].set_ylabel('Cumulative Importance (%)')
axes[1].set_title('Cumulative Feature Importance', fontweight='bold')
axes[1].legend()

plt.tight_layout(); plt.show()

print("\nTop 10 Most Important Features:")
print(feat_imp.head(10).round(4).to_string())


# ## Business Insights
# > Translate data findings into actionable recommendations for financial institutions.
# 
# ---
# 
# ### Key Findings
# 
# | # | Insight | Business Impact |
# |---|---------|----------------|
# | 1 | **Loan-to-income ratio** is the strongest predictor | Applicants borrowing >50% of annual income are significantly higher risk |
# | 2 | **Credit history length** correlates strongly with approval | Longer credit history = more reliable repayment patterns |
# | 3 | **Loan intent** affects approval rates | Education/medical loans have different risk profiles than personal loans |
# | 4 | **Previous defaults** is a near-perfect disqualifier | Applicants with prior defaults are rejected at a much higher rate |
# | 5 | **Person income** shows clear separation between classes | Higher income applicants have substantially higher approval rates |
# 
# ### Recommendations
# - **Risk Scoring Model**: Deploy this model as an automated pre-screening tool to flag high-risk applications before human review.
# - **Threshold Adjustment**: Lower the classification threshold (e.g. 0.4) to reduce false negatives (missed bad loans) at the cost of slightly more rejections.
# - **Feature Monitoring**: Monitor `debt_to_income_ratio` drift in production as economic conditions change.
# - **Fairness Audit**: Regularly audit model outputs for gender/demographic bias using the `person_gender` feature.
# - **Re-training Cadence**: Retrain quarterly using rolling data windows to capture credit market shifts.

# In[37]:


# Visual business insight: approval rate by loan intent
if 'loan_intent' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    intent_approval = df.groupby('loan_intent').agg(
        approval_rate=('loan_status','mean'),
        count=('loan_status','count')
    ).sort_values('approval_rate', ascending=False)
    intent_approval['approval_rate_pct'] = intent_approval['approval_rate'] * 100

    colors = sns.color_palette('Set2', len(intent_approval))
    axes[0].bar(range(len(intent_approval)), intent_approval['approval_rate_pct'],
                color=colors, edgecolor='white')
    axes[0].set_xticks(range(len(intent_approval)))
    axes[0].set_xticklabels(intent_approval.index, rotation=30)
    axes[0].set_title('Approval Rate by Loan Intent', fontweight='bold')
    axes[0].set_ylabel('Approval Rate (%)')
    for i, v in enumerate(intent_approval['approval_rate_pct']):
        axes[0].text(i, v+0.5, f'{v:.1f}%', ha='center', fontsize=9)

    axes[1].bar(range(len(intent_approval)), intent_approval['count'],
                color=colors, edgecolor='white')
    axes[1].set_xticks(range(len(intent_approval)))
    axes[1].set_xticklabels(intent_approval.index, rotation=30)
    axes[1].set_title('Application Volume by Loan Intent', fontweight='bold')
    axes[1].set_ylabel('Number of Applications')
    plt.suptitle('Business Insight: Loan Intent Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()


# ##  Model Saving
# > Save the best model and preprocessing artifacts for production deployment.

# In[38]:


os.makedirs('saved_model', exist_ok=True)

# Save best tuned model
joblib.dump(best_tuned_model, 'saved_model/best_model.pkl')
print("Model saved: saved_model/best_model.pkl")

# Save scaler
joblib.dump(scaler, 'saved_model/scaler.pkl')
print("Scaler saved: saved_model/scaler.pkl")

# Save feature names (needed for inference)
joblib.dump(feature_names, 'saved_model/feature_names.pkl')
print("Feature names saved: saved_model/feature_names.pkl")

# Save label encodings
joblib.dump(label_mappings, 'saved_model/label_mappings.pkl')
print("Label mappings saved: saved_model/label_mappings.pkl")

# Verify
loaded_model = joblib.load('saved_model/best_model.pkl')
test_pred = loaded_model.predict(X_test_scaled[:5])
print(f"\nVerification — predictions on 5 test samples: {test_pred}")
print("Model loading verified successfully!")


# ## Deployment Preparation (Streamlit App)
# > The cell below generates a ready-to-run `app.py` file for Streamlit deployment. Run it with: `streamlit run app.py`

# In[41]:


streamlit_app = '''
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(page_title="Loan Approval Predictor",
                   page_icon="🏦", layout="centered")

st.title("🏦 Loan Approval Prediction")
st.markdown("Enter applicant details below to predict loan approval likelihood.")

# ── Load Artifacts ───────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("saved_model/best_model.pkl")
    scaler   = joblib.load("saved_model/scaler.pkl")
    features = joblib.load("saved_model/feature_names.pkl")
    mappings = joblib.load("saved_model/label_mappings.pkl")
    return model, scaler, features, mappings

model, scaler, feature_names, label_mappings = load_artifacts()

# ── Input Form ───────────────────────────────────────────────
st.subheader("Applicant Information")
col1, col2 = st.columns(2)

with col1:
    age            = st.number_input("Age", 18, 80, 30)
    income         = st.number_input("Annual Income ($)", 5000, 500000, 50000, step=1000)
    loan_amnt      = st.number_input("Loan Amount ($)", 500, 100000, 10000, step=500)
    loan_int_rate  = st.number_input("Interest Rate (%)", 1.0, 30.0, 12.0, step=0.1)

with col2:
    loan_pct_income= st.slider("Loan % of Income", 0.0, 1.0, 0.2, step=0.01)
    cred_hist      = st.number_input("Credit History Length (years)", 0, 30, 5)
    prev_default   = st.selectbox("Previous Loan Default?", ["No", "Yes"])
    loan_intent    = st.selectbox("Loan Intent", list(label_mappings.get("loan_intent", {"PERSONAL":0}).keys()))
    person_gender  = st.selectbox("Gender", list(label_mappings.get("person_gender", {"Male":0}).keys()))
    education      = st.selectbox("Education", list(label_mappings.get("person_education", {"Bachelor":0}).keys()))

# ── Predict ──────────────────────────────────────────────────
if st.button("🔍 Predict Loan Approval", type="primary"):
    # Build input dict
    input_data = {
        "person_age": age,
        "person_income": np.log1p(income),
        "person_gender": label_mappings.get("person_gender", {}).get(person_gender, 0),
        "person_education": label_mappings.get("person_education", {}).get(education, 0),
        "loan_amnt": np.log1p(loan_amnt),
        "loan_intent": label_mappings.get("loan_intent", {}).get(loan_intent, 0),
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": np.log1p(loan_pct_income),
        "cb_person_cred_hist_length": cred_hist,
        "previous_loan_defaults_on_file": 1 if prev_default=="Yes" else 0,
        "debt_to_income_ratio": loan_amnt / (income + 1),
        "income_per_age": income / (age + 1),
        "loan_to_credit_hist": loan_amnt / (cred_hist + 1),
        "is_high_loan": 1 if loan_amnt > 25000 else 0,
        "is_high_income": 1 if income > 60000 else 0,
    }

    # Align features
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Scale & predict
    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    # Show result
    st.divider()
    if prediction == 1:
        st.success(f"✅ **APPROVED** — Approval Probability: {probability*100:.1f}%")
        st.progress(probability)
    else:
        st.error(f"❌ **REJECTED** — Approval Probability: {probability*100:.1f}%")
        st.progress(probability)

    st.caption("This prediction is generated by a machine learning model and should be used as a supporting tool only.")
'''
with open("app.py", "w", encoding="utf-8") as f:
    f.write(streamlit_app)

print("Streamlit app written to: app.py")
print("To run: streamlit run app.py")


# In[ ]:


get_ipython().system('{sys.executable} -m streamlit run app.py')


# ## ✅ Section 14 — Conclusion & Next Steps
# 
# ---
# 
# ### Project Summary
# 
# This end-to-end ML project successfully built a **loan approval prediction pipeline** covering:
# 
# - **Data Cleaning**: Missing values, duplicates, outliers treated systematically
# - **Advanced EDA**: Univariate, bivariate, multivariate analysis with actionable insights
# - **Feature Engineering**: 5 domain-driven features created (debt-to-income ratio, income-per-age, etc.)
# - **5 Models Trained**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost
# - **Rigorous Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC, 5-Fold Cross Validation
# - **Hyperparameter Tuning**: RandomizedSearchCV + GridSearchCV for optimal performance
# - **Feature Importance**: Identified top predictors with business explanation
# - **Model Persistence**: Saved model, scaler, encodings for production use
# - **Streamlit App**: Deployment-ready interactive web application
# 
# ---
# 
# ### Next Steps
# | Enhancement | Description |
# |-------------|-------------|
# | SMOTE | Apply Synthetic Minority Oversampling for better handling of class imbalance |
# | SHAP Values | Use SHAP for local + global model explainability |
# | MLflow | Track experiments with MLflow for model versioning |
# | FastAPI | Build REST API endpoint for integration with banking systems |
# | Docker | Containerize the Streamlit app for cloud deployment |
# | Monitoring | Set up data drift detection with Evidently AI |
# 
