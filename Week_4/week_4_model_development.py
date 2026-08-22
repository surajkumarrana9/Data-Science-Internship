import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs('data', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# 1. Dataset Generation
print("[1/4] Preparing dataset for Machine Learning...")
np.random.seed(42)
n_samples = 2000

age = np.random.randint(18, 70, size=n_samples)
gender = np.random.choice(['Male', 'Female', 'Unknown'], size=n_samples, p=[0.48, 0.48, 0.04])
annual_income = np.random.normal(55000, 18000, size=n_samples).round(2)
tenure_months = np.random.randint(1, 72, size=n_samples)
purchase_freq = np.random.poisson(lam=6, size=n_samples)
avg_order_val = np.random.gamma(shape=5, scale=20, size=n_samples).round(2)
csat_rating = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.08, 0.15, 0.25, 0.35, 0.17])

logit = -0.04*tenure_months - 0.22*purchase_freq - 0.35*csat_rating + (annual_income < 35000)*0.6 + 1.8
prob = 1 / (1 + np.exp(-logit))
churn = (np.random.rand(n_samples) < prob).astype(int)

df = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'AnnualIncome': annual_income,
    'TenureMonths': tenure_months,
    'PurchaseFrequency': purchase_freq,
    'AvgOrderValue': avg_order_val,
    'CSAT_Rating': csat_rating,
    'Churn': churn
})
df.to_csv('data/ml_customer_churn.csv', index=False)

# 2. Pipeline & Model Training
print("[2/4] Training Logistic Regression and Random Forest models...")
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

num_cols = ['Age', 'AnnualIncome', 'TenureMonths', 'PurchaseFrequency', 'AvgOrderValue', 'CSAT_Rating']
cat_cols = ['Gender']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first'), cat_cols)
    ])

# Model 1: Baseline Logistic Regression
lr_model = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(random_state=42))
])
lr_model.fit(X_train, y_train)

# Model 2: Champion Random Forest Classifier
rf_model = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
])
rf_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# 3. Visualizations
print("[3/4] Generating evaluation charts...")

# Visual 1: Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_rf = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
            xticklabels=['Retained (0)', 'Churned (1)'], yticklabels=['Retained (0)', 'Churned (1)'])
axes[0].set_title('Baseline: Logistic Regression Confusion Matrix', fontweight='bold')
axes[0].set_ylabel('True Label', fontweight='bold')
axes[0].set_xlabel('Predicted Label', fontweight='bold')

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[1],
            xticklabels=['Retained (0)', 'Churned (1)'], yticklabels=['Retained (0)', 'Churned (1)'])
axes[1].set_title('Champion: Random Forest Confusion Matrix', fontweight='bold')
axes[1].set_ylabel('True Label', fontweight='bold')
axes[1].set_xlabel('Predicted Label', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/fig1_confusion_matrices.png')
plt.close()

# Visual 2: ROC and PR Curves
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), dpi=300)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

axes[0].plot(fpr_lr, tpr_lr, color='#3182CE', lw=2.2, label=f'Logistic Regression (AUC = {roc_auc_lr:.3f})')
axes[0].plot(fpr_rf, tpr_rf, color='#38A169', lw=2.5, label=f'Random Forest (AUC = {roc_auc_rf:.3f})')
axes[0].plot([0, 1], [0, 1], color='grey', lw=1.2, linestyle='--')
axes[0].set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
axes[0].set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
axes[0].set_title('Receiver Operating Characteristic (ROC) Comparison', fontweight='bold')
axes[0].legend(loc="lower right")

prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_prob_lr)
prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_prob_rf)
axes[1].plot(rec_lr, prec_lr, color='#3182CE', lw=2.2, label='Logistic Regression')
axes[1].plot(rec_rf, prec_rf, color='#38A169', lw=2.5, label='Random Forest')
axes[1].set_xlabel('Recall', fontweight='bold')
axes[1].set_ylabel('Precision', fontweight='bold')
axes[1].set_title('Precision-Recall Curve Comparison', fontweight='bold')
axes[1].legend(loc="lower left")
plt.tight_layout()
plt.savefig('plots/fig2_roc_pr_curves.png')
plt.close()

# Visual 3: Feature Importances
fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=300)
rf_estimator = rf_model.named_steps['clf']
feature_names = num_cols + ['Gender_Male', 'Gender_Unknown']
importances = pd.Series(rf_estimator.feature_importances_, index=feature_names).sort_values(ascending=True)
importances.plot(kind='barh', color='#2B6CB0', edgecolor='black', ax=ax, width=0.6)
ax.set_title('Figure 3: Random Forest Feature Importance Rankings', fontweight='bold', fontsize=12)
ax.set_xlabel('Gini Importance', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/fig3_feature_importance.png')
plt.close()

print("[4/4] Completed! All data and charts created in 'data/' and 'plots/'.")