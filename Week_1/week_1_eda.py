import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# 1. DATA ACQUISITION & DATASET GENERATION
print("[1/4] Acquiring and creating dataset...")
np.random.seed(42)
n_samples = 1200

ages = np.random.randint(18, 70, size=n_samples).astype(float)
ages[np.random.choice(n_samples, 45, replace=False)] = np.nan

genders = np.random.choice(
    ['Male', 'Female', 'Other', None],
    size=n_samples,
    p=[0.48, 0.47, 0.03, 0.02]
)
annual_income = np.random.normal(55000, 18000, size=n_samples).round(2)
annual_income[annual_income < 15000] = 15000
annual_income[np.random.choice(n_samples, 60, replace=False)] = np.nan

tenure_months = np.random.randint(1, 72, size=n_samples)
purchase_frequency = np.random.poisson(lam=6, size=n_samples)
avg_order_value = np.random.gamma(shape=5, scale=20, size=n_samples).round(2)
avg_order_value[np.random.choice(n_samples, 10, replace=False)] = [
    850, 920, 1100, 1250, 980, 1400, 1050, 890, 950, 1300
]

churn_prob = 1 / (
    1 + np.exp(-(
        0.02 * (60 - tenure_months)
        - 0.00002 * np.nan_to_num(annual_income, nan=55000)
        - 0.15 * purchase_frequency
        + 0.5
    ))
)
churn = (np.random.rand(n_samples) < churn_prob).astype(int)

df_raw = pd.DataFrame({
    'CustomerID': [f'CUST-{10000+i}' for i in range(n_samples)],
    'Age': ages,
    'Gender': genders,
    'AnnualIncome': annual_income,
    'TenureMonths': tenure_months,
    'PurchaseFrequency': purchase_frequency,
    'AvgOrderValue': avg_order_value,
    'Churn': churn
})

# Add duplicates
duplicates = df_raw.sample(25, random_state=42)
df_messy = pd.concat([df_raw, duplicates], ignore_index=True)
df_messy.to_csv('data/raw_dataset.csv', index=False)
print(f"-> Raw data saved in 'data/raw_dataset.csv' (Rows: {len(df_messy)})")

# 2. DATA CLEANING & PREPROCESSING
print("\n[2/4] Cleaning data...")
initial_count = len(df_messy)
df_clean = df_messy.drop_duplicates(subset=['CustomerID']).copy()
print(f"-> Removed {initial_count - len(df_clean)} duplicates.")

# Missing value handling
df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median()).astype(int)
df_clean['AnnualIncome'] = df_clean['AnnualIncome'].fillna(df_clean['AnnualIncome'].median())
df_clean['Gender'] = df_clean['Gender'].fillna('Unknown').astype('category')
df_clean['Churn'] = df_clean['Churn'].astype(int)

# Outlier Detection via IQR
Q1 = df_clean['AvgOrderValue'].quantile(0.25)
Q3 = df_clean['AvgOrderValue'].quantile(0.75)
IQR = Q3 - Q1
upper_limit = Q3 + 3.0 * IQR
df_clean['Is_AOV_Outlier'] = (df_clean['AvgOrderValue'] > upper_limit).astype(int)

df_clean.to_csv('data/cleaned_dataset.csv', index=False)
print("-> Cleaned data saved in 'data/cleaned_dataset.csv'")

# 3. GENERATING VISUALIZATIONS (EDA)
print("\n[3/4] Generating and saving charts...")

# Figure 1: Missing Values
fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
missing_counts = df_messy.isnull().sum()
missing_counts = missing_counts[missing_counts > 0]
ax.bar(missing_counts.index, missing_counts.values, color=['#e74c3c', '#e67e22', '#f39c12'], edgecolor='#222')
ax.set_title('Figure 1: Pre-Cleaning Missing Values Count', fontsize=12, fontweight='bold')
ax.set_ylabel('Missing Records')
plt.savefig('plots/fig1_missing_values.png', bbox_inches='tight')
plt.close()

# Figure 2: Correlation Matrix
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
num_cols = ['Age', 'AnnualIncome', 'TenureMonths', 'PurchaseFrequency', 'AvgOrderValue', 'Churn']
sns.heatmap(df_clean[num_cols].corr(), annot=True, cmap='Blues', fmt='.2f', linewidths=1, ax=ax)
ax.set_title('Figure 2: Feature Correlation Heatmap', fontsize=12, fontweight='bold')
plt.savefig('plots/fig2_correlation_matrix.png', bbox_inches='tight')
plt.close()

# Figure 3: Churn Distributions
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
sns.boxplot(x='Churn', y='TenureMonths', data=df_clean, palette=['#2ecc71', '#e74c3c'], ax=axes[0])
axes[0].set_xticklabels(['Retained (0)', 'Churned (1)'])
axes[0].set_title('Tenure vs Churn', fontweight='bold')

sns.kdeplot(data=df_clean, x='PurchaseFrequency', hue='Churn', fill=True, palette=['#2ecc71', '#e74c3c'], ax=axes[1])
axes[1].set_title('Purchase Frequency by Churn', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/fig3_churn_analysis.png', bbox_inches='tight')
plt.close()

# Figure 4: Outlier Boxplot
fig, ax = plt.subplots(figsize=(8, 3.5), dpi=300)
sns.boxplot(x=df_clean['AvgOrderValue'], color='#3498db', ax=ax, flierprops={'markerfacecolor': '#e74c3c'})
ax.set_title('Figure 4: Average Order Value Outliers', fontsize=12, fontweight='bold')
plt.savefig('plots/fig4_outliers.png', bbox_inches='tight')
plt.close()

print("\n[4/4] Done! All CSVs and Charts generated successfully.")