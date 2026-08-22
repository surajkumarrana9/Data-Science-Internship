import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs('data', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# -------------------------------------------------------------
# 1. DATASET CREATION & DOMAIN FEATURE ENGINEERING
# -------------------------------------------------------------
print("[1/3] Synthesizing dataset and engineering behavioral signals...")
np.random.seed(42)
n_samples = 2200

tenure_months = np.random.randint(1, 72, size=n_samples)
purchase_freq = np.random.poisson(lam=6, size=n_samples)
avg_order_val = np.random.gamma(shape=5, scale=20, size=n_samples).round(2)
annual_income = np.random.normal(55000, 18000, size=n_samples).round(2)
csat_rating = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.08, 0.15, 0.25, 0.35, 0.17])
support_tickets = np.random.poisson(lam=2, size=n_samples)

# Feature synthesis
annual_spend = (purchase_freq * 12 * avg_order_val).round(2)
income_spend_ratio = (annual_spend / np.clip(annual_income, 10000, None)).round(4)
engagement_score = (purchase_freq / (support_tickets + 1)).round(3)
tenure_spend_velocity = (annual_spend / np.clip(tenure_months, 1, None)).round(2)

churn_logit = (-0.03*tenure_months - 0.4*engagement_score - 0.3*csat_rating 
               + 0.5*support_tickets + (income_spend_ratio > 0.15)*0.8 + 0.5)
churn_prob = 1 / (1 + np.exp(-churn_logit))
churn = (np.random.rand(n_samples) < churn_prob).astype(int)

df_feat = pd.DataFrame({
    'TenureMonths': tenure_months,
    'PurchaseFrequency': purchase_freq,
    'AvgOrderValue': avg_order_val,
    'AnnualIncome': annual_income,
    'CSAT_Rating': csat_rating,
    'SupportTickets': support_tickets,
    'AnnualSpend': annual_spend,
    'IncomeSpendRatio': income_spend_ratio,
    'EngagementScore': engagement_score,
    'TenureSpendVelocity': tenure_spend_velocity,
    'Churn': churn
})
df_feat.to_csv('data/engineered_customer_features.csv', index=False)

# -------------------------------------------------------------
# 2. GENERATE DIAGNOSTIC VISUALS
# -------------------------------------------------------------
print("[2/3] Generating transformation and dimensionality reduction visuals...")

# Visual 1: Distribution Transformation
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
sns.histplot(df_feat['AnnualSpend'], kde=True, color='#E53E3E', ax=axes[0], bins=30)
axes[0].set_title('Raw Skewed Distribution: Annual Spend ($)', fontweight='bold')
axes[0].set_xlabel('Annual Spend ($ USD)')

sns.histplot(np.log1p(df_feat['AnnualSpend']), kde=True, color='#2B6CB0', ax=axes[1], bins=30)
axes[1].set_title('Transformed Log(1 + Annual Spend) Normalization', fontweight='bold')
axes[1].set_xlabel('Log Transformed Value')
plt.tight_layout()
plt.savefig('plots/fig1_feature_distribution.png')
plt.close()

# Visual 2: Mutual Information Ranking
X = df_feat.drop('Churn', axis=1)
y = df_feat['Churn']
mi_scores = pd.Series(mutual_info_classif(X, y, random_state=42), index=X.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
mi_scores.plot(kind='barh', color='#3182CE', edgecolor='black', ax=ax, width=0.6)
ax.set_title('Figure 2: Mutual Information Scores of Raw vs Engineered Features', fontweight='bold', fontsize=12)
ax.set_xlabel('Information Gain / Mutual Information with Target (Churn)', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/fig2_mutual_info.png')
plt.close()

# Visual 3: PCA Variance & 2D Projection
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), dpi=300)
pca_full = PCA().fit(X_scaled)
axes[0].plot(range(1, len(pca_full.explained_variance_ratio_) + 1), np.cumsum(pca_full.explained_variance_ratio_), marker='o', color='#2B6CB0', lw=2.2)
axes[0].axhline(y=0.80, color='r', linestyle='--', label='80% Explained Variance')
axes[0].set_xlabel('Number of Principal Components', fontweight='bold')
axes[0].set_ylabel('Cumulative Explained Variance', fontweight='bold')
axes[0].set_title('PCA Cumulative Explained Variance Scree Plot', fontweight='bold')
axes[0].legend(loc='lower right')

scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.5, s=25, edgecolors='none')
axes[1].set_xlabel('Principal Component 1 (PC1)', fontweight='bold')
axes[1].set_ylabel('Principal Component 2 (PC2)', fontweight='bold')
axes[1].set_title('2D PCA Projection of Engineered Feature Space', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/fig3_pca_analysis.png')
plt.close()

print("[3/3] Feature engineering workflow complete! Visuals saved in 'plots/'.")