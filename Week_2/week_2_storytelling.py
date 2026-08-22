import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Styling for storytelling charts
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs('data', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# -------------------------------------------------------------
# 1. DATA GENERATION & CONTEXT
# -------------------------------------------------------------
print("[1/3] Generating executive customer journey dataset...")
np.random.seed(42)
n_records = 2500

segments = np.random.choice(['Budget Hunter', 'Regular Consumer', 'High-Tier VIP', 'Corporate/Bulk'], size=n_records, p=[0.38, 0.42, 0.15, 0.05])
channels = np.random.choice(['Paid Search', 'Social Media', 'Organic/Direct', 'Email Referral'], size=n_records, p=[0.35, 0.30, 0.20, 0.15])
regions = np.random.choice(['North America', 'Europe', 'Asia-Pacific', 'Latin America'], size=n_records, p=[0.40, 0.30, 0.20, 0.10])

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_revenue_k = [142, 138, 155, 168, 185, 210, 205, 224, 248, 290, 385, 460]
monthly_churn_pct = [4.8, 4.6, 4.2, 4.0, 3.8, 3.5, 3.6, 3.2, 3.0, 2.7, 2.4, 2.1]

tenure = np.random.randint(1, 48, size=n_records)
clv_base = {'Budget Hunter': 350, 'Regular Consumer': 1200, 'High-Tier VIP': 4500, 'Corporate/Bulk': 11000}
clv = np.array([np.random.normal(clv_base[s], clv_base[s]*0.25) for s in segments])
clv = np.clip(clv, 100, None)

csat_score = np.random.choice([1, 2, 3, 4, 5], size=n_records, p=[0.06, 0.12, 0.22, 0.40, 0.20])
churn_prob = np.clip(0.65 - (csat_score * 0.11) - (tenure * 0.005) + (clv < 500)*0.15, 0.05, 0.95)
churn = (np.random.rand(n_records) < churn_prob).astype(int)

df = pd.DataFrame({
    'Customer_Segment': segments,
    'Acquisition_Channel': channels,
    'Region': regions,
    'Tenure_Months': tenure,
    'Customer_Lifetime_Value': clv.round(2),
    'CSAT_Rating': csat_score,
    'Churn': churn
})
df.to_csv('data/week2_customer_analytics.csv', index=False)
print(f"-> Saved data to 'data/week2_customer_analytics.csv' ({len(df)} records)")

# -------------------------------------------------------------
# 2. GENERATING 5 ADVANCED STORYTELLING VISUALS
# -------------------------------------------------------------
print("\n[2/3] Rendering 5 narrative visualization figures...")

# Visual 1: Dual-Axis Macro Trajectory
fig, ax1 = plt.subplots(figsize=(8.5, 4.5), dpi=300)
ax1.set_xlabel('Annual Progression (Months)', fontsize=11, fontweight='bold', labelpad=8)
ax1.set_ylabel('Total Revenue ($ in Thousands)', color='#1A365D', fontsize=11, fontweight='bold')
ax1.bar(months, monthly_revenue_k, color='#2B6CB0', alpha=0.85, width=0.55)
ax1.set_ylim(0, 520)

ax1.annotate('Q4 Holiday Surge\n(+58% MoM)', xy=(10.5, 420), xytext=(8.2, 460),
             arrowprops=dict(facecolor='#E53E3E', shrink=0.08, width=1.5, headwidth=6),
             fontweight='bold', color='#C53030', fontsize=9.5)

ax2 = ax1.twinx()
ax2.set_ylabel('Monthly Churn Rate (%)', color='#E53E3E', fontsize=11, fontweight='bold')
ax2.plot(months, monthly_churn_pct, color='#E53E3E', marker='o', linewidth=2.5, markersize=7)
ax2.set_ylim(0, 6.5)
ax2.grid(False)
plt.title('Story 1: Macro Trajectory — Revenue Scaling Inversely with Customer Churn', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('plots/fig1_macro_trajectory.png')
plt.close()

# Visual 2: Value Hierarchy Boxplot
fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
sns.boxplot(x='Customer_Segment', y='Customer_Lifetime_Value', data=df, palette='Blues_r', ax=ax, width=0.5, fliersize=3)
ax.set_yscale('log')
ax.set_title('Story 2: Value Hierarchy — Disproportionate CLV Across Customer Tiers', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Customer Segment', fontsize=11, fontweight='bold')
ax.set_ylabel('Customer Lifetime Value ($ Log Scale)', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/fig2_clv_hierarchy.png')
plt.close()

# Visual 3: Attribution Bubble Chart
fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
channel_summary = df.groupby('Acquisition_Channel').agg(
    Avg_CLV=('Customer_Lifetime_Value', 'mean'),
    Retention_Rate=('Churn', lambda x: (1 - x.mean()) * 100)
).reset_index()
ax.scatter(channel_summary['Avg_CLV'], channel_summary['Retention_Rate'], 
           s=[1200, 1500, 1800, 1100], c=['#3182CE', '#38A169', '#DD6B20', '#805AD5'], alpha=0.85, edgecolors='black', linewidth=1.5)
for i, txt in enumerate(channel_summary['Acquisition_Channel']):
    ax.annotate(f"{txt}\n({channel_summary['Retention_Rate'][i]:.1f}% Retained)", 
                (channel_summary['Avg_CLV'][i], channel_summary['Retention_Rate'][i]),
                textcoords="offset points", xytext=(0, 12), ha='center', fontweight='bold', fontsize=9.5)
ax.set_title('Story 3: Channel Efficacy — Acquisition Channel vs Retention & CLV', fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('Average Customer Lifetime Value ($ USD)', fontsize=11, fontweight='bold')
ax.set_ylabel('Customer Retention Rate (%)', fontsize=11, fontweight='bold')
ax.set_ylim(40, 85)
ax.set_xlim(1200, 2400)
plt.tight_layout()
plt.savefig('plots/fig3_channel_attribution.png')
plt.close()

# Visual 4: CSAT vs Churn Heatmap
fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
cross_tab = pd.crosstab(df['CSAT_Rating'], df['Customer_Segment'], values=df['Churn'], aggfunc='mean') * 100
sns.heatmap(cross_tab, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=1.2, linecolor='white', cbar_kws={'label': 'Churn Rate (%)'}, ax=ax)
ax.set_title('Story 4: Risk Matrix — Churn Propensity by CSAT Rating & Segment', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Customer Segment', fontsize=11, fontweight='bold')
ax.set_ylabel('CSAT Score (1=Poor, 5=Excellent)', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/fig4_csat_risk_matrix.png')
plt.close()

# Visual 5: Donut Regional Breakdown
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
region_perf = df.groupby('Region')['Customer_Lifetime_Value'].sum()
wedges, texts, autotexts = ax.pie(region_perf, labels=region_perf.index, autopct='%1.1f%%', 
                                  startangle=140, colors=['#2B6CB0', '#4299E1', '#63B3ED', '#90CDF4'],
                                  wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2), pctdistance=0.75)
for at in autotexts:
    at.set_color('white')
    at.set_fontweight('bold')
for t in texts:
    t.set_fontsize(10.5)
    t.set_fontweight('bold')
ax.set_title('Story 5: Global Strategic Footprint — Regional Revenue Share', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('plots/fig5_regional_footprint.png')
plt.close()

print("[3/3] Done! All 5 figures generated in 'plots/'.")