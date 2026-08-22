import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs('data', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# -------------------------------------------------------------
# 1. SYNTHESIS DATASETS
# -------------------------------------------------------------
print("[1/3] Generating executive summary KPI records...")

cohorts = ['0-12m', '13-24m', '25-36m', '37-48m', '49-60m', '60m+']
retention_rate = [58.4, 72.1, 84.6, 91.2, 94.8, 97.5]
clv_contribution = [1200, 2450, 4800, 7100, 9600, 13400]

df_cohort = pd.DataFrame({
    'TenureBracket': cohorts,
    'RetentionRate_Pct': retention_rate,
    'AvgCLV_USD': clv_contribution
})
df_cohort.to_csv('data/executive_cohort_summary.csv', index=False)

# -------------------------------------------------------------
# 2. GENERATE STRATEGIC VISUALIZATIONS
# -------------------------------------------------------------
print("[2/3] Generating final capstone visuals...")

# Visual 1: Retention Rate & CLV Expansion
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

axes[0].bar(cohorts, retention_rate, color='#2B6CB0', edgecolor='black', width=0.6)
axes[0].set_title('Figure 1: Retention Rate by Customer Tenure Cohort', fontweight='bold', fontsize=11)
axes[0].set_ylabel('Retention Rate (%)', fontweight='bold')
axes[0].set_xlabel('Tenure Lifespan Bracket', fontweight='bold')
for i, v in enumerate(retention_rate):
    axes[0].text(i, v + 1.2, f"{v}%", ha='center', fontweight='bold', fontsize=9)

axes[1].plot(cohorts, clv_contribution, marker='o', color='#38A169', lw=2.5, markersize=7)
axes[1].set_title('Figure 2: Cumulative Customer Lifetime Value ($ USD)', fontweight='bold', fontsize=11)
axes[1].set_ylabel('Average CLV ($)', fontweight='bold')
axes[1].set_xlabel('Tenure Lifespan Bracket', fontweight='bold')
for i, v in enumerate(clv_contribution):
    axes[1].text(i, v + 350, f"${v:,}", ha='center', fontweight='bold', fontsize=8.5)

plt.tight_layout()
plt.savefig('plots/fig1_executive_kpi.png')
plt.close()

# Visual 2: Strategic Intervention ROI Matrix
fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
initiatives = [
    'Proactive Onboarding (0-6m)',
    'Automated Retention Trigger (CSAT < 3)',
    'High-Spend Loyalty Program',
    'Self-Serve Support Ticketing AI',
    'Discounted Long-Term Lock-in'
]
roi = [340, 280, 210, 195, 140]
colors = ['#2B6CB0', '#3182CE', '#38A169', '#48BB78', '#A0AEC0']

y_pos = np.arange(len(initiatives))
ax.barh(y_pos, roi, color=colors, edgecolor='black', height=0.55)
ax.set_yticks(y_pos)
ax.set_yticklabels(initiatives, fontweight='bold', fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Projected 12-Month ROI (%) on Churn Mitigation', fontweight='bold')
ax.set_title('Figure 3: Strategic Roadmap Expected ROI & Impact Prioritization', fontweight='bold', fontsize=12)

for i, v in enumerate(roi):
    ax.text(v + 5, i, f"{v}% ROI", va='center', fontweight='bold', fontsize=9.5)

plt.tight_layout()
plt.savefig('plots/fig2_strategic_roi.png')
plt.close()

print("[3/3] Completed! Visuals and datasets saved in 'Week_5/plots/' and 'Week_5/data/'.")