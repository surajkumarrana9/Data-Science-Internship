import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. Load / Generate Business Dataset (Customer Spending & Campaign Analysis)
np.random.seed(42)
n = 200
data = pd.DataFrame({
    'Customer_ID': range(1, n + 1),
    'Group': np.random.choice(['Control', 'Treatment'], size=n),
    'Region': np.random.choice(['North', 'South', 'East'], size=n),
    'Purchased': np.random.choice(['Yes', 'No'], size=n, p=[0.45, 0.55]),
    'Spend_Amount': np.concatenate([
        np.random.normal(loc=1200, scale=250, size=100),
        np.random.normal(loc=1380, scale=270, size=100)
    ])
})

# 2. Independent Two-Sample T-Test (Control vs Treatment Spend)
control_spend = data[data['Group'] == 'Control']['Spend_Amount']
treatment_spend = data[data['Group'] == 'Treatment']['Spend_Amount']
t_stat, p_val_t = stats.ttest_ind(treatment_spend, control_spend)

# 95% Confidence Interval for Mean Difference
diff_mean = treatment_spend.mean() - control_spend.mean()
se_diff = np.sqrt(treatment_spend.var()/len(treatment_spend) + control_spend.var()/len(control_spend))
ci_low, ci_high = stats.t.interval(0.95, df=len(data)-2, loc=diff_mean, scale=se_diff)

print(f"T-Test: t-statistic = {t_stat:.4f}, p-value = {p_val_t:.4f}")
print(f"95% Confidence Interval for Mean Difference: ({ci_low:.2f}, {ci_high:.2f})")

# 3. One-Way ANOVA (Spend across Regions)
north = data[data['Region'] == 'North']['Spend_Amount']
south = data[data['Region'] == 'South']['Spend_Amount']
east = data[data['Region'] == 'East']['Spend_Amount']
f_stat, p_val_anova = stats.f_oneway(north, south, east)
print(f"ANOVA: F-statistic = {f_stat:.4f}, p-value = {p_val_anova:.4f}")

# 4. Chi-Square Test of Independence (Group vs Purchase Conversion)
contingency_table = pd.crosstab(data['Group'], data['Purchased'])
chi2_stat, p_val_chi2, dof, ex = stats.chi2_contingency(contingency_table)
print(f"Chi-Square Test: chi2 = {chi2_stat:.4f}, p-value = {p_val_chi2:.4f}")

# 5. Visualization
plt.figure(figsize=(8, 5))
sns.boxplot(x='Group', y='Spend_Amount', data=data, palette='Set2')
plt.title('Customer Spend Distribution: Control vs Treatment Group')
plt.xlabel('Campaign Group')
plt.ylabel('Spend Amount (INR)')
plt.savefig('hypothesis_test_boxplot.png', dpi=300)
plt.show()
