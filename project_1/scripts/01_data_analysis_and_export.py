import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'creditcard.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print(f"Loading dataset from: {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded successfully. Shape: {df.shape}")

# 1. Summary Statistics for Q3
print("\n--- DATASET METADATA & SUMMARY (Q3) ---")
print(f"Total Transactions: {len(df):,}")
print(f"Total Columns: {len(df.columns)}")
print(f"Missing Values: {df.isnull().sum().sum()}")

class_counts = df['Class'].value_counts()
n_legit = int(class_counts.get(0, 0))
n_fraud = int(class_counts.get(1, 0))
pct_legit = (n_legit / len(df)) * 100
pct_fraud = (n_fraud / len(df)) * 100

print(f"Class 0 (Legitimate): {n_legit:,} ({pct_legit:.4f}%)")
print(f"Class 1 (Fraudulent): {n_fraud:,} ({pct_fraud:.4f}%)")
print(f"Imbalance Ratio: {n_legit / n_fraud:.1f} : 1")

# Compute summary stats table
stats_df = df[['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V10', 'V12', 'V14', 'V17']].describe().T
stats_df['skew'] = df[['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V10', 'V12', 'V14', 'V17']].skew()
stats_csv = os.path.join(DATA_DIR, 'dataset_summary_statistics_q3.csv')
stats_df.to_csv(stats_csv)
print(f"Saved summary statistics table to {stats_csv}")

# 2. Visualisation 1 (Q4): Class Imbalance and Transaction Amount Distribution
print("\nGenerating Visualisation 1 (Q4)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# 1A: Class Distribution Bar Plot
palette = ['#1f77b4', '#d62728']
bars = axes[0].bar(['Legitimate (0)', 'Fraudulent (1)'], [n_legit, n_fraud], color=palette, width=0.5, edgecolor='black', linewidth=1)
axes[0].set_yscale('log')
axes[0].set_title('Transaction Class Distribution (Logarithmic Scale)', fontsize=12, fontweight='bold', pad=12)
axes[0].set_ylabel('Number of Transactions (Log Scale)', fontsize=11)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

for bar, count, pct in zip(bars, [n_legit, n_fraud], [pct_legit, pct_fraud]):
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2.0, yval * 1.5, f"{count:,}\n({pct:.2f}%)", ha='center', va='bottom', fontsize=10, fontweight='bold')

# 1B: Transaction Amount Boxplot
sns.boxplot(x='Class', y='Amount', data=df, ax=axes[1], palette=palette, showfliers=False, width=0.4)
axes[1].set_xticklabels(['Legitimate (0)', 'Fraudulent (1)'], fontsize=11)
axes[1].set_title('Transaction Amount Distribution by Class (Without Extreme Outliers)', fontsize=12, fontweight='bold', pad=12)
axes[1].set_xlabel('Transaction Class', fontsize=11)
axes[1].set_ylabel('Amount (EUR)', fontsize=11)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

med_legit = df[df['Class'] == 0]['Amount'].median()
med_fraud = df[df['Class'] == 1]['Amount'].median()
mean_legit = df[df['Class'] == 0]['Amount'].mean()
mean_fraud = df[df['Class'] == 1]['Amount'].mean()

axes[1].text(0, med_legit + 15, f"Median: €{med_legit:.2f}\nMean: €{mean_legit:.2f}", ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#1f77b4', alpha=0.9))
axes[1].text(1, med_fraud + 15, f"Median: €{med_fraud:.2f}\nMean: €{mean_fraud:.2f}", ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#d62728', alpha=0.9))

plt.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, 'figure_1_class_imbalance_and_amount_q4.png')
plt.savefig(fig1_path)
plt.close()
print(f"Saved Figure 1 to: {fig1_path}")

# 3. Visualisation 2 (Q4): Discriminative Feature Distributions (V14, V17, V12, V10)
print("\nGenerating Visualisation 2 (Q4)...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
top_features = ['V14', 'V17', 'V12', 'V10']

for idx, feat in enumerate(top_features):
    ax = axes[idx // 2, idx % 2]
    sns.kdeplot(df[df['Class'] == 0][feat], ax=ax, label='Legitimate (0)', color='#1f77b4', fill=True, alpha=0.35, linewidth=1.8)
    sns.kdeplot(df[df['Class'] == 1][feat], ax=ax, label='Fraudulent (1)', color='#d62728', fill=True, alpha=0.35, linewidth=1.8)
    ax.set_title(f'Feature Distribution: {feat} (Legitimate vs Fraud)', fontsize=12, fontweight='bold')
    ax.set_xlabel(f'{feat} Value', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, 'figure_2_predictive_features_distribution_q4.png')
plt.savefig(fig2_path)
plt.close()
print(f"Saved Figure 2 to: {fig2_path}")

# 4. Prepare WEKA-Ready ARFF Files
print("\nPreparing WEKA ARFF file...")
def export_to_arff(dataframe, filename, relation_name="creditcard"):
    target_path = os.path.join(DATA_DIR, filename)
    with open(target_path, 'w') as f:
        f.write(f"@relation {relation_name}\n\n")
        for col in dataframe.columns:
            if col == 'Class':
                f.write("@attribute Class {0,1}\n")
            else:
                f.write(f"@attribute {col} numeric\n")
        f.write("\n@data\n")
        for row in dataframe.itertuples(index=False):
            vals = [str(x) for x in row[:-1]]
            vals.append(str(int(row[-1])))
            f.write(",".join(vals) + "\n")
    print(f"Generated ARFF: {target_path} ({len(dataframe):,} rows)")

# Stratified sample: 5000 legitimate + all 492 frauds = 5,492 rows (perfect for responsive WEKA exploration)
legit_sample = df[df['Class'] == 0].sample(n=5000, random_state=42)
fraud_all = df[df['Class'] == 1]
sample_df = pd.concat([legit_sample, fraud_all]).sample(frac=1.0, random_state=42).reset_index(drop=True)

export_to_arff(sample_df, 'creditcard_weka_sample.arff', 'creditcard_sample')
sample_csv = os.path.join(DATA_DIR, 'creditcard_sample.csv')
sample_df.to_csv(sample_csv, index=False)
print(f"Saved stratified sample CSV: {sample_csv}")

print("\nAll preprocessing and visualization tasks completed successfully!")
