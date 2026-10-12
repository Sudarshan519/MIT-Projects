import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_curve, roc_curve, auc
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['font.size'] = 10

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'creditcard.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print("Loading dataset for visualization generation...")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['Class'])
y = df['Class']

# -------------------------------------------------------------
# Figure 3 (Q8 & Q9): Feature Selection & Importance Ranking
# -------------------------------------------------------------
print("Generating Figure 3: Feature Selection Analysis (Q8/Q9)...")
# Calculate Information Gain (Mutual Information) on stratified sample
legit_sub = df[df['Class'] == 0].sample(n=10000, random_state=42)
fraud_sub = df[df['Class'] == 1]
sub_df = pd.concat([legit_sub, fraud_sub]).reset_index(drop=True)
X_sub = sub_df.drop(columns=['Class'])
y_sub = sub_df['Class']

mi_scores = mutual_info_classif(X_sub, y_sub, random_state=42)
corr_scores = sub_df.corr()['Class'].drop('Class').abs()

fs_df = pd.DataFrame({
    'Feature': X_sub.columns,
    'InfoGain': mi_scores,
    'AbsCorrelation': corr_scores.values
}).sort_values(by='InfoGain', ascending=False)

fs_csv = os.path.join(DATA_DIR, 'feature_selection_rankings_q8_q9.csv')
fs_df.to_csv(fs_csv, index=False)
print(f"Saved feature selection ranking to {fs_csv}")

fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
top_15_ig = fs_df.head(15).sort_values(by='InfoGain', ascending=True)
axes[0].barh(top_15_ig['Feature'], top_15_ig['InfoGain'], color='#2b5c8f', edgecolor='black', linewidth=0.8)
axes[0].set_title('Top 15 Predictive Features by Information Gain (InfoGainAttributeEval)', fontsize=11, fontweight='bold', pad=10)
axes[0].set_xlabel('Information Gain (Mutual Information Score)', fontsize=10)
axes[0].grid(axis='x', linestyle='--', alpha=0.7)

top_15_corr = fs_df.sort_values(by='AbsCorrelation', ascending=True).tail(15)
axes[1].barh(top_15_corr['Feature'], top_15_corr['AbsCorrelation'], color='#c0392b', edgecolor='black', linewidth=0.8)
axes[1].set_title('Top 15 Features by Absolute Class Correlation (CFS Relevance)', fontsize=11, fontweight='bold', pad=10)
axes[1].set_xlabel('Absolute Pearson Correlation |r| with Class', fontsize=10)
axes[1].grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
fig3_path = os.path.join(FIGURES_DIR, 'figure_3_feature_selection_ranking_q8_q9.png')
plt.savefig(fig3_path)
plt.close()
print(f"Saved Figure 3 to: {fig3_path}")

# -------------------------------------------------------------
# Figure 4 (Q13): Confusion Matrices Side-by-Side
# -------------------------------------------------------------
print("Generating Figure 4: Confusion Matrices (Q13)...")
# WEKA 10-fold CV confusion matrices
cm_j48 = np.array([[4966, 34], [69, 423]])
cm_log = np.array([[4984, 16], [74, 418]])

fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

for ax, cm, title, cmap in zip(
    axes, 
    [cm_j48, cm_log], 
    ['J48 Decision Tree (10-Fold Cross-Validation)', 'Logistic Regression (10-Fold Cross-Validation)'],
    ['Blues', 'Reds']
):
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False, ax=ax,
                xticklabels=['Pred Legit (0)', 'Pred Fraud (1)'],
                yticklabels=['Actual Legit (0)', 'Actual Fraud (1)'],
                annot_kws={'size': 13, 'weight': 'bold'})
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('Ground Truth Class', fontsize=11)
    ax.set_xlabel('Predicted Class', fontsize=11)

plt.tight_layout()
fig4_path = os.path.join(FIGURES_DIR, 'figure_4_confusion_matrices_q13.png')
plt.savefig(fig4_path)
plt.close()
print(f"Saved Figure 4 to: {fig4_path}")

# -------------------------------------------------------------
# Figure 5 (Q13 & Q14): Comprehensive Metric Comparison
# -------------------------------------------------------------
print("Generating Figure 5: Model Metric Comparison (Q13/Q14)...")
metrics_data = {
    'Metric': ['Precision (Fraud)', 'Recall (Fraud)', 'F1-Measure (Fraud)', 'ROC-AUC', 'PRC-AUC', 'Accuracy'],
    'J48 Decision Tree': [0.926, 0.860, 0.891, 0.938, 0.817, 0.9812],
    'Logistic Regression': [0.963, 0.850, 0.903, 0.980, 0.937, 0.9836]
}
comp_df = pd.DataFrame(metrics_data)
comp_csv = os.path.join(DATA_DIR, 'model_performance_comparison_q13_q14.csv')
comp_df.to_csv(comp_csv, index=False)
print(f"Saved model performance table to {comp_csv}")

fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
x = np.arange(len(comp_df['Metric']))
width = 0.35

rects1 = ax.bar(x - width/2, comp_df['J48 Decision Tree'], width, label='J48 Decision Tree', color='#2b5c8f', edgecolor='black', linewidth=0.8)
rects2 = ax.bar(x + width/2, comp_df['Logistic Regression'], width, label='Logistic Regression', color='#c0392b', edgecolor='black', linewidth=0.8)

ax.set_title('Evaluation Metrics Comparison: J48 Decision Tree vs Logistic Regression (10-Fold CV)', fontsize=12, fontweight='bold', pad=14)
ax.set_ylabel('Score / Metric Value (0.0 to 1.0)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(comp_df['Metric'], fontsize=10, fontweight='bold')
ax.set_ylim(0.7, 1.02)
ax.legend(loc='lower right', frameon=True, fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.7)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.tight_layout()
fig5_path = os.path.join(FIGURES_DIR, 'figure_5_metrics_comparison_q13_q14.png')
plt.savefig(fig5_path)
plt.close()
print(f"Saved Figure 5 to: {fig5_path}")

# -------------------------------------------------------------
# Figure 6: Cost-Benefit Matrix & Trade-Off Analysis (Q14)
# -------------------------------------------------------------
print("Generating Figure 6: Financial Cost-Benefit Trade-Off Analysis (Q14)...")
# Financial cost scenario: False Negative costs €250 (average fraud loss), False Positive costs €15 (customer support / friction)
fn_cost = 250
fp_cost = 15

j48_fn, j48_fp = 69, 34
log_fn, log_fp = 74, 16

cost_j48_fn = j48_fn * fn_cost
cost_j48_fp = j48_fp * fp_cost
total_j48 = cost_j48_fn + cost_j48_fp

cost_log_fn = log_fn * fn_cost
cost_log_fp = log_fp * fp_cost
total_log = cost_log_fn + cost_log_fp

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
categories = ['J48 Decision Tree', 'Logistic Regression']
fn_costs = [cost_j48_fn, cost_log_fn]
fp_costs = [cost_j48_fp, cost_log_fp]

p1 = ax.bar(categories, fn_costs, width=0.45, label='Fraud Loss (False Negatives @ €250/case)', color='#e74c3c', edgecolor='black', linewidth=0.8)
p2 = ax.bar(categories, fp_costs, width=0.45, bottom=fn_costs, label='Investigation Friction (False Positives @ €15/case)', color='#f39c12', edgecolor='black', linewidth=0.8)

ax.set_title('Operational Cost Impact of Classification Errors (Q14)', fontsize=12, fontweight='bold', pad=12)
ax.set_ylabel('Total Estimated Cost (€ EUR)', fontsize=11)
ax.set_ylim(0, 22000)
ax.legend(loc='upper right', frameon=True, fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.7)

for i, (tot, fn, fp) in enumerate(zip([total_j48, total_log], fn_costs, fp_costs)):
    ax.text(i, tot + 400, f"Total: €{tot:,}\n(FN: €{fn:,} | FP: €{fp:,})", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

plt.tight_layout()
fig6_path = os.path.join(FIGURES_DIR, 'figure_6_cost_tradeoff_analysis_q14.png')
plt.savefig(fig6_path)
plt.close()
print(f"Saved Figure 6 to: {fig6_path}")

# -------------------------------------------------------------
# Figure 7: CRISP-DM Methodology Diagram (Q2)
# -------------------------------------------------------------
print("Generating Figure 7: CRISP-DM Process Architecture (Q2)...")
fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
ax.axis('off')

boxes = [
    ("1. Business\nUnderstanding", "Define fraud risk,\nfalse positive costs,\nregulatory audit"),
    ("2. Data\nUnderstanding", "284,807 txns,\n31 attributes,\nimbalance (0.17%)"),
    ("3. Data\nPreparation", "Duplicate checks,\nAmount scaling,\nSMOTE balancing"),
    ("4. Modeling", "J48 Decision Tree\n& Logistic Reg.\n10-fold CV"),
    ("5. Evaluation", "Precision, Recall,\nROC-AUC, Cost\nMatrix Trade-off"),
    ("6. Deployment", "Real-time scoring,\nHybrid rule engine\nfor fraud triage")
]

x_starts = np.linspace(0.05, 0.85, 6)
box_w = 0.13
box_h = 0.65

colors = ['#2c3e50', '#34495e', '#2980b9', '#16a085', '#d35400', '#27ae60']

for i, (title, desc) in enumerate(boxes):
    xs = x_starts[i]
    rect = plt.Rectangle((xs, 0.2), box_w, box_h, facecolor=colors[i], edgecolor='black', linewidth=1.2, transform=ax.transAxes, zorder=2)
    ax.add_patch(rect)
    ax.text(xs + box_w/2, 0.62, title, transform=ax.transAxes, ha='center', va='center', color='white', fontweight='bold', fontsize=9, zorder=3)
    ax.text(xs + box_w/2, 0.38, desc, transform=ax.transAxes, ha='center', va='center', color='#ecf0f1', fontsize=7.5, zorder=3)
    
    if i < 5:
        ax.annotate('', xy=(x_starts[i+1] - 0.01, 0.525), xytext=(xs + box_w + 0.01, 0.525),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle="->", color='#333333', lw=2), zorder=1)

ax.set_title('CRISP-DM Cycle Tailored for Credit Card Fraud Detection (Q2)', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
fig7_path = os.path.join(FIGURES_DIR, 'figure_7_crisp_dm_methodology_q2.png')
plt.savefig(fig7_path)
plt.close()
print(f"Saved Figure 7 to: {fig7_path}")

print("\nAll 7 report visualization figures successfully created in project_1/figures/!")
