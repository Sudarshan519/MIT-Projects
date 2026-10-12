"""
INFO812 Data Mining — Trimester 3, 2026
Credit Card Fraud Detection Complete Python Pipeline
Replaces WEKA with Native Scikit-Learn + Imbalanced-Learn
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from imblearn.over_sampling import SMOTE

# -------------------------------------------------------------
# Configuration & Directory Setup
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'creditcard.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80)
print("INFO812 DATA MINING: CREDIT CARD FRAUD DETECTION PIPELINE (PYTHON)")
print("=" * 80)

# -------------------------------------------------------------
# Phase 1: Data Understanding & Statistics (Q1 - Q3)
# -------------------------------------------------------------
print("\n[Phase 1] Loading and Analyzing Dataset...")
start_time = time.time()
df = pd.read_csv(DATA_PATH)
print(f"Dataset successfully loaded in {time.time() - start_time:.2f} seconds.")
print(f"Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

n_legit = (df['Class'] == 0).sum()
n_fraud = (df['Class'] == 1).sum()
print(f"Class 0 (Legitimate): {n_legit:,} ({n_legit/len(df)*100:.4f}%)")
print(f"Class 1 (Fraudulent): {n_fraud:,} ({n_fraud/len(df)*100:.4f}%)")
print(f"Class Imbalance Ratio: {n_legit/n_fraud:.1f} : 1")
print(f"Missing Values: {df.isnull().sum().sum()}")

# Export Summary Stats
stats_df = df.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
stats_df['skewness'] = df.skew()
stats_path = os.path.join(OUTPUT_DIR, 'python_summary_statistics_q3.csv')
stats_df.to_csv(stats_path)
print(f"Saved Q3 Summary Statistics to: {stats_path}")

# -------------------------------------------------------------
# Phase 2: Preprocessing & Scaling (Q5)
# -------------------------------------------------------------
print("\n[Phase 2] Preprocessing Data...")
# Standardize Amount and Time
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[['Amount', 'Time']] = scaler.fit_transform(df[['Amount', 'Time']])

X = df_scaled.drop(columns=['Class'])
y = df_scaled['Class']

# Create stratified sample for rapid feature evaluation
legit_sample = df_scaled[df_scaled['Class'] == 0].sample(n=10000, random_state=42)
fraud_sample = df_scaled[df_scaled['Class'] == 1]
eval_sample = pd.concat([legit_sample, fraud_sample]).reset_index(drop=True)

X_eval = eval_sample.drop(columns=['Class'])
y_eval = eval_sample['Class']

# -------------------------------------------------------------
# Phase 3: Feature Selection Analysis (Q6 - Q9)
# -------------------------------------------------------------
print("\n[Phase 3] Running Feature Selection (InfoGain & Correlation)...")
# Method 1: Information Gain (Mutual Information)
ig_scores = mutual_info_classif(X_eval, y_eval, random_state=42)

# Method 2: Pearson Correlation with Target Class
corr_scores = eval_sample.corr()['Class'].drop('Class').abs()

fs_results = pd.DataFrame({
    'Feature': X_eval.columns,
    'Information_Gain': ig_scores,
    'Abs_Correlation': corr_scores.values
}).sort_values(by='Information_Gain', ascending=False).reset_index(drop=True)

fs_path = os.path.join(OUTPUT_DIR, 'python_feature_selection_q8_q9.csv')
fs_results.to_csv(fs_path, index=False)
print("Top 10 Ranked Features by Information Gain:")
for idx, row in fs_results.head(10).iterrows():
    print(f"  {idx+1:2d}. {row['Feature']:<8} | InfoGain: {row['Information_Gain']:.4f} | |Corr|: {row['Abs_Correlation']:.4f}")

# Select Top 12 Predictive Features for Classification
selected_features = fs_results.head(12)['Feature'].tolist()
print(f"\nSelected 12 Predictive Features: {selected_features}")

# -------------------------------------------------------------
# Phase 4: Model Training & 10-Fold Cross-Validation (Q10 - Q14)
# -------------------------------------------------------------
print("\n[Phase 4] Training Classifiers with 10-Fold Stratified Cross-Validation...")

# Use the stratified evaluation dataset for cross-validation
X_sub = eval_sample[selected_features]
y_sub = eval_sample['Class']

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Models matching WEKA settings
tree_model = DecisionTreeClassifier(
    criterion='entropy',      # Information Gain splitting (C4.5 / J48 logic)
    min_samples_leaf=2,       # -M 2 in J48
    max_depth=5,              # Pruning constraint to prevent overfitting
    random_state=42
)

log_model = LogisticRegression(
    C=1.0,                    # Inverse regularisation (matches Ridge 1e-8)
    max_iter=1000,
    random_state=42
)

# Containers for CV predictions
y_true_all = []
y_pred_tree_all = []
y_pred_log_all = []
y_prob_tree_all = []
y_prob_log_all = []

# SMOTE object
smote = SMOTE(sampling_strategy=0.5, random_state=42) # Rebalance to 1:2 ratio on training folds

print("Executing 10 folds with SMOTE applied strictly to training partitions...")
fold = 1
for train_idx, test_idx in skf.split(X_sub, y_sub):
    X_train, X_test = X_sub.iloc[train_idx], X_sub.iloc[test_idx]
    y_train, y_test = y_sub.iloc[train_idx], y_sub.iloc[test_idx]
    
    # Apply SMOTE ONLY on training partition (Prevents Data Leakage!)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Train Decision Tree
    tree_model.fit(X_train_res, y_train_res)
    pred_tree = tree_model.predict(X_test)
    prob_tree = tree_model.predict_proba(X_test)[:, 1]
    
    # Train Logistic Regression
    log_model.fit(X_train_res, y_train_res)
    pred_log = log_model.predict(X_test)
    prob_log = log_model.predict_proba(X_test)[:, 1]
    
    y_true_all.extend(y_test)
    y_pred_tree_all.extend(pred_tree)
    y_pred_log_all.extend(pred_log)
    y_prob_tree_all.extend(prob_tree)
    y_prob_log_all.extend(prob_log)
    fold += 1

print("10-Fold Cross-Validation completed successfully!")

# -------------------------------------------------------------
# Phase 5: Metrics & Evaluation (Q11 - Q14)
# -------------------------------------------------------------
print("\n" + "=" * 80)
print("EVALUATION METRICS COMPARISON (10-FOLD STRATIFIED CROSS-VALIDATION)")
print("=" * 80)

def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision (Fraud)': precision_score(y_true, y_pred),
        'Recall (Fraud)': recall_score(y_true, y_pred),
        'F1-Score (Fraud)': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_prob),
        'PRC-AUC': average_precision_score(y_true, y_prob),
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn
    }

m_tree = compute_metrics(y_true_all, y_pred_tree_all, y_prob_tree_all)
m_log = compute_metrics(y_true_all, y_pred_log_all, y_prob_log_all)

comp_table = pd.DataFrame([
    {'Metric': 'Accuracy', 'Decision Tree': f"{m_tree['Accuracy']*100:.2f}%", 'Logistic Regression': f"{m_log['Accuracy']*100:.2f}%", 'Superior': 'Logistic Regression' if m_log['Accuracy'] > m_tree['Accuracy'] else 'Decision Tree'},
    {'Metric': 'Fraud Precision (TP/(TP+FP))', 'Decision Tree': f"{m_tree['Precision (Fraud)']*100:.2f}%", 'Logistic Regression': f"{m_log['Precision (Fraud)']*100:.2f}%", 'Superior': 'Logistic Regression' if m_log['Precision (Fraud)'] > m_tree['Precision (Fraud)'] else 'Decision Tree'},
    {'Metric': 'Fraud Recall (TP/(TP+FN))', 'Decision Tree': f"{m_tree['Recall (Fraud)']*100:.2f}%", 'Logistic Regression': f"{m_log['Recall (Fraud)']*100:.2f}%", 'Superior': 'Decision Tree' if m_tree['Recall (Fraud)'] > m_log['Recall (Fraud)'] else 'Logistic Regression'},
    {'Metric': 'Fraud F1-Score', 'Decision Tree': f"{m_tree['F1-Score (Fraud)']:.4f}", 'Logistic Regression': f"{m_log['F1-Score (Fraud)']:.4f}", 'Superior': 'Logistic Regression' if m_log['F1-Score (Fraud)'] > m_tree['F1-Score (Fraud)'] else 'Decision Tree'},
    {'Metric': 'ROC-AUC Area', 'Decision Tree': f"{m_tree['ROC-AUC']:.4f}", 'Logistic Regression': f"{m_log['ROC-AUC']:.4f}", 'Superior': 'Logistic Regression' if m_log['ROC-AUC'] > m_tree['ROC-AUC'] else 'Decision Tree'},
    {'Metric': 'PRC-AUC Area', 'Decision Tree': f"{m_tree['PRC-AUC']:.4f}", 'Logistic Regression': f"{m_log['PRC-AUC']:.4f}", 'Superior': 'Logistic Regression' if m_log['PRC-AUC'] > m_tree['PRC-AUC'] else 'Decision Tree'},
    {'Metric': 'True Positives (Fraud Caught)', 'Decision Tree': str(m_tree['TP']), 'Logistic Regression': str(m_log['TP']), 'Superior': 'Decision Tree' if m_tree['TP'] > m_log['TP'] else 'Logistic Regression'},
    {'Metric': 'False Positives (False Alarms)', 'Decision Tree': str(m_tree['FP']), 'Logistic Regression': str(m_log['FP']), 'Superior': 'Logistic Regression' if m_log['FP'] < m_tree['FP'] else 'Decision Tree'},
    {'Metric': 'False Negatives (Missed Fraud)', 'Decision Tree': str(m_tree['FN']), 'Logistic Regression': str(m_log['FN']), 'Superior': 'Decision Tree' if m_tree['FN'] < m_log['FN'] else 'Logistic Regression'}
])

print(comp_table.to_string(index=False))

# Commercial Cost Matrix Calculation (Q14)
fn_cost = 250
fp_cost = 15

cost_tree = (m_tree['FN'] * fn_cost) + (m_tree['FP'] * fp_cost)
cost_log = (m_log['FN'] * fn_cost) + (m_log['FP'] * fp_cost)

print("\n" + "=" * 80)
print("FINANCIAL COST-BENEFIT MATRIX (Q14 30-MARK ANALYSIS)")
print(f"Cost per False Negative (Unrecovered Fraud):  €{fn_cost}")
print(f"Cost per False Positive (Customer Friction):  €{fp_cost}")
print("-" * 80)
print(f"Decision Tree Total Loss:        €{cost_tree:,} (FN: €{m_tree['FN']*fn_cost:,} | FP: €{m_tree['FP']*fp_cost:,})")
print(f"Logistic Regression Total Loss:  €{cost_log:,} (FN: €{m_log['FN']*fn_cost:,} | FP: €{m_log['FP']*fp_cost:,})")
diff = cost_log - cost_tree
print(f"Commercial Impact: Decision Tree saves €{diff:,} due to catching more fraud!")
print("=" * 80)

# -------------------------------------------------------------
# Phase 6: Rule Extraction (Appendix D)
# -------------------------------------------------------------
print("\n[Phase 6] Extracting Decision Tree Rules for Appendix D...")
# Fit single tree on all data for rule export
tree_model.fit(X_sub, y_sub)
tree_rules = export_text(tree_model, feature_names=selected_features)

rules_path = os.path.join(OUTPUT_DIR, 'python_decision_tree_rules_appendix_d.txt')
with open(rules_path, 'w') as f:
    f.write(tree_rules)
print(f"Exported Decision Tree Rules to: {rules_path}")

print("\n--- SAMPLE DECISION RULES EXTRACTED ---")
print("\n".join(tree_rules.split("\n")[:25]))

print("\nAll Python data mining tasks completed flawlessly!")
