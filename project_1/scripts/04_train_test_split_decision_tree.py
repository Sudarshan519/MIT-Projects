# -*- coding: utf-8 -*-
"""
INFO812 Data Mining — Decision Tree & Logistic Regression Classifier Building in Scikit-learn
Adapted for Credit Card Fraud Detection using 70/30 Train-Test Split
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from imblearn.over_sampling import SMOTE
import pydotplus
from io import StringIO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'creditcard.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80)
print("SCIKIT-LEARN TRAIN-TEST SPLIT CLASSIFICATION WORKFLOW (CREDIT CARD FRAUD)")
print("=" * 80)

# 1. Loading Data
print("\n1. Loading Data...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

# 2. Feature Selection (Top predictive features identified in Q8/Q9)
print("\n2. Feature Selection...")
feature_cols = ['V14', 'V17', 'V12', 'V10', 'V11', 'V16', 'V4', 'V3', 'V7', 'V9', 'Amount']
X = df[feature_cols] # Features
y = df['Class']      # Target variable
print(f"Features selected: {feature_cols}")

# 3. Splitting Data (70% training and 30% test)
print("\n3. Splitting Data into 70% Train and 30% Test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"X_train shape: {X_train.shape}, y_train class counts: {np.bincount(y_train)}")
print(f"X_test shape:  {X_test.shape},  y_test class counts:  {np.bincount(y_test)}")

# 4. Optional Class Balancing on Training Set (SMOTE)
print("\n4. Applying SMOTE to rebalance training data...")
smote = SMOTE(sampling_strategy=0.2, random_state=42) # Balance minority to 20% of majority in training set
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Post-SMOTE y_train class counts: {np.bincount(y_train_res)}")

# 5. Building Model 1: Decision Tree Classifier
print("\n5. Building Decision Tree Classifier (Criterion=Entropy, Max Depth=4)...")
clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=5, random_state=42)
clf_tree = clf_tree.fit(X_train_res, y_train_res)

# Predict on test dataset
y_pred_tree = clf_tree.predict(X_test)
y_prob_tree = clf_tree.predict_proba(X_test)[:, 1]

# 6. Building Model 2: Logistic Regression (Algorithm Comparison for Q12-Q14)
print("\n6. Building Logistic Regression Classifier...")
clf_log = LogisticRegression(max_iter=1000, random_state=42)
clf_log = clf_log.fit(X_train_res, y_train_res)

y_pred_log = clf_log.predict(X_test)
y_prob_log = clf_log.predict_proba(X_test)[:, 1]

# 7. Evaluating the Models
print("\n" + "=" * 80)
print("EVALUATING MODEL PERFORMANCE ON 30% UNSEEN TEST DATA (85,443 TRANSACTIONS)")
print("=" * 80)

def print_evaluation(name, y_true, y_pred, y_prob):
    print(f"\n--- {name} Results ---")
    print(f"Accuracy:         {metrics.accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"Fraud Precision:  {metrics.precision_score(y_true, y_pred)*100:.2f}%")
    print(f"Fraud Recall:     {metrics.recall_score(y_true, y_pred)*100:.2f}%")
    print(f"Fraud F1-Score:   {metrics.f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC Area:     {metrics.roc_auc_score(y_true, y_prob):.4f}")
    print("\nConfusion Matrix:")
    cm = metrics.confusion_matrix(y_true, y_pred)
    print(f"  TN: {cm[0,0]:<6} | FP: {cm[0,1]:<5} (Legitimate)")
    print(f"  FN: {cm[1,0]:<6} | TP: {cm[1,1]:<5} (Fraudulent)")
    print("\nClassification Report:")
    print(metrics.classification_report(y_true, y_pred, target_names=['Legitimate (0)', 'Fraud (1)'], digits=4))

print_evaluation("Decision Tree Classifier", y_test, y_pred_tree, y_prob_tree)
print_evaluation("Logistic Regression Classifier", y_test, y_pred_log, y_prob_log)

# 8. Visualizing Decision Trees
print("\n8. Visualizing Decision Tree...")

# Method A: Matplotlib plot_tree (High-Resolution PNG)
tree_png_path = os.path.join(FIGURES_DIR, 'credit_card_decision_tree_plot.png')
plt.figure(figsize=(20, 10), dpi=300)
plot_tree(
    clf_tree,
    feature_names=feature_cols,
    class_names=['Legitimate', 'Fraud'],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Credit Card Fraud Detection — Decision Tree Architecture (Scikit-Learn)", fontsize=14, fontweight='bold', pad=15)
plt.savefig(tree_png_path, bbox_inches='tight')
plt.close()
print(f"Saved Decision Tree visualization (plot_tree) to: {tree_png_path}")

# Method B: Graphviz / Pydotplus (if graphviz dot binary is available)
try:
    dot_data = StringIO()
    export_graphviz(
        clf_tree,
        out_file=dot_data,
        filled=True,
        rounded=True,
        special_characters=True,
        feature_names=feature_cols,
        class_names=['Legitimate', 'Fraud']
    )
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graphviz_png_path = os.path.join(FIGURES_DIR, 'credit_card_tree_graphviz.png')
    graph.write_png(graphviz_png_path)
    print(f"Saved Graphviz Decision Tree visualization to: {graphviz_png_path}")
except Exception as e:
    print(f"Note: Pydotplus/Graphviz system binary note: {e} (plot_tree PNG successfully saved above!)")

# 9. Exporting Decision Tree Rules
rules_text = export_text(clf_tree, feature_names=feature_cols)
rules_out = os.path.join(BASE_DIR, 'data', 'train_test_decision_tree_rules.txt')
with open(rules_out, 'w') as f:
    f.write(rules_text)
print(f"Saved text decision rules to: {rules_out}")

print("\n--- SAMPLE EXTRACTED IF-THEN RULES ---")
print("\n".join(rules_text.split("\n")[:20]))

print("\nWorkflow completed successfully!")
