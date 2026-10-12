import os
import json
import nbformat as nbf

def generate_fraud_detection_notebook(filename="credit_card_fraud_detection.ipynb"):
    # Initialize a new v4 Jupyter Notebook object
    nb = nbf.v4.new_notebook()

    # Define cells (Markdown & Code)
    cells = [
        # Cell 1: Markdown - Title
        nbf.v4.new_markdown_cell("""# Credit Card Fraud Detection Pipeline

This notebook implements an end-to-end Machine Learning pipeline to detect fraudulent credit card transactions using XGBoost and Scikit-Learn.

### Key Steps:
1. **Dynamic File Detection:** Finds `creditcard.csv` dynamically in current or parent directories (including Kaggle inputs).
2. **Exploratory Data Analysis (EDA):** Checks class distributions and missing values.
3. **Preprocessing:** Feature scaling using `RobustScaler` on non-PCA features (`Amount`, `Time`).
4. **Model Training:** XGBoost classifier with cost-sensitive weighting (`scale_pos_weight`).
5. **Evaluation:** Precision-Recall AUC, ROC-AUC, Confusion Matrix, and Precision-Recall Curves."""),

        # Cell 2: Code - Setup and Path Resolution
        nbf.v4.new_code_cell("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    precision_recall_curve, 
    auc
)
from xgboost import XGBClassifier

# Styling setup
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# Dynamic file lookup (handles Kaggle, local, and subfolder directory setups)
file_path = None
search_dirs = ['.', '..', '../input']

for s_dir in search_dirs:
    if os.path.exists(s_dir):
        for root, dirs, files in os.walk(s_dir):
            if 'creditcard.csv' in files:
                file_path = os.path.join(root, 'creditcard.csv')
                break
        if file_path:
            break

if file_path:
    print(f"Found dataset at: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset Shape: {df.shape}")
else:
    raise FileNotFoundError("Could not locate 'creditcard.csv'. Please verify your working directory.")

df.head()"""),

        # Cell 3: Markdown - EDA
        nbf.v4.new_markdown_cell("""## 1. Exploratory Data Analysis & Imbalance Check"""),

        # Cell 4: Code - Class Imbalance Visualization
        nbf.v4.new_code_cell("""# Check missing values
print(f"Total Missing Values: {df.isnull().sum().sum()}")

# Class distribution
class_counts = df['Class'].value_counts()
class_percentages = df['Class'].value_counts(normalize=True) * 100

print("\\nClass Counts:")
print(class_counts)
print("\\nPercentage Distribution:")
print(class_percentages)

# Visualizing target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df, palette='Set1')
plt.title('Transaction Class Distribution (0: Legit, 1: Fraud)')
plt.yscale('log')
plt.ylabel('Count (Log Scale)')
plt.show()"""),

        # Cell 5: Markdown - Scaling
        nbf.v4.new_markdown_cell("""## 2. Preprocessing & Stratified Splitting

* Features `V1` through `V28` are already PCA-transformed.
* `Time` and `Amount` require scaling. We use `RobustScaler` due to extreme financial outliers.
* A **Stratified Train-Test Split** preserves target ratios across train and test sets."""),

        # Cell 6: Code - Scaling & Train-Test Split
        nbf.v4.new_code_cell("""# Copy dataframe for preprocessing
df_scaled = df.copy()

# Scale Amount and Time using RobustScaler
scaler = RobustScaler()
df_scaled['scaled_amount'] = scaler.fit_transform(df_scaled['Amount'].values.reshape(-1, 1))
df_scaled['scaled_time'] = scaler.fit_transform(df_scaled['Time'].values.reshape(-1, 1))

# Drop original unscaled columns
df_scaled.drop(['Time', 'Amount'], axis=1, inplace=True)

# Separate features (X) and target (y)
X = df_scaled.drop('Class', axis=1)
y = df_scaled['Class']

# Stratified split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")"""),

        # Cell 7: Markdown - Training
        nbf.v4.new_markdown_cell("""## 3. XGBoost Model Training

We use `scale_pos_weight` inside XGBoost to directly handle class imbalance without synthetic sampling (SMOTE)."""),

        # Cell 8: Code - Training XGBoost
        nbf.v4.new_code_cell("""# Compute weight ratio: negative_count / positive_count
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

# Train XGBoost Classifier
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)
print("Model training complete!")"""),

        # Cell 9: Markdown - Evaluation
        nbf.v4.new_markdown_cell("""## 4. Evaluation & Metrics

Due to extreme target imbalance, we evaluate performance using **Precision-Recall AUC (PR-AUC)**, **ROC-AUC**, and a **Confusion Matrix**."""),

        # Cell 10: Code - Model Evaluation & Plots
        nbf.v4.new_code_cell("""# Predictions & Probabilities
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

# Metric calculations
precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)
roc_auc = roc_auc_score(y_test, y_probs)

print("--- Classification Report ---")
print(classification_report(y_test, y_pred, digits=4))

print(f"PR-AUC Score:  {pr_auc:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Plotting Confusion Matrix and Precision-Recall Curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# 2. Precision-Recall Curve
axes[1].plot(recall, precision, label=f'XGBoost (PR-AUC = {pr_auc:.4f})', color='purple')
axes[1].set_title('Precision-Recall Curve')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend(loc='lower left')

plt.tight_layout()
plt.show()""")
    ]

    nb['cells'] = cells

    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"Successfully exported Jupyter Notebook to '{filename}'.")

if __name__ == "__main__":
    generate_fraud_detection_notebook()