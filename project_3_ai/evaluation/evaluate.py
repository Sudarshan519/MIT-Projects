#!/usr/bin/env python3
# ============================================================================
# INFO813 Project 3 — Section 2.4: Model Evaluation Pipeline
# ============================================================================
# Evaluates CustomCNN and EfficientNet-B0 on the unseen holdout test set (1,481 images):
#   • Computes classification metrics (Accuracy, Balanced Acc, Macro/Weighted F1)
#   • Computes per-class Sensitivity, Specificity, Precision, Recall
#   • Focuses on clinical safety: Melanoma Recall / Sensitivity
#   • Generates publication-grade figures:
#       - Figure 15: Side-by-side Confusion Matrices (Raw Counts & Normalised)
#       - Figure 16: Multi-Class ROC-AUC Curves (One-vs-Rest)
#       - Figure 17: Multi-Class Precision-Recall Curves
#       - Figure 18: Comparative Performance Bar Chart
#       - Figure 19: Grad-CAM Explainability Visualisations
#   • Exports CSV comparison tables and JSON summary
# ============================================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

import torch
import torch.nn.functional as F

# ── Add project root to path for config imports ───────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_EVAL_DIR, FIGURES_GRADCAM_DIR,
    CLASS_NAMES, CLASS_FULL_NAMES, CLASS_RISK_LEVELS, COLOUR_PALETTE,
    PLOT_DPI, PLOT_STYLE, FONT_FAMILY, ensure_dirs
)
from analysis.02_data_transformation import HAM10000Dataset, get_transforms
from models.architectures import CustomCNN, EfficientNetB0Transfer
from models.gradcam import GradCAM, overlay_heatmap_on_image

ensure_dirs()
if PLOT_STYLE in plt.style.available:
    plt.style.use(PLOT_STYLE)
plt.rcParams['font.sans-serif'] = FONT_FAMILY
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.facecolor'] = 'white'


# ============================================================================
# 1. Prediction Generation on Test Set
# ============================================================================
def predict_test_set(model, loader, device):
    """Generate logits, probabilities, and class predictions for holdout set."""
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(outputs.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
            
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_labels, all_preds, all_probs


# ============================================================================
# 2. Comprehensive Metric Calculation
# ============================================================================
def compute_metrics(y_true, y_pred, y_prob, model_name):
    """Compute overall, macro, weighted, and per-class diagnostic metrics."""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    p_cls, r_cls, f1_cls, supp_cls = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    # Calculate specificity per class (True Negatives / (True Negatives + False Positives))
    cm = confusion_matrix(y_true, y_pred)
    spec_cls = []
    for i in range(len(CLASS_NAMES)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        spec_cls.append(spec)
        
    mel_idx = CLASS_NAMES.index('mel')
    mel_recall = r_cls[mel_idx]
    
    overall = {
        'Model': model_name,
        'Accuracy': float(acc),
        'Balanced_Accuracy': float(bal_acc),
        'Macro_Precision': float(p_macro),
        'Macro_Recall': float(r_macro),
        'Macro_F1': float(f1_macro),
        'Weighted_F1': float(f1_wt),
        'Melanoma_Recall_Sensitivity': float(mel_recall)
    }
    
    per_class = []
    for i, cls in enumerate(CLASS_NAMES):
        per_class.append({
            'Model': model_name,
            'Class': cls,
            'Full_Name': CLASS_FULL_NAMES[cls],
            'Risk': CLASS_RISK_LEVELS[cls],
            'Support': int(supp_cls[i]),
            'Precision': float(p_cls[i]),
            'Recall_Sensitivity': float(r_cls[i]),
            'Specificity': float(spec_cls[i]),
            'F1_Score': float(f1_cls[i])
        })
        
    return overall, per_class, cm


# ============================================================================
# 3. Figure 15: Side-by-Side Confusion Matrices
# ============================================================================
def plot_confusion_matrices(cm_cnn, cm_eff):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Custom CNN - Counts
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0, 0], cbar=False)
    axes[0, 0].set_title("Custom CNN: Confusion Matrix (Raw Counts)", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel("True Diagnosis", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("Predicted Diagnosis", fontsize=11, fontweight='bold')
    
    # Custom CNN - Normalised
    cm_cnn_norm = cm_cnn.astype('float') / cm_cnn.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_cnn_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0, 1])
    axes[0, 1].set_title("Custom CNN: Confusion Matrix (Normalised Recall)", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel("True Diagnosis", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("Predicted Diagnosis", fontsize=11, fontweight='bold')
    
    # EfficientNet-B0 - Counts
    sns.heatmap(cm_eff, annot=True, fmt='d', cmap='Greens',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1, 0], cbar=False)
    axes[1, 0].set_title("EfficientNet-B0: Confusion Matrix (Raw Counts)", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("True Diagnosis", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel("Predicted Diagnosis", fontsize=11, fontweight='bold')
    
    # EfficientNet-B0 - Normalised
    cm_eff_norm = cm_eff.astype('float') / cm_eff.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_eff_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1, 1])
    axes[1, 1].set_title("EfficientNet-B0: Confusion Matrix (Normalised Recall)", fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel("True Diagnosis", fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel("Predicted Diagnosis", fontsize=11, fontweight='bold')
    
    plt.suptitle("Figure 15: Cross-Model Confusion Matrix Comparison on 1,481 Test Lesions",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig15_path = os.path.join(FIGURES_EVAL_DIR, 'fig15_confusion_matrices.png')
    plt.savefig(fig15_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved Figure 15 to: {fig15_path}")


# ============================================================================
# 4. Figure 16 & 17: Multi-Class ROC-AUC and Precision-Recall Curves
# ============================================================================
def plot_roc_and_pr_curves(y_true, y_prob_cnn, y_prob_eff):
    # One-hot encode ground truth
    y_bin = label_binarize(y_true, classes=range(len(CLASS_NAMES)))
    
    # ── ROC Curves ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for i, cls in enumerate(CLASS_NAMES):
        fpr_c, tpr_c, _ = roc_curve(y_bin[:, i], y_prob_cnn[:, i])
        auc_c = auc(fpr_c, tpr_c)
        axes[0].plot(fpr_c, tpr_c, label=f"{cls} (AUC = {auc_c:.3f})", color=COLOUR_PALETTE[cls], lw=1.8)
        
        fpr_e, tpr_e, _ = roc_curve(y_bin[:, i], y_prob_eff[:, i])
        auc_e = auc(fpr_e, tpr_e)
        axes[1].plot(fpr_e, tpr_e, label=f"{cls} (AUC = {auc_e:.3f})", color=COLOUR_PALETTE[cls], lw=1.8)
        
    for ax, title in zip(axes, ["Custom CNN (Baseline)", "EfficientNet-B0 (Transfer Learning)"]):
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
        ax.set_title(f"ROC Curves — {title}", fontsize=12, fontweight='bold')
        ax.legend(loc="lower right", frameon=True, fontsize=9)
        ax.grid(True, alpha=0.3)
        
    plt.suptitle("Figure 16: Multi-Class One-vs-Rest ROC-AUC Diagnostic Performance",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig16_path = os.path.join(FIGURES_EVAL_DIR, 'fig16_roc_auc_curves.png')
    plt.savefig(fig16_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved Figure 16 to: {fig16_path}")
    
    # ── Precision-Recall Curves ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for i, cls in enumerate(CLASS_NAMES):
        p_c, r_c, _ = precision_recall_curve(y_bin[:, i], y_prob_cnn[:, i])
        ap_c = average_precision_score(y_bin[:, i], y_prob_cnn[:, i])
        axes[0].plot(r_c, p_c, label=f"{cls} (AP = {ap_c:.3f})", color=COLOUR_PALETTE[cls], lw=1.8)
        
        p_e, r_e, _ = precision_recall_curve(y_bin[:, i], y_prob_eff[:, i])
        ap_e = average_precision_score(y_bin[:, i], y_prob_eff[:, i])
        axes[1].plot(r_e, p_e, label=f"{cls} (AP = {ap_e:.3f})", color=COLOUR_PALETTE[cls], lw=1.8)
        
    for ax, title in zip(axes, ["Custom CNN (Baseline)", "EfficientNet-B0 (Transfer Learning)"]):
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall (Sensitivity)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Precision (PPV)', fontsize=11, fontweight='bold')
        ax.set_title(f"Precision-Recall Curves — {title}", fontsize=12, fontweight='bold')
        ax.legend(loc="lower left", frameon=True, fontsize=9)
        ax.grid(True, alpha=0.3)
        
    plt.suptitle("Figure 17: Precision-Recall Curves (Addressing Class Imbalance Impact)",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig17_path = os.path.join(FIGURES_EVAL_DIR, 'fig17_precision_recall_curves.png')
    plt.savefig(fig17_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved Figure 17 to: {fig17_path}")


# ============================================================================
# 5. Figure 18: Comparative Performance Bar Chart
# ============================================================================
def plot_comparative_performance(df_per_class):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.barplot(data=df_per_class, x='Class', y='Recall_Sensitivity', hue='Model',
                palette=['#3498db', '#2ecc71'], ax=axes[0])
    axes[0].set_title("Per-Class Recall / Clinical Sensitivity", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Lesion Class", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Recall Rate", fontsize=11, fontweight='bold')
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)
    
    sns.barplot(data=df_per_class, x='Class', y='F1_Score', hue='Model',
                palette=['#3498db', '#2ecc71'], ax=axes[1])
    axes[1].set_title("Per-Class F1-Score (Harmonic Mean of Precision & Recall)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Lesion Class", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("F1 Score", fontsize=11, fontweight='bold')
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("Figure 18: Custom CNN vs EfficientNet-B0 Per-Class Performance Comparison",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig18_path = os.path.join(FIGURES_EVAL_DIR, 'fig18_comparative_performance.png')
    plt.savefig(fig18_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved Figure 18 to: {fig18_path}")


# ============================================================================
# 6. Figure 19: Grad-CAM Explainability Heatmaps
# ============================================================================
def generate_gradcam_comparisons(model_cnn, model_eff, test_df, device):
    """Generates visual attention maps highlighting model decision regions."""
    print("   📊 Generating Figure 19: Grad-CAM Explainability Overlays...")
    cam_cnn = GradCAM(model_cnn, model_cnn.get_last_conv_layer())
    cam_eff = GradCAM(model_eff, model_eff.get_last_conv_layer())
    
    _, eval_tf = get_transforms()
    
    # Pick 4 representative cases: melanoma, basal cell carcinoma, actinic keratosis, benign nevus
    target_classes = ['mel', 'bcc', 'akiec', 'nv']
    selected_samples = []
    for cls in target_classes:
        match = test_df[test_df['dx'] == cls]
        if len(match) > 0:
            selected_samples.append(match.iloc[0])
            
    fig, axes = plt.subplots(len(selected_samples), 4, figsize=(16, 4 * len(selected_samples)))
    
    for row_idx, sample in enumerate(selected_samples):
        img_path = sample['image_path']
        raw_pil = Image.open(img_path).convert('RGB')
        raw_resized = raw_pil.resize((224, 224))
        img_tensor = eval_tf(raw_pil).unsqueeze(0).to(device)
        true_dx = sample['dx']
        
        # Custom CNN Heatmap
        hm_cnn, pred_cnn_idx = cam_cnn.generate_heatmap(img_tensor)
        over_cnn, _ = overlay_heatmap_on_image(raw_resized, hm_cnn)
        pred_cnn = CLASS_NAMES[pred_cnn_idx]
        
        # EfficientNet-B0 Heatmap
        hm_eff, pred_eff_idx = cam_eff.generate_heatmap(img_tensor)
        over_eff, _ = overlay_heatmap_on_image(raw_resized, hm_eff)
        pred_eff = CLASS_NAMES[pred_eff_idx]
        
        # Col 0: Original Dermoscopy Image
        axes[row_idx, 0].imshow(raw_resized)
        axes[row_idx, 0].set_title(f"True: {true_dx.upper()} ({CLASS_FULL_NAMES[true_dx][:18]})\nID: {sample['image_id']}",
                                   fontsize=10, fontweight='bold')
        axes[row_idx, 0].axis('off')
        
        # Col 1: Custom CNN Heatmap
        axes[row_idx, 1].imshow(hm_cnn, cmap='jet')
        axes[row_idx, 1].set_title(f"Custom CNN Activation Map\nPred: {pred_cnn.upper()}", fontsize=10, fontweight='bold')
        axes[row_idx, 1].axis('off')
        
        # Col 2: Custom CNN Overlay
        axes[row_idx, 2].imshow(over_cnn)
        axes[row_idx, 2].set_title(f"Custom CNN Overlay\nPred: {pred_cnn.upper()}", fontsize=10, fontweight='bold')
        axes[row_idx, 2].axis('off')
        
        # Col 3: EfficientNet-B0 Overlay
        axes[row_idx, 3].imshow(over_eff)
        axes[row_idx, 3].set_title(f"EfficientNet-B0 Overlay\nPred: {pred_eff.upper()}", fontsize=10, fontweight='bold')
        axes[row_idx, 3].axis('off')
        
    plt.suptitle("Figure 19: Grad-CAM Diagnostic Explainability Across Dermatological Lesions",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig19_path = os.path.join(FIGURES_GRADCAM_DIR, 'fig19_gradcam_explanations.png')
    plt.savefig(fig19_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved Figure 19 to: {fig19_path}")


# ============================================================================
# Main Evaluation Controller
# ============================================================================
def main():
    print("=" * 72)
    print("INFO813 PROJECT 3 — SECTION 2.4: MODEL EVALUATION ON HOLDOUT TEST SET")
    print("========================================================================")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")
    
    test_csv = os.path.join(PROCESSED_DATA_DIR, 'test_metadata.csv')
    test_df = pd.read_csv(test_csv)
    print(f"Loaded unseen test partition: {len(test_df):,} images")
    
    _, eval_tf = get_transforms()
    test_dataset = HAM10000Dataset(test_df, transform=eval_tf)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # ── Load Custom CNN ───────────────────────────────────────────────────
    path_cnn = os.path.join(MODELS_DIR, 'custom_cnn_best.pt')
    if not os.path.exists(path_cnn):
        raise FileNotFoundError(f"Missing {path_cnn}! Run models/train.py first.")
    model_cnn = CustomCNN(num_classes=7).to(device)
    ckpt_cnn = torch.load(path_cnn, map_location=device)
    model_cnn.load_state_dict(ckpt_cnn['model_state_dict'])
    print(f"Loaded Custom CNN from {path_cnn}")
    
    # ── Load EfficientNet-B0 ──────────────────────────────────────────────
    path_eff = os.path.join(MODELS_DIR, 'efficientnet_b0_best.pt')
    if not os.path.exists(path_eff):
        raise FileNotFoundError(f"Missing {path_eff}! Run models/train.py first.")
    model_eff = EfficientNetB0Transfer(num_classes=7, pretrained=False).to(device)
    ckpt_eff = torch.load(path_eff, map_location=device)
    model_eff.load_state_dict(ckpt_eff['model_state_dict'])
    print(f"Loaded EfficientNet-B0 from {path_eff}")
    
    # ── Run Predictions ───────────────────────────────────────────────────
    print("\nRunning inference on holdout test set...")
    y_true, y_pred_cnn, y_prob_cnn = predict_test_set(model_cnn, test_loader, device)
    _, y_pred_eff, y_prob_eff = predict_test_set(model_eff, test_loader, device)
    
    # ── Compute Metrics ───────────────────────────────────────────────────
    overall_cnn, per_cls_cnn, cm_cnn = compute_metrics(y_true, y_pred_cnn, y_prob_cnn, "Custom CNN (Baseline)")
    overall_eff, per_cls_eff, cm_eff = compute_metrics(y_true, y_pred_eff, y_prob_eff, "EfficientNet-B0 (Transfer)")
    
    df_overall = pd.DataFrame([overall_cnn, overall_eff])
    df_per_cls = pd.DataFrame(per_cls_cnn + per_cls_eff)
    
    print("\n" + "=" * 72)
    print("MASTER MODEL COMPARISON TABLE (HOLDOUT TEST SET):")
    print("=" * 72)
    print(df_overall.to_string(index=False))
    
    # Save tables
    eval_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    df_overall.to_csv(os.path.join(eval_dir, 'overall_metrics_comparison.csv'), index=False)
    df_per_cls.to_csv(os.path.join(eval_dir, 'per_class_metrics_comparison.csv'), index=False)
    
    # Save test predictions for detailed error analysis
    test_df_preds = test_df.copy()
    test_df_preds['pred_custom_cnn'] = [CLASS_NAMES[i] for i in y_pred_cnn]
    test_df_preds['pred_efficientnet'] = [CLASS_NAMES[i] for i in y_pred_eff]
    test_df_preds.to_csv(os.path.join(eval_dir, 'test_set_predictions.csv'), index=False)
    
    # ── Generate All Visualisations ───────────────────────────────────────
    print("\nGenerating evaluation visualisations...")
    plot_confusion_matrices(cm_cnn, cm_eff)
    plot_roc_and_pr_curves(y_true, y_prob_cnn, y_prob_eff)
    plot_comparative_performance(df_per_cls)
    generate_gradcam_comparisons(model_cnn, model_eff, test_df, device)
    
    print("\n" + "=" * 72)
    print("✅ MODEL EVALUATION COMPLETED SUCCESSFULLY")
    print("========================================================================")
    print(f"📁 Evaluation CSVs: {eval_dir}")
    print(f"📁 Figures 15-18:   {FIGURES_EVAL_DIR}")
    print(f"📁 Figure 19:       {FIGURES_GRADCAM_DIR}")
    print("=" * 72)


if __name__ == '__main__':
    main()
