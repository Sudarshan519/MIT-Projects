#!/usr/bin/env python3
# ============================================================================
# INFO813 Project 3 — Section 2.3: Model Training Pipeline
# ============================================================================
# Trains:
#   1. CustomCNN (from-scratch baseline)
#   2. EfficientNet-B0 (transfer learning with 2-stage fine-tuning)
#
# Features:
#   • MPS (Apple Silicon GPU) acceleration
#   • Class-weighted CrossEntropyLoss (combating 58:1 class imbalance)
#   • Early Stopping & Model Checkpointing
#   • Learning Rate Scheduling
#   • Training & Validation Loss/Accuracy curves export (Figure 14)
# ============================================================================

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, balanced_accuracy_score

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ── Add project root to path for config imports ───────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_EVAL_DIR,
    CLASS_NAMES, BATCH_SIZE, RANDOM_STATE, PLOT_DPI, PLOT_STYLE, ensure_dirs
)
from analysis.02_data_transformation import HAM10000Dataset, get_transforms
from models.architectures import CustomCNN, EfficientNetB0Transfer

ensure_dirs()
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ============================================================================
# Device Configuration (MPS / CUDA / CPU)
# ============================================================================
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("⚡ Using Apple Silicon GPU via Metal Performance Shaders (MPS)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"⚡ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠️ Using CPU device")
    return device


# ============================================================================
# Data Loaders
# ============================================================================
def load_datasets(batch_size=BATCH_SIZE):
    train_csv = os.path.join(PROCESSED_DATA_DIR, 'train_metadata.csv')
    val_csv   = os.path.join(PROCESSED_DATA_DIR, 'val_metadata.csv')
    
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError("Processed split CSVs not found! Run 02_data_transformation.py first.")
        
    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    
    train_tf, eval_tf = get_transforms()
    
    train_dataset = HAM10000Dataset(train_df, transform=train_tf)
    val_dataset   = HAM10000Dataset(val_df, transform=eval_tf)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, drop_last=False
    )
    
    # Load class weights
    weights_path = os.path.join(PROCESSED_DATA_DIR, 'class_weights.json')
    with open(weights_path, 'r') as f:
        weight_dict = json.load(f)
    weights = torch.tensor([weight_dict[cls] for cls in CLASS_NAMES], dtype=torch.float32)
    
    return train_loader, val_loader, weights


# ============================================================================
# Epoch Train & Validate Steps
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, macro_f1, bal_acc


# ============================================================================
# Training Routine with Early Stopping
# ============================================================================
def run_training(model, train_loader, val_loader, criterion, optimizer, scheduler,
                 epochs, save_path, device, model_name, patience=7):
    best_macro_f1 = 0.0
    best_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_macro_f1': [], 'val_bal_acc': []
    }
    
    print(f"\n🚀 Training {model_name} for up to {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_bal_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_macro_f1'].append(val_f1)
        history['val_bal_acc'].append(val_bal_acc)
        
        ep_time = time.time() - ep_start
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] ({ep_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | "
              f"Macro F1: {val_f1:.4f} Bal Acc: {val_bal_acc*100:.2f}%", end="")
        
        # Save best model based on validation macro F1
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, save_path)
            print(f"  ⭐ [BEST: F1={val_f1:.4f}]")
        else:
            patience_counter += 1
            print(f"  (patience {patience_counter}/{patience})")
            
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping triggered at epoch {epoch}.")
            break
            
    total_time = (time.time() - start_time) / 60
    print(f"\n🏁 Finished training {model_name} in {total_time:.1f} min | Best Val Macro F1: {best_macro_f1:.4f}")
    return history


# ============================================================================
# Plot Training Curves (Figure 14)
# ============================================================================
def plot_learning_curves(hist_cnn, hist_eff):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Custom CNN Loss & Acc
    if hist_cnn:
        epochs_cnn = range(1, len(hist_cnn['train_loss']) + 1)
        axes[0, 0].plot(epochs_cnn, hist_cnn['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs_cnn, hist_cnn['val_loss'], 'r--', label='Val Loss')
        axes[0, 0].set_title('Custom CNN: Loss Progression', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs_cnn, [a*100 for a in hist_cnn['train_acc']], 'b-', label='Train Acc')
        axes[0, 1].plot(epochs_cnn, [a*100 for a in hist_cnn['val_acc']], 'r--', label='Val Acc')
        axes[0, 1].plot(epochs_cnn, [f*100 for f in hist_cnn['val_macro_f1']], 'g-.', label='Val Macro F1')
        axes[0, 1].set_title('Custom CNN: Accuracy & F1 Progression', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
    # 2. EfficientNet-B0 Loss & Acc
    if hist_eff:
        epochs_eff = range(1, len(hist_eff['train_loss']) + 1)
        axes[1, 0].plot(epochs_eff, hist_eff['train_loss'], 'b-', label='Train Loss')
        axes[1, 0].plot(epochs_eff, hist_eff['val_loss'], 'r--', label='Val Loss')
        axes[1, 0].set_title('EfficientNet-B0: Loss Progression (Transfer Learning)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(epochs_eff, [a*100 for a in hist_eff['train_acc']], 'b-', label='Train Acc')
        axes[1, 1].plot(epochs_eff, [a*100 for a in hist_eff['val_acc']], 'r--', label='Val Acc')
        axes[1, 1].plot(epochs_eff, [f*100 for f in hist_eff['val_macro_f1']], 'g-.', label='Val Macro F1')
        axes[1, 1].set_title('EfficientNet-B0: Accuracy & F1 Progression', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
    plt.suptitle("Figure 14: Model Training & Validation Trajectories (Cross-Model Comparison)",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig14_path = os.path.join(FIGURES_EVAL_DIR, 'fig14_training_curves.png')
    plt.savefig(fig14_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Saved Figure 14 to: {fig14_path}")


# ============================================================================
# Main Controller
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train Skin Lesion Classification Models")
    parser.add_argument('--model', choices=['custom_cnn', 'efficientnet_b0', 'all'], default='all',
                        help='Which model to train')
    parser.add_argument('--epochs_cnn', type=int, default=15, help='Epochs for Custom CNN')
    parser.add_argument('--epochs_head', type=int, default=8, help='Epochs for EfficientNet frozen head')
    parser.add_argument('--epochs_fine', type=int, default=8, help='Epochs for EfficientNet fine-tuning')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()
    
    device = get_device()
    train_loader, val_loader, class_weights = load_datasets(batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    hist_cnn, hist_eff = None, None
    
    # ── 1. Train Custom CNN (Baseline) ────────────────────────────────────
    if args.model in ['custom_cnn', 'all']:
        print("\n" + "=" * 72)
        print("PHASE 1: TRAINING CUSTOM CNN (FROM SCRATCH BASELINE)")
        print("=" * 72)
        model_cnn = CustomCNN(num_classes=7).to(device)
        optimizer_cnn = AdamW(model_cnn.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler_cnn = ReduceLROnPlateau(optimizer_cnn, mode='min', factor=0.5, patience=3)
        save_path_cnn = os.path.join(MODELS_DIR, 'custom_cnn_best.pt')
        
        hist_cnn = run_training(
            model=model_cnn, train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer_cnn, scheduler=scheduler_cnn,
            epochs=args.epochs_cnn, save_path=save_path_cnn, device=device,
            model_name="Custom CNN", patience=5
        )
        with open(os.path.join(MODELS_DIR, 'training_history_custom_cnn.json'), 'w') as f:
            json.dump(hist_cnn, f, indent=2)
            
    # ── 2. Train EfficientNet-B0 (Transfer Learning) ──────────────────────
    if args.model in ['efficientnet_b0', 'all']:
        print("\n" + "=" * 72)
        print("PHASE 2: TRAINING EFFICIENTNET-B0 (TRANSFER LEARNING)")
        print("=" * 72)
        model_eff = EfficientNetB0Transfer(num_classes=7, pretrained=True).to(device)
        save_path_eff = os.path.join(MODELS_DIR, 'efficientnet_b0_best.pt')
        
        # Stage A: Train Classifier Head Only
        print("\n[Stage A] Training custom medical classification head (backbone frozen)...")
        model_eff.freeze_backbone()
        optimizer_head = AdamW(filter(lambda p: p.requires_grad, model_eff.parameters()), lr=1e-3, weight_decay=1e-4)
        scheduler_head = ReduceLROnPlateau(optimizer_head, mode='min', factor=0.5, patience=2)
        
        hist_stage_a = run_training(
            model=model_eff, train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer_head, scheduler=scheduler_head,
            epochs=args.epochs_head, save_path=save_path_eff, device=device,
            model_name="EfficientNet-B0 (Head Only)", patience=4
        )
        
        # Stage B: Fine-tune Top Convolutional Layers
        print("\n[Stage B] Fine-tuning top MBConv convolutional blocks...")
        # Reload best weights from Stage A
        checkpoint = torch.load(save_path_eff, map_location=device)
        model_eff.load_state_dict(checkpoint['model_state_dict'])
        
        model_eff.unfreeze_top_layers(num_blocks=3)
        optimizer_fine = AdamW([
            {'params': [p for n, p in model_eff.model.named_parameters() if 'classifier' not in n and p.requires_grad], 'lr': 1e-4},
            {'params': model_eff.model.classifier.parameters(), 'lr': 5e-4}
        ], weight_decay=1e-4)
        scheduler_fine = ReduceLROnPlateau(optimizer_fine, mode='min', factor=0.5, patience=2)
        
        hist_stage_b = run_training(
            model=model_eff, train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer_fine, scheduler=scheduler_fine,
            epochs=args.epochs_fine, save_path=save_path_eff, device=device,
            model_name="EfficientNet-B0 (Fine-Tuning)", patience=4
        )
        
        # Concatenate history
        hist_eff = {k: hist_stage_a[k] + hist_stage_b[k] for k in hist_stage_a}
        with open(os.path.join(MODELS_DIR, 'training_history_efficientnet_b0.json'), 'w') as f:
            json.dump(hist_eff, f, indent=2)
            
    # Generate Figure 14
    plot_learning_curves(hist_cnn, hist_eff)
    print("\n✅ Training phase completed successfully!")


if __name__ == '__main__':
    main()
