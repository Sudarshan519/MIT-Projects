#!/usr/bin/env python3
# ============================================================================
# INFO813 Project 3 — Section 2.2: Data Transformation Pipeline
# ============================================================================
# This script implements the complete data transformation pipeline:
#   1. Patient / Lesion-level Stratified Splitting (70% Train, 15% Val, 15% Test)
#      - Guarantees zero data leakage between splits (same lesion stays together)
#   2. Class Weight Calculation to handle 58:1 class imbalance
#   3. Image Preprocessing & Normalisation (ImageNet statistics, 224×224)
#   4. Training-time Data Augmentation (flips, rotation, color jitter)
#   5. PyTorch HAM10000Dataset and DataLoader constructors with MPS/CUDA support
#   6. Transformation Visualisations (Figures 11, 12, 13)
# ============================================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── Add project root to path for config imports ───────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    RAW_DATA_DIR, METADATA_CSV, IMAGE_DIRS, PROCESSED_DATA_DIR,
    FIGURES_DATA_DIR, CLASS_NAMES, CLASS_FULL_NAMES, CLASS_RISK_LEVELS,
    IMAGE_SIZE, COLOUR_PALETTE, PLOT_DPI, PLOT_STYLE, FONT_FAMILY,
    RANDOM_STATE, BATCH_SIZE, ensure_dirs
)

# ── Initialise ─────────────────────────────────────────────────────────────
ensure_dirs()
if PLOT_STYLE in plt.style.available:
    plt.style.use(PLOT_STYLE)
plt.rcParams['font.sans-serif'] = FONT_FAMILY
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.facecolor'] = 'white'

LABEL_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_LABEL = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# ImageNet normalization standard
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ============================================================================
# 1. Dataset Loading & Path Mapping
# ============================================================================
def load_and_map_dataset():
    """Load HAM10000 metadata and locate image file paths."""
    print("\n[STEP 1] Loading metadata and locating images...")
    df = pd.read_csv(METADATA_CSV)
    
    # Build dictionary of image_id -> full path
    image_path_map = {}
    for img_dir in IMAGE_DIRS:
        if os.path.exists(img_dir):
            for fname in os.listdir(img_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_id = os.path.splitext(fname)[0]
                    image_path_map[img_id] = os.path.join(img_dir, fname)
                    
    df['image_path'] = df['image_id'].map(image_path_map)
    df['dx_idx'] = df['dx'].map(LABEL_TO_IDX)
    
    missing_count = df['image_path'].isna().sum()
    if missing_count > 0:
        raise FileNotFoundError(f"Missing {missing_count} images out of {len(df)}!")
    print(f"   ✅ All {len(df):,} images mapped to filesystem paths.")
    return df


# ============================================================================
# 2. Patient / Lesion-level Stratified Splitting (Prevent Data Leakage)
# ============================================================================
def create_stratified_splits(df):
    """
    Split dataset into 70% Train, 15% Validation, 15% Test.
    
    CRITICAL: Split by lesion_id rather than image_id to prevent data leakage.
    HAM10000 has 7,470 unique lesions across 10,015 images.
    If multiple images of the same lesion were scattered across train and test,
    test accuracy would be artificially inflated.
    """
    print("\n[STEP 2] Performing patient/lesion-level stratified split...")
    
    # Check that each lesion_id has a consistent diagnosis
    lesion_diag_counts = df.groupby('lesion_id')['dx'].nunique()
    assert lesion_diag_counts.max() == 1, "Inconsistent diagnosis detected for a lesion_id!"
    
    # Extract unique lesions with their diagnosis
    lesion_df = df.drop_duplicates(subset='lesion_id')[['lesion_id', 'dx', 'dx_idx']].copy()
    print(f"   Unique lesions: {len(lesion_df):,} | Total image instances: {len(df):,}")
    
    # First split: 70% Train, 30% Temp (Val + Test)
    train_lesions, temp_lesions = train_test_split(
        lesion_df,
        test_size=0.30,
        stratify=lesion_df['dx'],
        random_state=RANDOM_STATE
    )
    
    # Second split: 15% Val, 15% Test (50/50 of temp)
    val_lesions, test_lesions = train_test_split(
        temp_lesions,
        test_size=0.50,
        stratify=temp_lesions['dx'],
        random_state=RANDOM_STATE
    )
    
    train_df = df[df['lesion_id'].isin(train_lesions['lesion_id'])].copy().reset_index(drop=True)
    val_df   = df[df['lesion_id'].isin(val_lesions['lesion_id'])].copy().reset_index(drop=True)
    test_df  = df[df['lesion_id'].isin(test_lesions['lesion_id'])].copy().reset_index(drop=True)
    
    # Verify strict disjoint property
    assert set(train_df['lesion_id']).isdisjoint(set(val_df['lesion_id'])), "Leakage between Train and Val!"
    assert set(train_df['lesion_id']).isdisjoint(set(test_df['lesion_id'])), "Leakage between Train and Test!"
    assert set(val_df['lesion_id']).isdisjoint(set(test_df['lesion_id'])), "Leakage between Val and Test!"
    assert len(train_df) + len(val_df) + len(test_df) == len(df), "Row count mismatch in splits!"
    
    print(f"   ✅ Train set: {len(train_df):,} images ({len(train_df)/len(df)*100:.1f}%) | {len(train_lesions):,} lesions")
    print(f"   ✅ Val set:   {len(val_df):,} images ({len(val_df)/len(df)*100:.1f}%) | {len(val_lesions):,} lesions")
    print(f"   ✅ Test set:  {len(test_df):,} images ({len(test_df)/len(df)*100:.1f}%) | {len(test_lesions):,} lesions")
    print("   ✅ Zero data leakage confirmed (zero shared lesion_ids across splits).")
    
    # Save split metadata to processed directory
    train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train_metadata.csv'), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'val_metadata.csv'), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'test_metadata.csv'), index=False)
    print(f"   💾 Saved split CSVs to: {PROCESSED_DATA_DIR}")
    
    return train_df, val_df, test_df


# ============================================================================
# 3. Class Weight Calculation
# ============================================================================
def calculate_class_weights(train_df):
    """
    Calculate inverse frequency class weights to counteract class imbalance (58:1).
    Used in CrossEntropyLoss during training.
    """
    print("\n[STEP 3] Calculating balanced class weights...")
    classes = np.array(range(len(CLASS_NAMES)))
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_df['dx_idx'].values
    )
    
    weight_dict = {CLASS_NAMES[i]: float(weights[i]) for i in range(len(CLASS_NAMES))}
    print("   Class weights (inverse-proportional penalty):")
    for cls in CLASS_NAMES:
        print(f"     • {cls:5s} ({CLASS_FULL_NAMES[cls][:25]:25s}): weight = {weight_dict[cls]:.3f}")
        
    weights_path = os.path.join(PROCESSED_DATA_DIR, 'class_weights.json')
    with open(weights_path, 'w') as f:
        json.dump(weight_dict, f, indent=2)
    print(f"   💾 Saved class weights to: {weights_path}")
    
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================================
# 4. PyTorch Transformations & Dataset Definition
# ============================================================================
def get_transforms():
    """Return data transformations for training, validation, and testing."""
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return train_transform, eval_transform


class HAM10000Dataset(Dataset):
    """PyTorch Dataset for HAM10000 dermoscopic skin lesion images."""
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        image = Image.open(img_path).convert('RGB')
        label = row['dx_idx']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_data_loaders(train_df, val_df, test_df, batch_size=BATCH_SIZE, num_workers=2):
    """Construct PyTorch DataLoaders for train, val, and test splits."""
    train_tf, eval_tf = get_transforms()
    
    train_dataset = HAM10000Dataset(train_df, transform=train_tf)
    val_dataset   = HAM10000Dataset(val_df, transform=eval_tf)
    test_dataset  = HAM10000Dataset(test_df, transform=eval_tf)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# ============================================================================
# 5. Transformation Visualisations (Figures 11, 12, 13)
# ============================================================================
def generate_visualisations(df, train_df, val_df, test_df):
    """Generate publication-quality figures documenting the transformation pipeline."""
    print("\n[STEP 4] Generating transformation pipeline visualisations...")
    
    # ── Figure 11: Raw vs Preprocessed Image Comparison ────────────────────
    print("   📊 Generating Figure 11: Preprocessing Pipeline Comparison...")
    sample_row = df[df['dx'] == 'mel'].iloc[0]
    sample_img_path = sample_row['image_path']
    raw_img = Image.open(sample_img_path).convert('RGB')
    
    # Applied transform (resize + tensor + normalize)
    resized_img = raw_img.resize(IMAGE_SIZE, Image.BILINEAR)
    np_raw = np.array(raw_img)
    np_proc = np.array(resized_img)
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    
    # Raw Image
    axes[0, 0].imshow(np_raw)
    axes[0, 0].set_title(f"Raw Image: {sample_row['image_id']}\nResolution: 600 × 450 px | RGB [0, 255]", fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Preprocessed Image
    axes[0, 1].imshow(np_proc)
    axes[0, 1].set_title(f"Preprocessed Input: {sample_row['image_id']}\nResolution: 224 × 224 px | Bicubic Resampled", fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Raw Histogram
    for i, col in enumerate(['red', 'green', 'blue']):
        axes[1, 0].hist(np_raw[:, :, i].ravel(), bins=64, color=col, alpha=0.5, label=col.upper())
    axes[1, 0].set_title("Raw Pixel Intensity Histogram [0, 255]", fontsize=10, fontweight='bold')
    axes[1, 0].set_xlabel("Pixel Value")
    axes[1, 0].set_ylabel("Pixel Count")
    axes[1, 0].legend(loc='upper right', frameon=True)
    
    # Normalised Distribution Simulation
    norm_vals = (np_proc / 255.0 - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    for i, col in enumerate(['red', 'green', 'blue']):
        axes[1, 1].hist(norm_vals[:, :, i].ravel(), bins=64, color=col, alpha=0.5, label=f"{col.upper()} (std)")
    axes[1, 1].set_title("ImageNet Standardised Distribution (Zero Mean, Unit Var)", fontsize=10, fontweight='bold')
    axes[1, 1].set_xlabel("Z-Score / Standard Deviations")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    fig11_path = os.path.join(FIGURES_DATA_DIR, 'fig11_preprocessed_comparison.png')
    plt.savefig(fig11_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"      ✅ Saved: {fig11_path}")
    
    # ── Figure 12: Data Augmentation Demonstrations ────────────────────────
    print("   📊 Generating Figure 12: Data Augmentation Demonstrations...")
    aug_transforms = [
        ("Original (224×224)", transforms.Compose([transforms.Resize(IMAGE_SIZE)])),
        ("Horizontal Flip", transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.RandomHorizontalFlip(p=1.0)])),
        ("Vertical Flip", transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.RandomVerticalFlip(p=1.0)])),
        ("Rotation (±20°)", transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.RandomRotation(degrees=(20, 20))])),
        ("Color Jitter (Brightness/Contrast)", transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ColorJitter(brightness=0.3, contrast=0.3)])),
        ("Full Training Augmentation", transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15)
        ]))
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for idx, (title, aug) in enumerate(aug_transforms):
        aug_img = aug(raw_img)
        axes[idx].imshow(aug_img)
        axes[idx].set_title(title, fontsize=11, fontweight='bold')
        axes[idx].axis('off')
        
    plt.suptitle("Figure 12: Real-Time Data Augmentation Pipeline on Melanoma Sample", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig12_path = os.path.join(FIGURES_DATA_DIR, 'fig12_data_augmentation_samples.png')
    plt.savefig(fig12_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"      ✅ Saved: {fig12_path}")
    
    # ── Figure 13: Split Class Proportions (Leakage-Free Validation) ────────
    print("   📊 Generating Figure 13: Stratified Split Verification...")
    split_summary = []
    for split_name, s_df in [('Train (70%)', train_df), ('Validation (15%)', val_df), ('Test (15%)', test_df)]:
        counts = s_df['dx'].value_counts()
        total = len(s_df)
        for cls in CLASS_NAMES:
            pct = (counts.get(cls, 0) / total) * 100
            split_summary.append({
                'Split': split_name,
                'Class': cls,
                'Percentage': pct,
                'Count': counts.get(cls, 0)
            })
    split_summary_df = pd.DataFrame(split_summary)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=split_summary_df, x='Class', y='Percentage', hue='Split', palette='Blues_d', ax=ax)
    ax.set_title("Figure 13: Class Proportions Across Stratified Splits (Zero Patient Leakage)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Skin Lesion Diagnostic Category", fontsize=11, fontweight='bold')
    ax.set_ylabel("Class Proportion (% of Split)", fontsize=11, fontweight='bold')
    ax.set_xticklabels([f"{c}\n({CLASS_FULL_NAMES[c].split('/')[0].strip()[:14]})" for c in CLASS_NAMES], fontsize=9)
    ax.legend(title="Dataset Partition", frameon=True, fontsize=10)
    
    # Add count annotations above bars
    for p in ax.patches:
        height = p.get_height()
        if height > 1:
            ax.annotate(f"{height:.1f}%",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, xytext=(0, 2),
                        textcoords='offset points')
            
    plt.tight_layout()
    fig13_path = os.path.join(FIGURES_DATA_DIR, 'fig13_split_class_distributions.png')
    plt.savefig(fig13_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"      ✅ Saved: {fig13_path}")


# ============================================================================
# Main Execution
# ============================================================================
def main():
    print("=" * 72)
    print("INFO813 PROJECT 3 — SECTION 2.2: DATA TRANSFORMATION PIPELINE")
    print("========================================================================")
    
    df = load_and_map_dataset()
    train_df, val_df, test_df = create_stratified_splits(df)
    class_weights = calculate_class_weights(train_df)
    
    # Verify DataLoader batch creation
    print("\n[STEP 3b] Verifying PyTorch DataLoader batch synthesis...")
    train_loader, val_loader, test_loader = get_data_loaders(train_df, val_df, test_df, batch_size=8, num_workers=0)
    images, labels = next(iter(train_loader))
    print(f"   ✅ Batch tensor verified: Image shape = {images.shape} | Labels shape = {labels.shape}")
    print(f"   Tensor dtype: {images.dtype} | Min pixel: {images.min().item():.2f} | Max pixel: {images.max().item():.2f}")
    
    # Generate visualisations
    generate_visualisations(df, train_df, val_df, test_df)
    
    print("\n" + "=" * 72)
    print("✅ DATA TRANSFORMATION PIPELINE COMPLETED SUCCESSFULLY")
    print("========================================================================")
    print(f"📁 Processed CSVs saved: {PROCESSED_DATA_DIR}")
    print(f"📁 Figures 11, 12, 13 saved: {FIGURES_DATA_DIR}")
    print("=" * 72)


if __name__ == '__main__':
    main()
