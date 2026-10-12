#!/usr/bin/env python3
# ============================================================================
# INFO813 Project 3 — Section 1: Data Analysis & Visualisation
# ============================================================================
# This script performs comprehensive exploratory data analysis (EDA) on the
# HAM10000 dermoscopic skin lesion dataset. It generates all visualisations
# required for Sections 1.1 (Description, 10 marks) and 1.2 (Visualisation,
# 10 marks) of the assignment report.
#
# Output:
#   - Summary statistics printed to console and saved as CSV
#   - 10 publication-quality visualisation figures saved to figures/data_analysis/
#
# Usage:
#   cd project_3_ai
#   python analysis/01_data_analysis.py
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
from collections import Counter
from glob import glob

# ── Add project root to path for config imports ───────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    RAW_DATA_DIR, METADATA_CSV, IMAGE_DIRS, FIGURES_DATA_DIR,
    PROCESSED_DATA_DIR, CLASS_NAMES, CLASS_FULL_NAMES, CLASS_RISK_LEVELS,
    COLOUR_PALETTE, PLOT_DPI, PLOT_STYLE, FONT_FAMILY, ensure_dirs
)

# ── Initialise ─────────────────────────────────────────────────────────────
ensure_dirs()
if PLOT_STYLE in plt.style.available:
    plt.style.use(PLOT_STYLE)
plt.rcParams['font.sans-serif'] = FONT_FAMILY
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.facecolor'] = 'white'

print("=" * 72)
print("INFO813 PROJECT 3 — SKIN LESION DATA ANALYSIS")
print("HAM10000 Dataset: Dermoscopic Images of Pigmented Skin Lesions")
print("=" * 72)


# ============================================================================
# STEP 1: Load and Validate Dataset
# ============================================================================
print("\n[STEP 1] Loading HAM10000 metadata...")

if not os.path.exists(METADATA_CSV):
    print(f"\n❌ ERROR: Metadata file not found at:\n   {METADATA_CSV}")
    print("\n📥 Please download the HAM10000 dataset from:")
    print("   https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
    print("\n📂 Extract into: data/raw/")
    print("   Expected files:")
    print("     - data/raw/HAM10000_metadata.csv")
    print("     - data/raw/HAM10000_images_part_1/  (contains .jpg files)")
    print("     - data/raw/HAM10000_images_part_2/  (contains .jpg files)")
    sys.exit(1)

df = pd.read_csv(METADATA_CSV)
print(f"   ✅ Metadata loaded successfully. Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

# Build image path lookup
print("\n[STEP 1b] Mapping images to file paths...")
image_path_map = {}
for img_dir in IMAGE_DIRS:
    if os.path.exists(img_dir):
        for img_file in os.listdir(img_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_id = os.path.splitext(img_file)[0]
                image_path_map[img_id] = os.path.join(img_dir, img_file)

df['image_path'] = df['image_id'].map(image_path_map)
found = df['image_path'].notna().sum()
missing = df['image_path'].isna().sum()
print(f"   ✅ Mapped {found:,} images | ❌ Missing: {missing:,}")

if missing > 0 and found == 0:
    print("\n⚠️  No images found. Please check that image directories exist:")
    for d in IMAGE_DIRS:
        print(f"     {d} — {'EXISTS' if os.path.exists(d) else 'NOT FOUND'}")
    print("\n   Continuing with metadata-only analysis...\n")


# ============================================================================
# STEP 2: Dataset Overview (Section 1.1)
# ============================================================================
print("\n" + "=" * 72)
print("SECTION 1.1 — DATASET DESCRIPTION & STATISTICS")
print("=" * 72)

print(f"\n{'─' * 50}")
print(f"{'PROPERTY':<30} {'VALUE'}")
print(f"{'─' * 50}")
print(f"{'Total Images:':<30} {len(df):,}")
print(f"{'Number of Classes:':<30} {df['dx'].nunique()}")
print(f"{'Image Format:':<30} JPEG (.jpg)")
print(f"{'Metadata Fields:':<30} {len(df.columns)}")
print(f"{'Missing Values (total):':<30} {df.isnull().sum().sum()}")
print(f"{'Unique Lesions (lesion_id):':<30} {df['lesion_id'].nunique():,}")
print(f"{'Duplicate Lesion Images:':<30} {len(df) - df['lesion_id'].nunique():,}")
print(f"{'─' * 50}")

# Class distribution
print("\n📊 Class Distribution:")
print(f"{'─' * 65}")
print(f"{'Code':<8} {'Full Name':<40} {'Count':>6} {'%':>7} {'Risk'}")
print(f"{'─' * 65}")

class_counts = df['dx'].value_counts()
for cls in CLASS_NAMES:
    count = class_counts.get(cls, 0)
    pct = (count / len(df)) * 100
    risk = CLASS_RISK_LEVELS.get(cls, 'Unknown')
    full_name = CLASS_FULL_NAMES.get(cls, cls)
    print(f"{cls:<8} {full_name:<40} {count:>6,} {pct:>6.2f}% {risk}")

print(f"{'─' * 65}")
print(f"{'TOTAL':<49} {len(df):>6,} {'100.00%':>7}")

majority = class_counts.max()
minority = class_counts.min()
print(f"\nImbalance Ratio (majority:minority): {majority}:{minority} = {majority/minority:.1f}:1")

# Metadata analysis
print("\n📋 Metadata Variable Summary:")
print(f"{'─' * 50}")
for col in df.columns:
    dtype = df[col].dtype
    nunique = df[col].nunique()
    nulls = df[col].isnull().sum()
    print(f"  {col:<20} type={str(dtype):<10} unique={nunique:<8} nulls={nulls}")

# Age statistics
if 'age' in df.columns:
    print(f"\n👤 Age Statistics:")
    age_stats = df['age'].describe()
    print(f"   Mean: {age_stats['mean']:.1f} years")
    print(f"   Std:  {age_stats['std']:.1f} years")
    print(f"   Min:  {age_stats['min']:.0f} | Max: {age_stats['max']:.0f}")
    print(f"   Missing: {df['age'].isnull().sum()}")

# Sex distribution
if 'sex' in df.columns:
    print(f"\n⚥ Sex Distribution:")
    sex_counts = df['sex'].value_counts(dropna=False)
    for sex, count in sex_counts.items():
        print(f"   {sex}: {count:,} ({count/len(df)*100:.1f}%)")

# Localisation
if 'localization' in df.columns:
    print(f"\n📍 Body Localisation (top 5):")
    loc_counts = df['localization'].value_counts().head(5)
    for loc, count in loc_counts.items():
        print(f"   {loc}: {count:,} ({count/len(df)*100:.1f}%)")

# Save summary statistics
summary_path = os.path.join(PROCESSED_DATA_DIR, 'dataset_summary_statistics.csv')
summary_data = []
for cls in CLASS_NAMES:
    subset = df[df['dx'] == cls]
    summary_data.append({
        'class_code': cls,
        'class_name': CLASS_FULL_NAMES[cls],
        'count': len(subset),
        'percentage': len(subset) / len(df) * 100,
        'risk_level': CLASS_RISK_LEVELS[cls],
        'mean_age': subset['age'].mean() if 'age' in subset else None,
        'median_age': subset['age'].median() if 'age' in subset else None,
    })
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(summary_path, index=False)
print(f"\n💾 Saved summary statistics to: {summary_path}")


# ============================================================================
# STEP 3: Visualisations (Section 1.2)
# ============================================================================
print("\n" + "=" * 72)
print("SECTION 1.2 — DATA VISUALISATIONS")
print("=" * 72)

palette_list = [COLOUR_PALETTE[cls] for cls in CLASS_NAMES]


# ── FIGURE 1: Class Distribution ──────────────────────────────────────────
print("\n📊 Generating Figure 1: Class Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=PLOT_DPI)

# Bar chart
counts = [class_counts.get(cls, 0) for cls in CLASS_NAMES]
labels = [f"{cls}\n({CLASS_FULL_NAMES[cls].split('/')[0].strip()[:18]})" for cls in CLASS_NAMES]
bars = axes[0].bar(labels, counts, color=palette_list, edgecolor='black', linewidth=0.5)
axes[0].set_title('Skin Lesion Class Distribution', fontsize=14, fontweight='bold', pad=15)
axes[0].set_ylabel('Number of Images', fontsize=12)
axes[0].set_xlabel('Lesion Type', fontsize=12)
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

for bar, count in zip(bars, counts):
    pct = count / len(df) * 100
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Pie chart
axes[1].pie(counts, labels=CLASS_NAMES, colors=palette_list, autopct='%1.1f%%',
            startangle=90, pctdistance=0.85, textprops={'fontsize': 10})
centre_circle = plt.Circle((0, 0), 0.55, fc='white')
axes[1].add_artist(centre_circle)
axes[1].set_title('Class Proportion Distribution', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
fig1_path = os.path.join(FIGURES_DATA_DIR, 'fig01_class_distribution.png')
plt.savefig(fig1_path, bbox_inches='tight')
plt.close()
print(f"   ✅ Saved: {fig1_path}")


# ── FIGURE 2: Sample Image Grid ──────────────────────────────────────────
if found > 0:
    print("\n🖼️  Generating Figure 2: Sample Image Grid (7 classes × 3 examples)...")
    fig, axes = plt.subplots(7, 3, figsize=(12, 24), dpi=PLOT_DPI)

    for row, cls in enumerate(CLASS_NAMES):
        cls_images = df[df['dx'] == cls].dropna(subset=['image_path'])
        samples = cls_images.sample(n=min(3, len(cls_images)), random_state=42)

        for col, (_, sample) in enumerate(samples.iterrows()):
            img = Image.open(sample['image_path'])
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_title(
                    f"{cls.upper()} — {CLASS_FULL_NAMES[cls]}",
                    fontsize=10, fontweight='bold', loc='left', pad=8,
                    color=COLOUR_PALETTE[cls]
                )
            else:
                axes[row, col].set_title(f"Sample {col+1}", fontsize=9)

    plt.suptitle('Sample Dermoscopic Images by Lesion Type',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DATA_DIR, 'fig02_sample_images_grid.png')
    plt.savefig(fig2_path, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {fig2_path}")
else:
    print("\n⚠️  Skipping Figure 2 (no images found)")


# ── FIGURE 3: Age Distribution by Lesion Type ────────────────────────────
if 'age' in df.columns:
    print("\n📊 Generating Figure 3: Age Distribution by Lesion Type...")
    fig, ax = plt.subplots(figsize=(14, 7), dpi=PLOT_DPI)

    df_age = df.dropna(subset=['age'])
    sns.violinplot(x='dx', y='age', data=df_age, order=CLASS_NAMES,
                   palette=COLOUR_PALETTE, inner='box', ax=ax, linewidth=0.8)
    ax.set_title('Patient Age Distribution by Lesion Type', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Lesion Type', fontsize=12)
    ax.set_ylabel('Patient Age (Years)', fontsize=12)
    ax.set_xticklabels([f"{cls}\n({CLASS_FULL_NAMES[cls].split('/')[0].strip()[:15]})"
                        for cls in CLASS_NAMES], fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Add mean age annotations
    for i, cls in enumerate(CLASS_NAMES):
        mean_age = df_age[df_age['dx'] == cls]['age'].mean()
        ax.text(i, mean_age + 2, f'μ={mean_age:.0f}', ha='center', fontsize=8,
                fontweight='bold', color=COLOUR_PALETTE[cls])

    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DATA_DIR, 'fig03_age_distribution_by_class.png')
    plt.savefig(fig3_path, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {fig3_path}")


# ── FIGURE 4: Sex Distribution per Class ──────────────────────────────────
if 'sex' in df.columns:
    print("\n📊 Generating Figure 4: Sex Distribution per Lesion Type...")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=PLOT_DPI)

    sex_class = df.groupby(['dx', 'sex']).size().unstack(fill_value=0)
    sex_class = sex_class.reindex(CLASS_NAMES)
    sex_colours = {'male': '#3498db', 'female': '#e74c3c', 'unknown': '#95a5a6'}
    sex_class.plot(kind='bar', stacked=True, ax=ax,
                   color=[sex_colours.get(c, '#999') for c in sex_class.columns],
                   edgecolor='black', linewidth=0.3)
    ax.set_title('Sex Distribution per Lesion Type', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Lesion Type', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.legend(title='Sex', fontsize=10)
    ax.set_xticklabels(CLASS_NAMES, rotation=0, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DATA_DIR, 'fig04_sex_distribution_by_class.png')
    plt.savefig(fig4_path, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {fig4_path}")


# ── FIGURE 5: Body Localisation Distribution ──────────────────────────────
if 'localization' in df.columns:
    print("\n📊 Generating Figure 5: Body Localisation Distribution...")
    fig, ax = plt.subplots(figsize=(14, 8), dpi=PLOT_DPI)

    loc_class = pd.crosstab(df['localization'], df['dx'])
    loc_class = loc_class[CLASS_NAMES]
    loc_order = loc_class.sum(axis=1).sort_values(ascending=True).index
    loc_class = loc_class.reindex(loc_order)

    loc_class.plot(kind='barh', stacked=True, ax=ax, color=palette_list,
                   edgecolor='black', linewidth=0.3)
    ax.set_title('Lesion Distribution by Body Localisation', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Number of Images', fontsize=12)
    ax.set_ylabel('Body Location', fontsize=12)
    ax.legend(title='Lesion Type', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig5_path = os.path.join(FIGURES_DATA_DIR, 'fig05_localisation_distribution.png')
    plt.savefig(fig5_path, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {fig5_path}")


# ── FIGURE 6: Diagnosis Confirmation Method ──────────────────────────────
if 'dx_type' in df.columns:
    print("\n📊 Generating Figure 6: Diagnosis Confirmation Method...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=PLOT_DPI)

    # Overall breakdown
    dx_type_counts = df['dx_type'].value_counts()
    dx_type_colours = {'histo': '#2ecc71', 'follow_up': '#3498db',
                       'consensus': '#e67e22', 'confocal': '#9b59b6'}
    axes[0].pie(dx_type_counts.values,
                labels=[f"{k}\n({v:,})" for k, v in dx_type_counts.items()],
                colors=[dx_type_colours.get(k, '#999') for k in dx_type_counts.index],
                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    axes[0].set_title('Overall Diagnosis Confirmation Methods', fontsize=13, fontweight='bold')

    # Per-class breakdown
    dx_type_class = pd.crosstab(df['dx'], df['dx_type'], normalize='index') * 100
    dx_type_class = dx_type_class.reindex(CLASS_NAMES)
    dx_type_class.plot(kind='bar', stacked=True, ax=axes[1],
                       color=[dx_type_colours.get(c, '#999') for c in dx_type_class.columns],
                       edgecolor='black', linewidth=0.3)
    axes[1].set_title('Confirmation Method by Lesion Type (%)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Lesion Type', fontsize=11)
    axes[1].set_ylabel('Percentage (%)', fontsize=11)
    axes[1].set_xticklabels(CLASS_NAMES, rotation=0, fontsize=10)
    axes[1].legend(title='Method', fontsize=9)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig6_path = os.path.join(FIGURES_DATA_DIR, 'fig06_diagnosis_confirmation_methods.png')
    plt.savefig(fig6_path, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {fig6_path}")


# ── FIGURE 7: Pixel Intensity (RGB) Distribution per Class ───────────────
if found > 0:
    print("\n📊 Generating Figure 7: RGB Pixel Intensity Distributions...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=PLOT_DPI)

    n_samples_per_class = 50  # Sample images per class for efficiency
    channels = ['Red', 'Green', 'Blue']
    channel_colours = ['#e74c3c', '#2ecc71', '#3498db']

    for idx, cls in enumerate(CLASS_NAMES):
        if idx >= 7:
            break
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        cls_images = df[df['dx'] == cls].dropna(subset=['image_path'])
        sample_imgs = cls_images.sample(n=min(n_samples_per_class, len(cls_images)), random_state=42)

        for ch_idx, (ch_name, ch_colour) in enumerate(zip(channels, channel_colours)):
            all_pixels = []
            for _, row_data in sample_imgs.iterrows():
                img = np.array(Image.open(row_data['image_path']).resize((64, 64)))
                all_pixels.extend(img[:, :, ch_idx].flatten())
            ax.hist(all_pixels, bins=50, alpha=0.5, color=ch_colour,
                    label=ch_name, density=True, edgecolor='none')

        ax.set_title(f'{cls.upper()} — {CLASS_FULL_NAMES[cls][:25]}',
                     fontsize=10, fontweight='bold', color=COLOUR_PALETTE[cls])
        ax.set_xlabel('Pixel Intensity', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 255)

    # Hide unused subplots
    for idx in range(len(CLASS_NAMES), 9):
        row, col = idx // 3, idx % 3
        axes[row, col].axis('off')

    plt.suptitle('RGB Pixel Intensity Distributions by Lesion Type',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig7_path = os.path.join(FIGURES_DATA_DIR, 'fig07_rgb_pixel_distributions.png')
    plt.savefig(fig7_path, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {fig7_path}")
else:
    print("\n⚠️  Skipping Figure 7 (no images found)")


# ── FIGURE 8: Mean Image per Class ───────────────────────────────────────
if found > 0:
    print("\n📊 Generating Figure 8: Mean Image per Class...")
    fig, axes = plt.subplots(1, 7, figsize=(21, 4), dpi=PLOT_DPI)
    target_size = (128, 128)

    for idx, cls in enumerate(CLASS_NAMES):
        cls_images = df[df['dx'] == cls].dropna(subset=['image_path'])
        sample_imgs = cls_images.sample(n=min(100, len(cls_images)), random_state=42)

        img_sum = np.zeros((*target_size, 3), dtype=np.float64)
        count = 0
        for _, row_data in sample_imgs.iterrows():
            try:
                img = np.array(Image.open(row_data['image_path']).resize(target_size))
                img_sum += img.astype(np.float64)
                count += 1
            except Exception:
                continue

        if count > 0:
            mean_img = (img_sum / count).astype(np.uint8)
            axes[idx].imshow(mean_img)
        axes[idx].axis('off')
        axes[idx].set_title(f'{cls.upper()}\n(n={len(cls_images):,})',
                           fontsize=10, fontweight='bold', color=COLOUR_PALETTE[cls])

    plt.suptitle('Average (Mean) Image per Lesion Class',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig8_path = os.path.join(FIGURES_DATA_DIR, 'fig08_mean_images_per_class.png')
    plt.savefig(fig8_path, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {fig8_path}")
else:
    print("\n⚠️  Skipping Figure 8 (no images found)")


# ── FIGURE 9: Image Dimension Analysis ───────────────────────────────────
if found > 0:
    print("\n📊 Generating Figure 9: Image Dimension Analysis...")
    widths, heights, sizes_kb = [], [], []
    sample_for_dims = df.dropna(subset=['image_path']).sample(
        n=min(500, found), random_state=42)

    for _, row_data in sample_for_dims.iterrows():
        try:
            img = Image.open(row_data['image_path'])
            w, h = img.size
            widths.append(w)
            heights.append(h)
            sizes_kb.append(os.path.getsize(row_data['image_path']) / 1024)
        except Exception:
            continue

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=PLOT_DPI)

    axes[0].hist(widths, bins=30, color='#3498db', edgecolor='black', linewidth=0.5, alpha=0.8)
    axes[0].set_title('Image Width Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Width (pixels)', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].axvline(np.median(widths), color='red', linestyle='--', label=f'Median: {np.median(widths):.0f}px')
    axes[0].legend()

    axes[1].hist(heights, bins=30, color='#2ecc71', edgecolor='black', linewidth=0.5, alpha=0.8)
    axes[1].set_title('Image Height Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Height (pixels)', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].axvline(np.median(heights), color='red', linestyle='--', label=f'Median: {np.median(heights):.0f}px')
    axes[1].legend()

    axes[2].hist(sizes_kb, bins=30, color='#e74c3c', edgecolor='black', linewidth=0.5, alpha=0.8)
    axes[2].set_title('Image File Size Distribution', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('File Size (KB)', fontsize=11)
    axes[2].set_ylabel('Count', fontsize=11)
    axes[2].axvline(np.median(sizes_kb), color='blue', linestyle='--', label=f'Median: {np.median(sizes_kb):.0f}KB')
    axes[2].legend()

    plt.suptitle('Image Dimension & File Size Analysis (HAM10000)',
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    fig9_path = os.path.join(FIGURES_DATA_DIR, 'fig09_image_dimensions_analysis.png')
    plt.savefig(fig9_path, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {fig9_path}")
else:
    print("\n⚠️  Skipping Figure 9 (no images found)")


# ── FIGURE 10: Metadata Correlation Matrix ────────────────────────────────
print("\n📊 Generating Figure 10: Metadata Correlation & Relationships...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=PLOT_DPI)

# Encode categorical variables for correlation
df_encoded = df.copy()
if 'dx' in df_encoded.columns:
    df_encoded['dx_code'] = pd.Categorical(df_encoded['dx'], categories=CLASS_NAMES).codes
if 'sex' in df_encoded.columns:
    df_encoded['sex_code'] = df_encoded['sex'].map({'male': 0, 'female': 1}).fillna(2)
if 'localization' in df_encoded.columns:
    df_encoded['loc_code'] = pd.Categorical(df_encoded['localization']).codes

corr_cols = [c for c in ['age', 'dx_code', 'sex_code', 'loc_code'] if c in df_encoded.columns]
if len(corr_cols) >= 2:
    corr_matrix = df_encoded[corr_cols].corr()
    corr_labels = {'age': 'Age', 'dx_code': 'Diagnosis', 'sex_code': 'Sex', 'loc_code': 'Location'}
    corr_matrix.index = [corr_labels.get(c, c) for c in corr_matrix.index]
    corr_matrix.columns = [corr_labels.get(c, c) for c in corr_matrix.columns]

    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, square=True,
                ax=axes[0], fmt='.3f', linewidths=0.5,
                cbar_kws={'label': 'Correlation Coefficient'})
    axes[0].set_title('Metadata Variable Correlations', fontsize=13, fontweight='bold')

# Age vs diagnosis scatter
if 'age' in df.columns:
    age_by_class = df.dropna(subset=['age']).groupby('dx')['age'].mean().reindex(CLASS_NAMES)
    bars = axes[1].bar(CLASS_NAMES, age_by_class.values, color=palette_list,
                       edgecolor='black', linewidth=0.5)
    axes[1].set_title('Mean Patient Age by Lesion Type', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Lesion Type', fontsize=11)
    axes[1].set_ylabel('Mean Age (Years)', fontsize=11)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, age_by_class.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
fig10_path = os.path.join(FIGURES_DATA_DIR, 'fig10_metadata_correlations.png')
plt.savefig(fig10_path, bbox_inches='tight')
plt.close()
print(f"   ✅ Saved: {fig10_path}")


# ============================================================================
# COMPLETE
# ============================================================================
print("\n" + "=" * 72)
print("✅ DATA ANALYSIS COMPLETE")
print("=" * 72)
print(f"\n📁 All figures saved to: {FIGURES_DATA_DIR}")
print(f"📁 Summary statistics saved to: {PROCESSED_DATA_DIR}")
print(f"\n📊 Figures generated:")
print(f"   1. Class Distribution (bar + pie)")
if found > 0:
    print(f"   2. Sample Image Grid (7×3)")
print(f"   3. Age Distribution by Lesion Type (violin)")
print(f"   4. Sex Distribution per Class (stacked bar)")
print(f"   5. Body Localisation Distribution (horizontal stacked bar)")
print(f"   6. Diagnosis Confirmation Methods (pie + bar)")
if found > 0:
    print(f"   7. RGB Pixel Intensity Distributions")
    print(f"   8. Mean Image per Class")
    print(f"   9. Image Dimension Analysis")
print(f"   10. Metadata Correlations (heatmap + bar)")
print(f"\n🔜 Next: Run analysis/02_data_transformation.py")
