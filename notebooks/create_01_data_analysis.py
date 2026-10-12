"""
Generate the Data Analysis Jupyter Notebook (Section 1 of INFO813 Project).
Run this script to create: notebooks/01_data_analysis.ipynb

This notebook covers:
- 1.1 Description and Explanation (10 marks)
- 1.2 Visualisation (10 marks)
"""

import os
import nbformat as nbf


def create_data_analysis_notebook(output_path="notebooks/01_data_analysis.ipynb"):
    nb = nbf.v4.new_notebook()

    cells = [
        # =====================================================================
        # TITLE
        # =====================================================================
        nbf.v4.new_markdown_cell("""# Plant Disease Detection — Data Analysis
## INFO813 Artificial Intelligence | Trimester 3, 2026

This notebook covers **Section 1: Data Analysis** of the project report.

### Table of Contents
1. [1.1 Description and Explanation](#1.1-Description-and-Explanation)
2. [1.2 Visualisation](#1.2-Visualisation)

---

### Dataset: PlantVillage
- **Source:** Penn State University (Hughes & Salathé, 2015)
- **Platform:** Kaggle — [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Total Images:** ~54,303
- **Classes:** 38 (14 crop species × healthy + various diseases)
- **Image Format:** JPG, 256×256 pixels, RGB"""),

        # =====================================================================
        # IMPORTS & SETUP
        # =====================================================================
        nbf.v4.new_code_cell("""# =============================================================================
# Import Libraries
# =============================================================================
import os
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from collections import Counter
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print("Setup complete!")"""),

        # =====================================================================
        # 1.1 DESCRIPTION AND EXPLANATION
        # =====================================================================
        nbf.v4.new_markdown_cell("""---
## 1.1 Description and Explanation

### Problem Definition

**Plant disease detection** is the task of identifying diseases in plants by analysing images
of their leaves. This is a **multi-class image classification problem** where the goal is to
assign each leaf image to one of 38 categories representing different crop-disease combinations.

### Why is this an appropriate problem for an intelligent system?

1. **Visual Pattern Recognition:** Plant diseases manifest as visible symptoms on leaves —
   spots, discoloration, wilting, lesions, and mould growth. These visual patterns are
   well-suited for Convolutional Neural Networks (CNNs), which excel at hierarchical
   feature extraction from images.

2. **Scale and Speed:** Manual disease identification by agricultural experts is slow,
   expensive, and not scalable. An AI system can process thousands of images in seconds,
   enabling rapid diagnosis across large farming operations.

3. **Real-World Impact:** Crop diseases cause an estimated **20–40% of global agricultural
   yield loss** annually (FAO, 2019). Early, automated detection can help farmers take
   timely corrective action, reducing crop loss and improving food security.

4. **Complexity:** With 38 classes across 14 crop species, the classification task is
   sufficiently complex to require intelligent algorithms rather than simple rule-based
   systems. Some diseases share similar visual symptoms, making this a challenging
   classification problem.

### Data Source

The **PlantVillage dataset** was created by researchers at **Penn State University**
(Hughes & Salathé, 2015) as part of the PlantVillage project — an open-access platform
aimed at providing farmers with tools for plant health management. The images were
collected under **controlled laboratory conditions** with uniform backgrounds (gray/black)
to facilitate computer vision research.

**Reference:**
- Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant
  health to enable the development of mobile disease diagnostics. *arXiv preprint
  arXiv:1511.08060*.

The dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)."""),

        # =====================================================================
        # LOAD DATASET
        # =====================================================================
        nbf.v4.new_code_cell("""# =============================================================================
# Load and Explore the Dataset
# =============================================================================

# Set the path to the dataset directory
# The dataset should be organized as: data/<class_name>/<image_files>
# Adjust this path based on your directory structure
DATASET_DIR = os.path.join('..', 'data')

# Auto-detect if the dataset has a nested structure (e.g., data/PlantVillage/...)
possible_paths = [
    os.path.join(DATASET_DIR, 'PlantVillage'),
    os.path.join(DATASET_DIR, 'plantvillage'),
    os.path.join(DATASET_DIR, 'color'),        # Some versions use this
    os.path.join(DATASET_DIR, 'plantvillage dataset', 'color'),
    DATASET_DIR
]

data_root = None
for path in possible_paths:
    if os.path.exists(path):
        # Check if this directory contains subdirectories (class folders)
        subdirs = [d for d in os.listdir(path)
                   if os.path.isdir(os.path.join(path, d))]
        if len(subdirs) >= 10:  # PlantVillage has 38 classes
            data_root = path
            break

if data_root is None:
    raise FileNotFoundError(
        f"Could not find the PlantVillage dataset. "
        f"Please download it from Kaggle and place it in the '{DATASET_DIR}' directory.\\n"
        f"Expected structure: data/<class_name>/<image_files>"
    )

print(f"Dataset found at: {os.path.abspath(data_root)}")"""),

        # =====================================================================
        # DATASET STATISTICS
        # =====================================================================
        nbf.v4.new_code_cell("""# =============================================================================
# Dataset Statistics
# =============================================================================

# Collect information about all images
class_names = sorted([d for d in os.listdir(data_root)
                       if os.path.isdir(os.path.join(data_root, d))])

# Build a DataFrame with image metadata
image_data = []
for class_name in class_names:
    class_dir = os.path.join(data_root, class_name)
    image_files = [f for f in os.listdir(class_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    for img_file in image_files:
        img_path = os.path.join(class_dir, img_file)
        # Extract crop and disease from class name
        parts = class_name.split('___')
        crop = parts[0].replace('_', ' ')
        disease = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
        image_data.append({
            'filename': img_file,
            'filepath': img_path,
            'class': class_name,
            'crop': crop,
            'disease': disease
        })

df = pd.DataFrame(image_data)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total number of images:    {len(df):,}")
print(f"Number of classes:         {len(class_names)}")
print(f"Number of crop species:    {df['crop'].nunique()}")
print(f"Number of disease types:   {df['disease'].nunique()}")
print(f"Image file formats:        {df['filename'].apply(lambda x: x.split('.')[-1].upper()).unique()}")
print()

# Class distribution statistics
class_counts = df['class'].value_counts()
print(f"Images per class (min):    {class_counts.min():,}")
print(f"Images per class (max):    {class_counts.max():,}")
print(f"Images per class (mean):   {class_counts.mean():,.0f}")
print(f"Images per class (median): {class_counts.median():,.0f}")
print(f"Images per class (std):    {class_counts.std():,.0f}")
print()

# List all classes
print("\\nAll 38 Classes:")
print("-" * 60)
for i, (cls, count) in enumerate(class_counts.items(), 1):
    print(f"  {i:2d}. {cls:<45s} ({count:,} images)")"""),

        # =====================================================================
        # IMAGE PROPERTIES
        # =====================================================================
        nbf.v4.new_code_cell("""# =============================================================================
# Image Properties Analysis
# =============================================================================

# Sample images to check dimensions and properties
sample_size = min(500, len(df))
sample_indices = np.random.choice(len(df), sample_size, replace=False)

widths, heights, channels = [], [], []
file_sizes = []

print(f"Analysing {sample_size} sample images for dimension consistency...")
for idx in sample_indices:
    row = df.iloc[idx]
    try:
        img = load_img(row['filepath'])
        img_array = img_to_array(img)
        widths.append(img_array.shape[1])
        heights.append(img_array.shape[0])
        channels.append(img_array.shape[2])
        file_sizes.append(os.path.getsize(row['filepath']) / 1024)  # KB
    except Exception as e:
        print(f"  Warning: Could not load {row['filepath']}: {e}")

print(f"\\n{'='*60}")
print("IMAGE PROPERTIES")
print(f"{'='*60}")
print(f"Width  — Min: {min(widths)}, Max: {max(widths)}, Mode: {Counter(widths).most_common(1)[0][0]}")
print(f"Height — Min: {min(heights)}, Max: {max(heights)}, Mode: {Counter(heights).most_common(1)[0][0]}")
print(f"Channels: {set(channels)} ({'RGB' if 3 in channels else 'Unknown'})")
print(f"File size — Min: {min(file_sizes):.1f} KB, Max: {max(file_sizes):.1f} KB, Mean: {np.mean(file_sizes):.1f} KB")"""),

        # =====================================================================
        # CROPS AND DISEASES TABLE
        # =====================================================================
        nbf.v4.new_code_cell("""# =============================================================================
# Crop Species and Their Diseases
# =============================================================================

crop_disease_summary = df.groupby('crop').agg(
    num_diseases=('disease', 'nunique'),
    total_images=('filename', 'count'),
    diseases=('disease', lambda x: ', '.join(sorted(x.unique())))
).sort_values('total_images', ascending=False)

print("\\nCrop Species Summary:")
print("=" * 80)
for crop, row in crop_disease_summary.iterrows():
    print(f"\\n{crop} ({row['total_images']:,} images, {row['num_diseases']} categories):")
    for disease in sorted(row['diseases'].split(', ')):
        count = len(df[(df['crop'] == crop) & (df['disease'] == disease)])
        is_healthy = 'healthy' in disease.lower()
        marker = '✓' if is_healthy else '✗'
        print(f"    {marker} {disease} ({count:,} images)")"""),

        # =====================================================================
        # 1.2 VISUALISATION
        # =====================================================================
        nbf.v4.new_markdown_cell("""---
## 1.2 Visualisation

This section presents a comprehensive set of visualisations to explore and understand
the PlantVillage dataset. Each visualisation is described, justified, and interpreted.

### Visualisation Techniques Used:

| # | Visualisation | Type | Purpose |
|---|---|---|---|
| 1 | Class Distribution Bar Chart | Bar chart | Identify class imbalance |
| 2 | Crop Distribution Pie Chart | Pie chart | Proportion of images per crop species |
| 3 | Sample Image Grid | Image grid | Visual inspection of data quality |
| 4 | Healthy vs. Diseased Distribution | Grouped bar chart | Compare healthy/diseased counts per crop |
| 5 | Pixel Intensity Histograms | Histogram (per RGB channel) | Understand colour distribution |
| 6 | Class-wise Mean Images | Image grid | Reveal average visual patterns per class |
| 7 | Image Dimension Scatter Plot | Scatter plot | Verify dimensional consistency |"""),

        # =====================================================================
        # VIS 1: CLASS DISTRIBUTION
        # =====================================================================
        nbf.v4.new_markdown_cell("""### Visualisation 1: Class Distribution Bar Chart

**Type:** Horizontal bar chart

**Justification:** A bar chart is the most effective way to display the count of images
across all 38 classes. It immediately reveals any class imbalance, which is critical for
choosing appropriate loss functions, data augmentation strategies, and evaluation metrics.
Class imbalance can cause models to be biased toward majority classes."""),

        nbf.v4.new_code_cell("""# =============================================================================
# Visualisation 1: Class Distribution Bar Chart
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 12))

# Sort classes by count
class_counts_sorted = df['class'].value_counts().sort_values(ascending=True)

# Colour bars by crop type
colours = []
crop_colour_map = {}
cmap = plt.cm.get_cmap('tab20', df['crop'].nunique())
for i, crop in enumerate(sorted(df['crop'].unique())):
    crop_colour_map[crop] = cmap(i)

for cls in class_counts_sorted.index:
    crop = cls.split('___')[0].replace('_', ' ')
    colours.append(crop_colour_map.get(crop, 'steelblue'))

bars = ax.barh(range(len(class_counts_sorted)), class_counts_sorted.values, color=colours)

# Labels
ax.set_yticks(range(len(class_counts_sorted)))
ax.set_yticklabels([c.replace('___', ' — ').replace('_', ' ')
                     for c in class_counts_sorted.index], fontsize=9)
ax.set_xlabel('Number of Images')
ax.set_title('Distribution of Images Across All 38 Classes', fontsize=16, fontweight='bold')

# Add count labels on bars
for bar, count in zip(bars, class_counts_sorted.values):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            f'{count:,}', ha='left', va='center', fontsize=8)

# Add mean line
mean_count = class_counts_sorted.mean()
ax.axvline(x=mean_count, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_count:.0f}')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('../models/class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\\nInterpretation:")
print(f"  - The dataset contains {len(class_counts_sorted)} classes.")
print(f"  - Class sizes range from {class_counts_sorted.min():,} to {class_counts_sorted.max():,} images.")
print(f"  - Mean images per class: {mean_count:.0f}")
print(f"  - Imbalance ratio (max/min): {class_counts_sorted.max()/class_counts_sorted.min():.1f}x")
print(f"  - This moderate imbalance suggests data augmentation and")
print(f"    class-weighted loss may be beneficial during model training.")"""),

        # =====================================================================
        # VIS 2: CROP DISTRIBUTION
        # =====================================================================
        nbf.v4.new_markdown_cell("""### Visualisation 2: Crop Species Distribution

**Type:** Pie chart with donut style

**Justification:** A pie chart effectively shows the proportional representation of each
crop species in the dataset. This helps identify if certain crops dominate the dataset,
which could bias the model toward those species."""),

        nbf.v4.new_code_cell("""# =============================================================================
# Visualisation 2: Crop Distribution Pie Chart
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Pie chart — crop distribution
crop_counts = df['crop'].value_counts()
colours_pie = plt.cm.Set3(np.linspace(0, 1, len(crop_counts)))

wedges, texts, autotexts = axes[0].pie(
    crop_counts.values,
    labels=crop_counts.index,
    autopct='%1.1f%%',
    colors=colours_pie,
    pctdistance=0.8,
    startangle=90,
    textprops={'fontsize': 9}
)
axes[0].set_title('Proportion of Images by Crop Species', fontsize=14, fontweight='bold')

# Bar chart — number of disease categories per crop
diseases_per_crop = df.groupby('crop')['disease'].nunique().sort_values(ascending=True)
axes[1].barh(diseases_per_crop.index, diseases_per_crop.values, color='coral', edgecolor='white')
axes[1].set_xlabel('Number of Categories (including Healthy)')
axes[1].set_title('Number of Categories per Crop Species', fontsize=14, fontweight='bold')
for i, v in enumerate(diseases_per_crop.values):
    axes[1].text(v + 0.1, i, str(v), va='center', fontsize=10)

plt.tight_layout()
plt.savefig('../models/crop_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\\nInterpretation:")
print(f"  - Tomato has the most representation with {crop_counts.iloc[0]:,} images,")
print(f"    covering the most disease categories.")
print(f"  - Some crops (e.g., Blueberry, Raspberry) have fewer categories,")
print(f"    reflecting fewer documented diseases for those species.")"""),

        # =====================================================================
        # VIS 3: SAMPLE IMAGE GRID
        # =====================================================================
        nbf.v4.new_markdown_cell("""### Visualisation 3: Sample Image Grid

**Type:** Image grid (representative samples from each class)

**Justification:** Displaying sample images from each class allows visual inspection of
the data quality, image consistency, and the visual characteristics of each disease.
This is essential for understanding what features the model needs to learn and for
identifying any data quality issues (e.g., mislabelled images, poor quality)."""),

        nbf.v4.new_code_cell("""# =============================================================================
# Visualisation 3: Sample Image Grid (one image per class)
# =============================================================================

n_classes = len(class_names)
cols = 6
rows = (n_classes + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 2.8))
axes = axes.flatten()

for i, class_name in enumerate(sorted(class_names)):
    class_dir = os.path.join(data_root, class_name)
    img_files = os.listdir(class_dir)
    # Pick a random image from this class
    sample_img = os.path.join(class_dir, random.choice(img_files))
    img = mpimg.imread(sample_img)

    axes[i].imshow(img)
    # Format label: Crop — Disease
    label = class_name.replace('___', '\\n').replace('_', ' ')
    axes[i].set_title(label, fontsize=7, fontweight='bold')
    axes[i].axis('off')

# Hide empty subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle('Sample Images from All 38 Classes', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('../models/sample_images_grid.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nInterpretation:")
print("  - Images are captured under controlled conditions with uniform backgrounds.")
print("  - Disease symptoms vary widely: spots, discoloration, wilting, lesions.")
print("  - Healthy leaves appear uniformly green, providing a clear baseline.")
print("  - Some diseases (e.g., late blight vs. early blight) have subtle visual differences,")
print("    which will challenge the classification model.")"""),

        # =====================================================================
        # VIS 4: HEALTHY VS DISEASED
        # =====================================================================
        nbf.v4.new_markdown_cell("""### Visualisation 4: Healthy vs. Diseased Distribution

**Type:** Grouped/stacked bar chart

**Justification:** Comparing the number of healthy vs. diseased images per crop species
reveals whether the dataset provides adequate representation of both conditions for each
crop. This is important for ensuring the model can distinguish healthy from diseased leaves."""),

        nbf.v4.new_code_cell("""# =============================================================================
# Visualisation 4: Healthy vs. Diseased Distribution per Crop
# =============================================================================

df['health_status'] = df['disease'].apply(
    lambda x: 'Healthy' if 'healthy' in x.lower() else 'Diseased'
)

health_crop = df.groupby(['crop', 'health_status']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(14, 7))

health_crop.plot(kind='bar', stacked=False, ax=ax,
                  color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.7)

ax.set_title('Healthy vs. Diseased Images per Crop Species', fontsize=16, fontweight='bold')
ax.set_xlabel('Crop Species', fontsize=12)
ax.set_ylabel('Number of Images', fontsize=12)
ax.legend(['Healthy', 'Diseased'], fontsize=11, loc='upper right')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('../models/healthy_vs_diseased.png', dpi=150, bbox_inches='tight')
plt.show()

# Summary statistics
total_healthy = len(df[df['health_status'] == 'Healthy'])
total_diseased = len(df[df['health_status'] == 'Diseased'])
print(f"\\nInterpretation:")
print(f"  - Total healthy images: {total_healthy:,} ({total_healthy/len(df)*100:.1f}%)")
print(f"  - Total diseased images: {total_diseased:,} ({total_diseased/len(df)*100:.1f}%)")
print(f"  - The dataset is predominantly diseased images, which is expected")
print(f"    since the goal is disease detection.")
print(f"  - Not all crops have 'healthy' reference images in the dataset.")"""),

        # =====================================================================
        # VIS 5: PIXEL INTENSITY
        # =====================================================================
        nbf.v4.new_markdown_cell("""### Visualisation 5: Pixel Intensity Distribution (RGB Channels)

**Type:** Overlaid histograms (one per RGB channel)

**Justification:** Analysing the pixel intensity distribution across the Red, Green, and
Blue channels reveals the colour characteristics of the dataset. This informs preprocessing
decisions — for example, whether normalisation is needed and whether the dataset has
consistent lighting conditions. For plant images, we expect the Green channel to be
dominant in healthy leaves."""),

        nbf.v4.new_code_cell("""# =============================================================================
# Visualisation 5: Pixel Intensity Distribution (RGB)
# =============================================================================

# Sample images for pixel analysis
n_samples = 200
sample_df = df.sample(n=min(n_samples, len(df)), random_state=SEED)

all_red, all_green, all_blue = [], [], []

print(f"Analysing pixel intensities across {len(sample_df)} sample images...")
for _, row in sample_df.iterrows():
    try:
        img = load_img(row['filepath'], target_size=(128, 128))
        img_array = img_to_array(img).astype(np.uint8)
        all_red.extend(img_array[:, :, 0].flatten())
        all_green.extend(img_array[:, :, 1].flatten())
        all_blue.extend(img_array[:, :, 2].flatten())
    except:
        pass

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Individual channel histograms
for ax, data, colour, name in zip(axes,
                                    [all_red, all_green, all_blue],
                                    ['red', 'green', 'blue'],
                                    ['Red', 'Green', 'Blue']):
    ax.hist(data, bins=50, color=colour, alpha=0.7, density=True, edgecolor='white')
    ax.set_title(f'{name} Channel Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Pixel Intensity (0-255)')
    ax.set_ylabel('Density')
    ax.axvline(np.mean(data), color='black', linestyle='--',
               label=f'Mean: {np.mean(data):.1f}')
    ax.legend()

plt.suptitle('Pixel Intensity Distribution Across RGB Channels',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../models/pixel_intensity.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\\nInterpretation:")
print(f"  - Red channel mean:   {np.mean(all_red):.1f}")
print(f"  - Green channel mean: {np.mean(all_green):.1f}")
print(f"  - Blue channel mean:  {np.mean(all_blue):.1f}")
print(f"  - The Green channel is expected to have higher intensity for healthy leaves,")
print(f"    while diseased leaves may show elevated Red/Brown tones.")
print(f"  - Pixel values span the full 0-255 range, confirming that normalisation")
print(f"    (scaling to [0,1] or [-1,1]) is necessary before model training.")"""),

        # =====================================================================
        # VIS 6: CLASS-WISE MEAN IMAGES
        # =====================================================================
        nbf.v4.new_markdown_cell("""### Visualisation 6: Class-wise Mean Images

**Type:** Average image grid

**Justification:** Computing the mean image for each class reveals the dominant visual
patterns and colour characteristics of each disease type. If the mean images for different
classes look visually distinct, it suggests the classification task is feasible. Overlapping
mean images indicate potentially challenging class boundaries."""),

        nbf.v4.new_code_cell("""# =============================================================================
# Visualisation 6: Class-wise Mean Images (selected classes)
# =============================================================================

# Select a subset of classes for clarity (6 diverse examples)
selected_classes = class_names[::max(1, len(class_names)//8)][:8]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

IMG_SIZE = 128  # Smaller size for faster computation

for i, class_name in enumerate(selected_classes):
    class_dir = os.path.join(data_root, class_name)
    img_files = os.listdir(class_dir)

    # Load up to 50 images and compute mean
    sample_files = random.sample(img_files, min(50, len(img_files)))
    imgs = []
    for f in sample_files:
        try:
            img = load_img(os.path.join(class_dir, f), target_size=(IMG_SIZE, IMG_SIZE))
            imgs.append(img_to_array(img))
        except:
            pass

    if imgs:
        mean_img = np.mean(imgs, axis=0).astype(np.uint8)
        axes[i].imshow(mean_img)

    label = class_name.replace('___', '\\n').replace('_', ' ')
    axes[i].set_title(label, fontsize=8, fontweight='bold')
    axes[i].axis('off')

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle('Mean Images for Selected Classes', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('../models/mean_images.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nInterpretation:")
print("  - Mean images reveal the dominant colour and texture patterns for each class.")
print("  - Healthy classes typically show uniform green, while diseased classes show")
print("    distinctive discoloration patterns.")
print("  - Visually distinct mean images suggest the classes are separable,")
print("    supporting the feasibility of automated classification.")"""),

        # =====================================================================
        # VIS 7: DIMENSION SCATTER
        # =====================================================================
        nbf.v4.new_markdown_cell("""### Visualisation 7: Image Dimension Consistency

**Type:** Scatter plot / histogram of image dimensions

**Justification:** Verifying that all images have consistent dimensions is critical for
determining whether resizing is needed during preprocessing. Inconsistent dimensions
would require resizing, which could distort features if not handled properly."""),

        nbf.v4.new_code_cell("""# =============================================================================
# Visualisation 7: Image Dimension Distribution
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Width distribution
axes[0].hist(widths, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_title('Image Width Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Width (pixels)')
axes[0].set_ylabel('Count')
axes[0].axvline(np.mean(widths), color='red', linestyle='--',
                label=f'Mean: {np.mean(widths):.0f}px')
axes[0].legend()

# Height distribution
axes[1].hist(heights, bins=30, color='coral', edgecolor='white', alpha=0.8)
axes[1].set_title('Image Height Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Height (pixels)')
axes[1].set_ylabel('Count')
axes[1].axvline(np.mean(heights), color='red', linestyle='--',
                label=f'Mean: {np.mean(heights):.0f}px')
axes[1].legend()

plt.suptitle('Image Dimension Consistency Check', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../models/image_dimensions.png', dpi=150, bbox_inches='tight')
plt.show()

# Check uniformity
unique_dims = set(zip(widths, heights))
print(f"\\nInterpretation:")
print(f"  - Unique dimension combinations found: {len(unique_dims)}")
if len(unique_dims) == 1:
    w, h = list(unique_dims)[0]
    print(f"  - All sampled images are uniformly {int(w)}×{int(h)} pixels.")
    print(f"  - Resizing to 224×224 will still be needed for EfficientNet input.")
else:
    print(f"  - Dimensions: {unique_dims}")
    print(f"  - Variable dimensions detected — resizing during preprocessing is essential.")"""),

        # =====================================================================
        # SUMMARY
        # =====================================================================
        nbf.v4.new_markdown_cell("""---
## Section 1 Summary

### Key Findings:

1. **Dataset Size:** The PlantVillage dataset contains approximately 54,303 images across
   38 classes, covering 14 crop species — well above the minimum 500-image threshold
   recommended for this project.

2. **Class Balance:** There is moderate class imbalance, with the largest class containing
   significantly more images than the smallest. This will be addressed through data
   augmentation and class-weighted loss functions during model training.

3. **Image Quality:** Images are captured under controlled conditions with uniform
   backgrounds, consistent dimensions (256×256 RGB), and good resolution. This reduces
   the need for extensive image cleaning.

4. **Visual Separability:** Mean images and sample visualisations show visually distinct
   patterns across most classes, suggesting that CNNs should be able to learn discriminative
   features effectively.

5. **Preprocessing Requirements:**
   - Resizing to 224×224 (EfficientNet input size)
   - Normalisation (pixel scaling)
   - Data augmentation (rotation, flipping, zoom) to prevent overfitting
   - Stratified splitting to maintain class proportions

These findings inform the model selection and data transformation decisions in Section 2."""),
    ]

    nb['cells'] = cells

    # Set kernel info
    nb['metadata'] = {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.9.0'
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"✓ Created notebook: {output_path}")


if __name__ == "__main__":
    create_data_analysis_notebook()
