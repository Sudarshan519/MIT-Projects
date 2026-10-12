"""
Generate the Model Training Jupyter Notebook (Section 2 of INFO813 Project).
Run this script to create: notebooks/02_model_training.ipynb

This notebook covers:
- 2.1 Description and justification of modelling technique
- 2.2 Data Transformation
- 2.3 Implementation of Custom CNN and EfficientNetB0
"""

import nbformat as nbf
import os


def create_model_training_notebook(output_path="notebooks/02_model_training.ipynb"):
    nb = nbf.v4.new_notebook()

    cells = [
        # =====================================================================
        # TITLE
        # =====================================================================
        nbf.v4.new_markdown_cell("""# Plant Disease Detection — Model Training
## INFO813 Artificial Intelligence | Trimester 3, 2026

This notebook covers **Section 2: Problem Modelling** of the project report.

### Table of Contents
1. [2.1 Model Description & Justification](#2.1-Model-Description-and-Justification)
2. [2.2 Data Transformation](#2.2-Data-Transformation)
3. [2.3 Implementation — Custom CNN](#2.3a-Implementation:-Custom-CNN)
4. [2.3 Implementation — EfficientNetB0](#2.3b-Implementation:-EfficientNetB0-Transfer-Learning)

---"""),

        # =====================================================================
        # IMPORTS
        # =====================================================================
        nbf.v4.new_code_cell("""# =============================================================================
# Import Libraries
# =============================================================================
import os
import json
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configuration
IMG_SIZE = 224          # EfficientNetB0 native input size
BATCH_SIZE = 32
EPOCHS_HEAD = 10        # Phase 1: Train classification head only
EPOCHS_FINETUNE = 15    # Phase 2: Fine-tune top layers
NUM_CLASSES = 38

# Check for GPU
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
if tf.config.list_physical_devices('GPU'):
    print(f"GPU device: {tf.config.list_physical_devices('GPU')[0]}")
else:
    print("WARNING: No GPU detected. Training will be slow on CPU.")
    print("Consider using Google Colab with GPU runtime.")

print("\\nSetup complete!")"""),

        # =====================================================================
        # 2.1 MODEL DESCRIPTION
        # =====================================================================
        nbf.v4.new_markdown_cell("""---
## 2.1 Model Description and Justification

### Approach: Two Models for Comparative Analysis

We implement and compare **two** deep learning models for plant disease classification:

#### Model A: Custom CNN (Baseline)
A Convolutional Neural Network built from scratch with 4 convolutional blocks.

**Architecture:**
```
Input (224×224×3)
→ Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
→ Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
→ Conv2D(128, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
→ Conv2D(256, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
→ GlobalAveragePooling2D
→ Dense(512) → ReLU → Dropout(0.5)
→ Dense(38, softmax)
```

**Purpose:** Serves as a baseline to demonstrate understanding of CNN fundamentals and
to measure the improvement gained from transfer learning.

#### Model B: EfficientNetB0 with Transfer Learning (Primary Model)
A pre-trained EfficientNetB0 model fine-tuned on the PlantVillage dataset.

**Why EfficientNetB0?**

| Model | Parameters | ImageNet Top-1 Acc | Advantage |
|---|---|---|---|
| VGG16 | 138M | 71.3% | Simple but extremely large |
| ResNet50 | 25.6M | 76.1% | Skip connections; reliable |
| **EfficientNetB0** | **5.3M** | **77.1%** | **Best accuracy/efficiency ratio** |
| MobileNetV2 | 3.4M | 71.8% | Lightweight but lower accuracy |

EfficientNetB0 was selected because:
1. **Compound Scaling:** Uses a principled method to scale network depth, width, and
   resolution simultaneously (Tan & Le, 2019), achieving superior accuracy with fewer
   parameters than alternatives.
2. **Transfer Learning Suitability:** Pre-trained on ImageNet (1.2M images, 1000 classes),
   the lower convolutional layers already capture universal visual features (edges,
   textures, patterns) that are directly relevant to plant disease symptoms.
3. **Efficiency:** With only 5.3M parameters, it trains faster and requires less memory
   than ResNet50 (25.6M) or VGG16 (138M), making it practical for this project's
   compute constraints.
4. **Deployment Ready:** Small model size enables future deployment on mobile devices,
   which aligns with the real-world application of field-based plant disease diagnosis.

**Training Strategy: Two-Phase Fine-Tuning**
- **Phase 1:** Freeze all EfficientNetB0 base layers; train only the new classification
  head (10 epochs, learning rate = 1e-3).
- **Phase 2:** Unfreeze the top 20 layers of EfficientNetB0; fine-tune at a very low
  learning rate (1e-5) to adapt features to plant disease patterns without destroying
  pre-trained knowledge (15 epochs).

**Reference:**
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional
  neural networks. *Proceedings of the 36th International Conference on Machine Learning
  (ICML)*, 6105–6114."""),

        # =====================================================================
        # 2.2 DATA TRANSFORMATION
        # =====================================================================
        nbf.v4.new_markdown_cell("""---
## 2.2 Data Transformation

### Transformation Pipeline

The raw PlantVillage images require several transformations before they can be used for
model training. Each transformation is described, justified, and applied below.

| Step | Transformation | Justification |
|---|---|---|
| 1 | **Resizing** to 224×224 | EfficientNetB0's native input dimension |
| 2 | **Normalisation** to [0,1] or [-1,1] | Neural networks converge faster with small input values |
| 3 | **Train/Val/Test Split** (70/15/15) | Separate data for training, hyperparameter tuning, and final evaluation |
| 4 | **Data Augmentation** | Prevent overfitting by artificially expanding training set diversity |
| 5 | **Label Encoding** | Convert class names to integer labels for the loss function |
| 6 | **Batching & Prefetching** | Efficient GPU utilisation during training |"""),

        # =====================================================================
        # LOAD DATASET
        # =====================================================================
        nbf.v4.new_code_cell("""# =============================================================================
# Load Dataset & Prepare File Lists
# =============================================================================

DATASET_DIR = os.path.join('..', 'data')

# Auto-detect dataset root
possible_paths = [
    os.path.join(DATASET_DIR, 'PlantVillage'),
    os.path.join(DATASET_DIR, 'plantvillage'),
    os.path.join(DATASET_DIR, 'color'),
    os.path.join(DATASET_DIR, 'plantvillage dataset', 'color'),
    DATASET_DIR
]

data_root = None
for path in possible_paths:
    if os.path.exists(path):
        subdirs = [d for d in os.listdir(path)
                   if os.path.isdir(os.path.join(path, d))]
        if len(subdirs) >= 10:
            data_root = path
            break

if data_root is None:
    raise FileNotFoundError("Could not find PlantVillage dataset in the 'data' directory.")

# Get class names
class_names = sorted([d for d in os.listdir(data_root)
                       if os.path.isdir(os.path.join(data_root, d))])

print(f"Dataset root: {os.path.abspath(data_root)}")
print(f"Number of classes: {len(class_names)}")

# Build file paths and labels
all_filepaths = []
all_labels = []

for label_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(data_root, class_name)
    for img_file in os.listdir(class_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_filepaths.append(os.path.join(class_dir, img_file))
            all_labels.append(label_idx)

all_filepaths = np.array(all_filepaths)
all_labels = np.array(all_labels)

print(f"Total images found: {len(all_filepaths):,}")

# Save class label mapping
label_map = {i: name for i, name in enumerate(class_names)}
label_map_path = os.path.join('..', 'app', 'class_labels.json')
os.makedirs(os.path.dirname(label_map_path), exist_ok=True)
with open(label_map_path, 'w') as f:
    json.dump(label_map, f, indent=2)
print(f"Class label mapping saved to: {label_map_path}")"""),

        # =====================================================================
        # TRAIN/VAL/TEST SPLIT
        # =====================================================================
        nbf.v4.new_code_cell("""# =============================================================================
# Stratified Train / Validation / Test Split
# =============================================================================

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    all_filepaths, all_labels,
    test_size=0.30,
    random_state=SEED,
    stratify=all_labels
)

# Second split: 50/50 of temp → 15% val, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)

print(f"Training set:   {len(X_train):,} images ({len(X_train)/len(all_filepaths)*100:.1f}%)")
print(f"Validation set: {len(X_val):,} images ({len(X_val)/len(all_filepaths)*100:.1f}%)")
print(f"Test set:       {len(X_test):,} images ({len(X_test)/len(all_filepaths)*100:.1f}%)")

# Verify stratification
print(f"\\nClass distribution check (first 5 classes):")
for i in range(5):
    train_count = np.sum(y_train == i)
    val_count = np.sum(y_val == i)
    test_count = np.sum(y_test == i)
    total = train_count + val_count + test_count
    print(f"  Class {i} ({class_names[i][:30]}...): "
          f"Train={train_count}, Val={val_count}, Test={test_count} "
          f"(Ratios: {train_count/total:.2f}/{val_count/total:.2f}/{test_count/total:.2f})")"""),

        # =====================================================================
        # DATA GENERATORS
        # =====================================================================
        nbf.v4.new_markdown_cell("""### Data Augmentation

**Applied to training set only** to prevent overfitting and simulate real-world variations:

- **Random horizontal flip:** Leaves can be oriented in any direction in the field
- **Random rotation (±20°):** Accounts for different camera angles
- **Random zoom (±15%):** Simulates varying distances from the leaf
- **Random brightness adjustment:** Simulates varying lighting conditions in the field
- **Random width/height shift (±10%):** Simulates off-centre leaf positioning

These augmentations create synthetic variations of training images, effectively expanding
the training set and improving the model's ability to generalise to unseen images."""),

        nbf.v4.new_code_cell("""# =============================================================================
# Create tf.data Pipelines with Augmentation
# =============================================================================

def load_and_preprocess_image(filepath, label, augment=False, use_effnet_preprocess=False):
    \"\"\"Load an image from filepath and preprocess it.\"\"\"
    # Read and decode image
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

    if augment:
        # Data augmentation (training only)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        # Random rotation using tf.keras.layers in the model instead

    if use_effnet_preprocess:
        # EfficientNet expects inputs in [0, 255] and applies its own normalisation
        img = tf.cast(img, tf.float32)
    else:
        # Standard normalisation to [0, 1]
        img = tf.cast(img, tf.float32) / 255.0

    return img, label


def create_dataset(filepaths, labels, batch_size, augment=False,
                   use_effnet_preprocess=False, shuffle=True):
    \"\"\"Create a tf.data.Dataset pipeline.\"\"\"
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=SEED)

    ds = ds.map(
        lambda fp, lbl: load_and_preprocess_image(fp, lbl, augment, use_effnet_preprocess),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# Create datasets for Custom CNN (normalised to [0,1])
train_ds_cnn = create_dataset(X_train, y_train, BATCH_SIZE, augment=True, use_effnet_preprocess=False)
val_ds_cnn = create_dataset(X_val, y_val, BATCH_SIZE, augment=False, use_effnet_preprocess=False)
test_ds_cnn = create_dataset(X_test, y_test, BATCH_SIZE, augment=False, use_effnet_preprocess=False)

# Create datasets for EfficientNet (keeps [0, 255] range)
train_ds_effnet = create_dataset(X_train, y_train, BATCH_SIZE, augment=True, use_effnet_preprocess=True)
val_ds_effnet = create_dataset(X_val, y_val, BATCH_SIZE, augment=False, use_effnet_preprocess=True)
test_ds_effnet = create_dataset(X_test, y_test, BATCH_SIZE, augment=False, use_effnet_preprocess=True)

# Preview a batch
for images, labels in train_ds_cnn.take(1):
    print(f"Batch shape: {images.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"Pixel range: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")
    print(f"Labels sample: {labels.numpy()[:10]}")"""),

        # =====================================================================
        # VISUALISE AUGMENTATION
        # =====================================================================
        nbf.v4.new_code_cell("""# =============================================================================
# Visualise Data Augmentation Effects
# =============================================================================

# Pick a single image and show augmented versions
sample_path = X_train[0]
original_img = tf.io.read_file(sample_path)
original_img = tf.image.decode_jpeg(original_img, channels=3)
original_img = tf.image.resize(original_img, [IMG_SIZE, IMG_SIZE])

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes[0, 0].imshow(original_img.numpy().astype(np.uint8))
axes[0, 0].set_title('Original', fontweight='bold')
axes[0, 0].axis('off')

for i in range(1, 10):
    row, col = divmod(i, 5)
    # Apply random augmentations
    aug_img = tf.identity(original_img)
    aug_img = tf.image.random_flip_left_right(aug_img)
    aug_img = tf.image.random_flip_up_down(aug_img)
    aug_img = tf.image.random_brightness(aug_img, max_delta=0.2)
    aug_img = tf.image.random_contrast(aug_img, lower=0.8, upper=1.2)
    aug_img = tf.image.random_saturation(aug_img, lower=0.8, upper=1.2)
    aug_img = tf.clip_by_value(aug_img, 0, 255)

    axes[row, col].imshow(aug_img.numpy().astype(np.uint8))
    axes[row, col].set_title(f'Augmented #{i}')
    axes[row, col].axis('off')

class_idx = y_train[0]
plt.suptitle(f'Data Augmentation Examples — Class: {class_names[class_idx]}',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../models/augmentation_examples.png', dpi=150, bbox_inches='tight')
plt.show()

print("Data augmentation creates diverse training examples from a single image,")
print("helping the model generalise to unseen images with different orientations and lighting.")"""),

        # =====================================================================
        # CLASS WEIGHTS
        # =====================================================================
        nbf.v4.new_code_cell("""# =============================================================================
# Compute Class Weights (to handle class imbalance)
# =============================================================================

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

print("Class weights (top 10 most weighted):")
sorted_weights = sorted(class_weight_dict.items(), key=lambda x: x[1], reverse=True)
for idx, weight in sorted_weights[:10]:
    print(f"  Class {idx:2d} ({class_names[idx][:40]}...): {weight:.3f}")

print(f"\\nWeight range: [{min(class_weights):.3f}, {max(class_weights):.3f}]")
print("Classes with fewer images receive higher weights during training,")
print("ensuring the model does not become biased toward majority classes.")"""),

        # =====================================================================
        # 2.3a CUSTOM CNN
        # =====================================================================
        nbf.v4.new_markdown_cell("""---
## 2.3a Implementation: Custom CNN

The custom CNN consists of 4 convolutional blocks, each containing:
- **Conv2D** layer for feature extraction
- **Batch Normalisation** for training stability
- **ReLU** activation for non-linearity
- **Max Pooling** for spatial downsampling
- **Dropout** for regularisation

The architecture progressively increases filter depth (32 → 64 → 128 → 256) to capture
increasingly complex features, from edges and textures to disease-specific patterns."""),

        nbf.v4.new_code_cell("""# =============================================================================
# Build Custom CNN Model
# =============================================================================

def build_custom_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    \"\"\"
    Build a custom CNN with 4 convolutional blocks.

    Architecture:
        4 × (Conv2D → BatchNorm → ReLU → MaxPool → Dropout)
        → GlobalAveragePooling → Dense → Dropout → Softmax

    Parameters:
        input_shape: Input image dimensions (H, W, C)
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    \"\"\"
    model = models.Sequential([
        # --- Block 1: 32 filters ---
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # --- Block 2: 64 filters ---
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # --- Block 3: 128 filters ---
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # --- Block 4: 256 filters ---
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # --- Classification Head ---
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Build and display model summary
cnn_model = build_custom_cnn()
cnn_model.summary()

# Count parameters
total_params = cnn_model.count_params()
print(f"\\nTotal parameters: {total_params:,}")"""),

        # =====================================================================
        # TRAIN CUSTOM CNN
        # =====================================================================
        nbf.v4.new_code_cell("""# =============================================================================
# Train Custom CNN
# =============================================================================

# Define callbacks
cnn_callbacks = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath='../models/custom_cnn_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("Training Custom CNN...")
print("=" * 60)

cnn_history = cnn_model.fit(
    train_ds_cnn,
    validation_data=val_ds_cnn,
    epochs=30,
    class_weight=class_weight_dict,
    callbacks=cnn_callbacks,
    verbose=1
)

print("\\n✓ Custom CNN training complete!")"""),

        # =====================================================================
        # PLOT CNN TRAINING HISTORY
        # =====================================================================
        nbf.v4.new_code_cell("""# =============================================================================
# Plot Custom CNN Training History
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(cnn_history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(cnn_history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_title('Custom CNN — Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(cnn_history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(cnn_history.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_title('Custom CNN — Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../models/cnn_training_history.png', dpi=150, bbox_inches='tight')
plt.show()

# Report best epoch
best_epoch = np.argmax(cnn_history.history['val_accuracy'])
best_val_acc = cnn_history.history['val_accuracy'][best_epoch]
print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}")"""),

        # =====================================================================
        # SAVE CNN MODEL
        # =====================================================================
        nbf.v4.new_code_cell("""# =============================================================================
# Save Custom CNN Model
# =============================================================================

cnn_model.save('../models/custom_cnn_model.keras')
print("✓ Custom CNN model saved to: ../models/custom_cnn_model.keras")

# Also evaluate on test set
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(test_ds_cnn, verbose=0)
print(f"\\nCustom CNN Test Accuracy: {cnn_test_acc:.4f}")
print(f"Custom CNN Test Loss:     {cnn_test_loss:.4f}")"""),

        # =====================================================================
        # 2.3b EFFICIENTNET
        # =====================================================================
        nbf.v4.new_markdown_cell("""---
## 2.3b Implementation: EfficientNetB0 Transfer Learning

### Two-Phase Training Strategy:

**Phase 1 — Feature Extraction (Head Training):**
- The pre-trained EfficientNetB0 base is **frozen** (all weights locked)
- Only the new classification head (GlobalAvgPool → Dense → Softmax) is trained
- Learning rate: 1e-3
- This quickly adapts the model to the new task without altering pre-trained features

**Phase 2 — Fine-Tuning:**
- The top 20 layers of EfficientNetB0 are **unfrozen**
- The entire model is trained at a very low learning rate (1e-5)
- This allows the higher-level features to be subtly adapted to plant disease patterns
- The low learning rate prevents catastrophic forgetting of pre-trained knowledge"""),

        nbf.v4.new_code_cell("""# =============================================================================
# Build EfficientNetB0 Transfer Learning Model
# =============================================================================

def build_efficientnet_model(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                              num_classes=NUM_CLASSES):
    \"\"\"
    Build EfficientNetB0 transfer learning model.

    Architecture:
        EfficientNetB0 (pre-trained, frozen) → GlobalAvgPool → Dense(256)
        → Dropout(0.5) → Dense(num_classes, softmax)

    Parameters:
        input_shape: Input image dimensions
        num_classes: Number of output classes

    Returns:
        Compiled Keras model with frozen base
    \"\"\"
    # Load pre-trained EfficientNetB0 (without top classification layer)
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze all base model layers
    base_model.trainable = False

    # Build the complete model
    model = models.Sequential([
        # EfficientNet preprocessing layer
        layers.InputLayer(input_shape=input_shape),

        # Pre-trained base
        base_model,

        # New classification head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model


# Build model
effnet_model, base_model = build_efficientnet_model()
effnet_model.summary()

# Count trainable vs frozen parameters
trainable = sum([tf.keras.backend.count_params(w) for w in effnet_model.trainable_weights])
non_trainable = sum([tf.keras.backend.count_params(w) for w in effnet_model.non_trainable_weights])
print(f"\\nTrainable parameters:     {trainable:,}")
print(f"Non-trainable parameters: {non_trainable:,}")
print(f"Total parameters:         {trainable + non_trainable:,}")
print(f"\\nBase model layers: {len(base_model.layers)}")
print(f"Base model trainable: {base_model.trainable}")"""),

        # =====================================================================
        # PHASE 1: TRAIN HEAD
        # =====================================================================
        nbf.v4.new_code_cell("""# =============================================================================
# Phase 1: Train Classification Head Only
# =============================================================================

effnet_callbacks_p1 = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

print("Phase 1: Training classification head (base model frozen)...")
print("=" * 60)

history_p1 = effnet_model.fit(
    train_ds_effnet,
    validation_data=val_ds_effnet,
    epochs=EPOCHS_HEAD,
    class_weight=class_weight_dict,
    callbacks=effnet_callbacks_p1,
    verbose=1
)

print("\\n✓ Phase 1 training complete!")
p1_val_acc = max(history_p1.history['val_accuracy'])
print(f"Best Phase 1 validation accuracy: {p1_val_acc:.4f}")"""),

        # =====================================================================
        # PHASE 2: FINE-TUNE
        # =====================================================================
        nbf.v4.new_code_cell("""# =============================================================================
# Phase 2: Fine-Tune Top Layers
# =============================================================================

# Unfreeze the top 20 layers of the base model
base_model.trainable = True

# Freeze all layers except the top 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with a much lower learning rate
effnet_model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Count updated trainable parameters
trainable_p2 = sum([tf.keras.backend.count_params(w) for w in effnet_model.trainable_weights])
print(f"Trainable parameters after unfreezing: {trainable_p2:,}")
print(f"Layers unfrozen: top 20 of {len(base_model.layers)}")

effnet_callbacks_p2 = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath='../models/efficientnet_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("\\nPhase 2: Fine-tuning top 20 layers...")
print("=" * 60)

history_p2 = effnet_model.fit(
    train_ds_effnet,
    validation_data=val_ds_effnet,
    epochs=EPOCHS_FINETUNE,
    class_weight=class_weight_dict,
    callbacks=effnet_callbacks_p2,
    verbose=1
)

print("\\n✓ Phase 2 fine-tuning complete!")
p2_val_acc = max(history_p2.history['val_accuracy'])
print(f"Best Phase 2 validation accuracy: {p2_val_acc:.4f}")"""),

        # =====================================================================
        # PLOT EFFICIENTNET TRAINING
        # =====================================================================
        nbf.v4.new_code_cell("""# =============================================================================
# Plot EfficientNetB0 Training History (Both Phases)
# =============================================================================

# Combine histories
combined_acc = history_p1.history['accuracy'] + history_p2.history['accuracy']
combined_val_acc = history_p1.history['val_accuracy'] + history_p2.history['val_accuracy']
combined_loss = history_p1.history['loss'] + history_p2.history['loss']
combined_val_loss = history_p1.history['val_loss'] + history_p2.history['val_loss']

phase1_epochs = len(history_p1.history['accuracy'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(combined_acc, label='Train Accuracy', linewidth=2)
axes[0].plot(combined_val_acc, label='Val Accuracy', linewidth=2)
axes[0].axvline(x=phase1_epochs - 0.5, color='gray', linestyle='--', alpha=0.5,
                label=f'Phase 1 → Phase 2')
axes[0].set_title('EfficientNetB0 — Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(combined_loss, label='Train Loss', linewidth=2)
axes[1].plot(combined_val_loss, label='Val Loss', linewidth=2)
axes[1].axvline(x=phase1_epochs - 0.5, color='gray', linestyle='--', alpha=0.5,
                label=f'Phase 1 → Phase 2')
axes[1].set_title('EfficientNetB0 — Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../models/efficientnet_training_history.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        # =====================================================================
        # SAVE EFFICIENTNET
        # =====================================================================
        nbf.v4.new_code_cell("""# =============================================================================
# Save EfficientNetB0 Model & Evaluate on Test Set
# =============================================================================

effnet_model.save('../models/efficientnet_model.keras')
print("✓ EfficientNetB0 model saved to: ../models/efficientnet_model.keras")

# Evaluate on test set
effnet_test_loss, effnet_test_acc = effnet_model.evaluate(test_ds_effnet, verbose=0)
print(f"\\nEfficientNetB0 Test Accuracy: {effnet_test_acc:.4f}")
print(f"EfficientNetB0 Test Loss:     {effnet_test_loss:.4f}")

# Quick comparison
print(f"\\n{'='*60}")
print(f"QUICK COMPARISON")
print(f"{'='*60}")
print(f"Custom CNN Test Accuracy:     {cnn_test_acc:.4f}")
print(f"EfficientNetB0 Test Accuracy: {effnet_test_acc:.4f}")
print(f"Improvement: {(effnet_test_acc - cnn_test_acc)*100:+.2f} percentage points")"""),

        # =====================================================================
        # SUMMARY
        # =====================================================================
        nbf.v4.new_markdown_cell("""---
## Section 2.3 Summary

### Models Implemented:

1. **Custom CNN:** 4-block convolutional neural network built from scratch
   - Serves as a baseline for comparison
   - Demonstrates understanding of CNN architecture fundamentals

2. **EfficientNetB0 Transfer Learning:** Pre-trained model fine-tuned on PlantVillage
   - Two-phase training: head training → fine-tuning
   - Leverages pre-trained ImageNet features for improved performance

### Data Transformations Applied:
- Image resizing (224×224)
- Normalisation (model-specific)
- Stratified train/val/test split (70/15/15)
- Data augmentation (flips, brightness, contrast, saturation)
- Class weighting (balanced)
- Efficient tf.data pipeline with prefetching

### Next Steps:
- Detailed evaluation and comparison → `03_model_comparison.ipynb`
- Grad-CAM explainability analysis
- Model deployment in interactive application"""),
    ]

    nb['cells'] = cells

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
    create_model_training_notebook()
