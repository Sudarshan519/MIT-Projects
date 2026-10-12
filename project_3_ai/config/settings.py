# ============================================================================
# INFO813 Project 3 — Configuration
# Centralised project settings and hyperparameters
# ============================================================================

import os

# ── Project Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
FIGURES_DATA_DIR = os.path.join(FIGURES_DIR, 'data_analysis')
FIGURES_EVAL_DIR = os.path.join(FIGURES_DIR, 'model_evaluation')
FIGURES_GRADCAM_DIR = os.path.join(FIGURES_DIR, 'gradcam')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')

# ── Dataset ────────────────────────────────────────────────────────────────
METADATA_CSV = os.path.join(RAW_DATA_DIR, 'HAM10000_metadata.csv')
IMAGE_DIRS = [
    os.path.join(RAW_DATA_DIR, 'HAM10000_images_part_1'),
    os.path.join(RAW_DATA_DIR, 'HAM10000_images_part_2'),
]

# ── Class Labels ───────────────────────────────────────────────────────────
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_FULL_NAMES = {
    'akiec': 'Actinic Keratoses / Bowen\'s Disease',
    'bcc':   'Basal Cell Carcinoma',
    'bkl':   'Benign Keratosis',
    'df':    'Dermatofibroma',
    'mel':   'Melanoma',
    'nv':    'Melanocytic Nevi',
    'vasc':  'Vascular Lesions',
}
CLASS_RISK_LEVELS = {
    'akiec': 'Pre-cancerous',
    'bcc':   'Malignant',
    'bkl':   'Benign',
    'df':    'Benign',
    'mel':   'Malignant',
    'nv':    'Benign',
    'vasc':  'Benign',
}
NUM_CLASSES = len(CLASS_NAMES)

# ── Image Preprocessing ───────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)     # EfficientNetB0 native input size
COLOUR_MODE = 'rgb'
PIXEL_RANGE = (0.0, 1.0)   # Normalised pixel range after rescaling

# ── Data Splitting ─────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

# ── Data Augmentation (Training Only) ──────────────────────────────────────
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'horizontal_flip': True,
    'vertical_flip': True,
    'zoom_range': 0.2,
    'brightness_range': (0.8, 1.2),
    'fill_mode': 'nearest',
}

# ── Model Hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS_INITIAL = 30        # Frozen base training
EPOCHS_FINETUNE = 20       # Fine-tuning phase
LEARNING_RATE_INITIAL = 1e-4
LEARNING_RATE_FINETUNE = 1e-5
DROPOUT_RATE = 0.3
DENSE_UNITS = 256
FINETUNE_LAYERS = 20       # Number of base layers to unfreeze

# ── Callbacks ──────────────────────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# ── Visualisation Style ───────────────────────────────────────────────────
PLOT_DPI = 300
PLOT_STYLE = 'seaborn-v0_8-whitegrid'
COLOUR_PALETTE = {
    'akiec': '#e74c3c',     # Red — pre-cancerous
    'bcc':   '#c0392b',     # Dark red — malignant
    'bkl':   '#2ecc71',     # Green — benign
    'df':    '#27ae60',     # Dark green — benign
    'mel':   '#8e44ad',     # Purple — malignant (most dangerous)
    'nv':    '#3498db',     # Blue — benign (majority)
    'vasc':  '#f39c12',     # Orange — benign
}
FONT_FAMILY = 'Helvetica, Arial, DejaVu Sans'


def ensure_dirs():
    """Create all project directories if they don't exist."""
    for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
              FIGURES_DIR, FIGURES_DATA_DIR, FIGURES_EVAL_DIR,
              FIGURES_GRADCAM_DIR, REPORT_DIR]:
        os.makedirs(d, exist_ok=True)
