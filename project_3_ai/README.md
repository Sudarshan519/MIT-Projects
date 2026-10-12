# Skin Lesion Classification System

## INFO813 Artificial Intelligence — Trimester 3, 2026

An AI-powered system for classifying dermoscopic images of skin lesions into 7 diagnostic categories using deep learning. The system uses transfer learning (EfficientNetB0 pre-trained on ImageNet) with Grad-CAM explainability to support clinical decision-making in dermatology.

---

## Project Structure

```
project_3_ai/
├── analysis/                        # Data exploration & visualisation scripts
│   ├── 01_data_analysis.py          # Section 1.1 + 1.2: EDA & visualisations
│   └── 02_data_transformation.py    # Section 2.2: Preprocessing pipeline
├── research/                        # Model development & training
│   ├── 03_model_training.py         # Section 2.3: CNN + EfficientNetB0 training
│   └── 04_model_evaluation.py       # Section 2.4: Evaluation & comparison
├── app/                             # Interactive application (Section 3)
│   ├── app.py                       # Gradio web interface
│   ├── predict.py                   # Prediction engine + Grad-CAM
│   └── class_labels.json            # Class metadata
├── config/                          # Centralised configuration
│   ├── __init__.py
│   └── settings.py                  # All paths, hyperparameters, class definitions
├── evaluation/                      # Evaluation outputs
├── models/                          # Trained model files (.keras)
├── data/
│   ├── raw/                         # Original HAM10000 dataset
│   │   ├── HAM10000_metadata.csv
│   │   ├── HAM10000_images_part_1/
│   │   └── HAM10000_images_part_2/
│   └── processed/                   # Generated statistics & splits
├── figures/
│   ├── data_analysis/               # Section 1 visualisations
│   ├── model_evaluation/            # Section 2 evaluation plots
│   └── gradcam/                     # Grad-CAM explanation images
├── report/                          # Assignment report documents
├── presentation/                    # Presentation slides & recording
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd project_3_ai
python -m venv venv
source venv/bin/activate          # macOS/Linux
pip install -r requirements.txt
```

### 2. Download Dataset

Download HAM10000 from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and extract into `data/raw/`:

```
data/raw/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/
```

### 3. Run Analysis Pipeline

```bash
python analysis/01_data_analysis.py            # Data exploration & visualisations
python analysis/02_data_transformation.py      # Preprocessing & splitting
python research/03_model_training.py           # Train models
python research/04_model_evaluation.py         # Evaluate & compare
```

### 4. Launch Application

```bash
python app/app.py
# Opens at http://localhost:7860
```

---

## Dataset

**HAM10000** ("Human Against Machine with 10,000 training images")

| Class | Full Name | Risk Level |
|---|---|---|
| `akiec` | Actinic Keratoses / Bowen's Disease | Pre-cancerous |
| `bcc` | Basal Cell Carcinoma | Malignant |
| `bkl` | Benign Keratosis | Benign |
| `df` | Dermatofibroma | Benign |
| `mel` | Melanoma | Malignant |
| `nv` | Melanocytic Nevi | Benign |
| `vasc` | Vascular Lesions | Benign |

**Source:** Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *Scientific Data*, 5, 180161.

---

## Models

| Model | Architecture | Purpose |
|---|---|---|
| **Custom CNN** | 4× Conv2D + BatchNorm + MaxPool → Dense | Baseline comparison |
| **EfficientNetB0** | ImageNet pre-trained + custom head | Primary (transfer learning) |

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| `tensorflow` | ≥ 2.12.0 | Deep learning framework |
| `numpy` | ≥ 1.23.0 | Numerical computation |
| `pandas` | ≥ 1.5.0 | Data manipulation |
| `matplotlib` | ≥ 3.6.0 | Visualisation |
| `seaborn` | ≥ 0.12.0 | Statistical visualisation |
| `scikit-learn` | ≥ 1.2.0 | Metrics & data splitting |
| `Pillow` | ≥ 9.3.0 | Image processing |
| `opencv-python` | ≥ 4.7.0 | Image processing |
| `gradio` | ≥ 4.0.0 | Interactive web interface |

---

## References

1. Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset. *Scientific Data*, 5, 180161.
2. Esteva, A., et al. (2017). Dermatologist-level classification of skin cancer. *Nature*, 542, 115–118.
3. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for CNNs. *ICML 2019*.
4. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks. *ICCV 2017*.

---

> ⚠️ **Disclaimer:** This system is for educational purposes only and is not a substitute for professional medical diagnosis.
