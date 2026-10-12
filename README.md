# Plant Disease Detection System

## INFO813 Artificial Intelligence — Trimester 3, 2026

An AI-powered system for detecting plant diseases from leaf images using deep learning. The system uses a trained EfficientNetB0 model (transfer learning from ImageNet) to classify leaf images into 38 categories covering 14 crop species and their associated diseases.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Dataset Setup](#dataset-setup)
4. [Training the Models](#training-the-models)
5. [Running the Application](#running-the-application)
6. [Using the Application](#using-the-application)
7. [Input/Output Specifications](#inputoutput-specifications)
8. [Project Structure](#project-structure)
9. [Dependencies](#dependencies)
10. [Troubleshooting](#troubleshooting)
11. [References](#references)

---

## System Requirements

- **Operating System:** Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python:** 3.9 or later
- **RAM:** 8 GB minimum (16 GB recommended)
- **Storage:** ~2 GB for dataset + ~500 MB for models
- **GPU (optional):** NVIDIA GPU with CUDA support significantly speeds up training

---

## Installation

### Step 1: Clone or download the project

```bash
# If using git
git clone <repository-url>
cd "MIT Data Mining"
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

### Download the PlantVillage Dataset

1. Visit [Kaggle — PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
2. Download the dataset (you may need a free Kaggle account)
3. Extract the contents into the `data/` directory

The expected directory structure after extraction:

```
data/
├── Apple___Apple_scab/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── Apple___Black_rot/
│   └── ...
├── Apple___Cedar_apple_rust/
│   └── ...
├── ...
└── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
    └── ...
```

**Note:** Some Kaggle versions may have a nested structure like `data/PlantVillage/...` or `data/color/...`. The code auto-detects these structures.

---

## Training the Models

### Step 1: Generate the Jupyter Notebooks

```bash
cd notebooks
python create_01_data_analysis.py
python create_02_model_training.py
python create_03_model_comparison.py
```

This creates three `.ipynb` notebooks in the `notebooks/` directory.

### Step 2: Run the Notebooks

Open Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

Run the notebooks in order:
1. **`01_data_analysis.ipynb`** — Data exploration and visualisation
2. **`02_model_training.ipynb`** — Model training (Custom CNN + EfficientNetB0)
3. **`03_model_comparison.ipynb`** — Model evaluation and comparison

**Important:** Notebook 2 (model training) may take 1-3 hours on CPU or 15-30 minutes on GPU.

### Step 3: Verify saved models

After training, these files should exist in the `models/` directory:
- `custom_cnn_best.keras` — Best Custom CNN model
- `efficientnet_best.keras` — Best EfficientNetB0 model

---

## Running the Application

```bash
python app/app.py
```

This starts a local web server. Open your browser at:

```
http://localhost:7860
```

The application will automatically load the trained models on startup.

---

## Using the Application

### Step 1: Upload an Image
- Click **"Upload a leaf image"** or drag-and-drop a photo of a plant leaf
- Supported formats: JPG, JPEG, PNG, BMP
- Maximum file size: 10 MB

### Step 2: Select a Model
- **EfficientNetB0 (Recommended):** Higher accuracy, uses transfer learning
- **Custom CNN (Baseline):** Simpler model built from scratch

### Step 3: Click "Analyse Leaf"
The system will display:
- **Diagnosis:** The predicted disease (or healthy status) with confidence
- **Original Image:** The uploaded leaf photo
- **Grad-CAM:** A heatmap overlay showing which regions of the leaf the model focused on (red/yellow = high importance)
- **Confidence Chart:** Bar chart of the top 5 predictions with confidence percentages

### Step 4: Interpret Results
- **Green ✅:** Leaf is predicted as healthy
- **Red 🔴:** Disease detected — the specific disease and crop are identified
- **Confidence > 80%:** High confidence prediction
- **Confidence < 50%:** Low confidence — consider uploading a clearer image

---

## Input/Output Specifications

### Input
| Property | Requirement |
|---|---|
| **Format** | JPG, JPEG, PNG, or BMP |
| **Size** | Maximum 10 MB |
| **Content** | Single plant leaf, preferably against a plain background |
| **Resolution** | Any resolution (automatically resized to 224×224) |

### Output
| Output | Description |
|---|---|
| **Disease Name** | The identified crop species and disease condition |
| **Confidence Score** | Probability (0-100%) of the prediction being correct |
| **Grad-CAM Heatmap** | Visual explanation showing disease-relevant regions |
| **Top-5 Predictions** | Alternative diagnoses with their confidence scores |

### Supported Crops and Diseases (38 Classes)

| Crop | Conditions |
|---|---|
| Apple | Apple scab, Black rot, Cedar apple rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery mildew, Healthy |
| Corn | Cercospora leaf spot, Common rust, Northern leaf blight, Healthy |
| Grape | Black rot, Esca, Leaf blight, Healthy |
| Orange | Haunglongbing (Citrus greening) |
| Peach | Bacterial spot, Healthy |
| Pepper | Bacterial spot, Healthy |
| Potato | Early blight, Late blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery mildew |
| Strawberry | Leaf scorch, Healthy |
| Tomato | Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Mosaic virus, Yellow leaf curl virus, Healthy |

---

## Project Structure

```
MIT Data Mining/
├── app/
│   ├── app.py                  # Main Gradio application
│   ├── predict.py              # Prediction logic & Grad-CAM
│   └── class_labels.json       # Class index → name mapping
├── data/                       # PlantVillage dataset (not included)
│   └── <class_folders>/
├── models/
│   ├── custom_cnn_best.keras   # Trained Custom CNN
│   ├── efficientnet_best.keras # Trained EfficientNetB0
│   └── *.png                   # Saved visualisations
├── notebooks/
│   ├── create_01_data_analysis.py
│   ├── create_02_model_training.py
│   ├── create_03_model_comparison.py
│   ├── 01_data_analysis.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_comparison.ipynb
├── requirements.txt
└── README.md                   # This file
```

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| `tensorflow` | ≥ 2.12.0 | Deep learning framework (model building, training, inference) |
| `numpy` | ≥ 1.23.0 | Numerical computation |
| `pandas` | ≥ 1.5.0 | Data manipulation and analysis |
| `matplotlib` | ≥ 3.6.0 | Visualisation and Grad-CAM rendering |
| `seaborn` | ≥ 0.12.0 | Statistical visualisation |
| `scikit-learn` | ≥ 1.2.0 | Metrics, train/test split, class weights |
| `Pillow` | ≥ 9.3.0 | Image loading and manipulation |
| `gradio` | ≥ 3.40.0 | Interactive web interface |
| `nbformat` | ≥ 5.7.0 | Jupyter notebook generation |
| `jupyter` | ≥ 1.0.0 | Running notebooks |
| `tqdm` | ≥ 4.64.0 | Progress bars |

---

## Troubleshooting

### "Model not found" error
- Run the training notebooks first (`02_model_training.ipynb`)
- Ensure model files exist in the `models/` directory

### "Dataset not found" error
- Download the PlantVillage dataset from Kaggle
- Place it in the `data/` directory
- Check the directory structure matches the expected format

### Training is very slow
- Use Google Colab with GPU runtime (Runtime → Change runtime type → GPU)
- Reduce `BATCH_SIZE` if running out of memory
- Reduce the number of `EPOCHS`

### Gradio app won't start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that port 7860 is not in use by another application
- Try: `python app/app.py` from the project root directory

### Import errors
- Ensure you're using Python 3.9+: `python --version`
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

---

## References

1. Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv preprint arXiv:1511.08060*.

2. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419. https://doi.org/10.3389/fpls.2016.01419

3. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, 6105–6114.

4. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 618–626.

5. PlantVillage Dataset on Kaggle: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

### Software & Tools Used
- Python 3.9 (https://www.python.org/)
- TensorFlow 2.12 (https://www.tensorflow.org/)
- Gradio (https://gradio.app/)
- Jupyter Notebook (https://jupyter.org/)
- Google Colab (https://colab.research.google.com/) — for GPU-accelerated training
