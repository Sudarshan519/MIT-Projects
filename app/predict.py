"""
Prediction logic for the Plant Disease Detection application.
Handles model loading, image preprocessing, prediction, and Grad-CAM generation.

INFO813 Artificial Intelligence — Section 3: Software Implementation
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# =============================================================================
# Configuration
# =============================================================================
IMG_SIZE = 224
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
LABELS_PATH = os.path.join(os.path.dirname(__file__), 'class_labels.json')

# Supported file formats for validation
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp'}
MAX_FILE_SIZE_MB = 10


# =============================================================================
# Model Loading
# =============================================================================
def load_model(model_name='efficientnet'):
    """
    Load a trained Keras model from disk.

    Parameters:
        model_name (str): Either 'efficientnet' or 'custom_cnn'

    Returns:
        tf.keras.Model: Loaded model

    Raises:
        FileNotFoundError: If model file does not exist
        ValueError: If model_name is not recognised
    """
    model_files = {
        'efficientnet': [
            os.path.join(MODEL_DIR, 'efficientnet_best.keras'),
            os.path.join(MODEL_DIR, 'efficientnet_model.keras'),
        ],
        'custom_cnn': [
            os.path.join(MODEL_DIR, 'custom_cnn_best.keras'),
            os.path.join(MODEL_DIR, 'custom_cnn_model.keras'),
        ]
    }

    if model_name not in model_files:
        raise ValueError(
            f"Unknown model: '{model_name}'. Choose 'efficientnet' or 'custom_cnn'."
        )

    for path in model_files[model_name]:
        if os.path.exists(path):
            model = tf.keras.models.load_model(path)
            print(f"✓ Loaded model: {path}")
            return model

    raise FileNotFoundError(
        f"Model '{model_name}' not found. Expected one of: {model_files[model_name]}"
    )


def load_class_labels():
    """
    Load the class label mapping from JSON.

    Returns:
        dict: Mapping of integer indices to class names
    """
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            f"Class labels file not found at: {LABELS_PATH}. "
            "Run the training notebook first."
        )
    with open(LABELS_PATH, 'r') as f:
        label_map = json.load(f)
    return label_map


# =============================================================================
# Input Validation
# =============================================================================
def validate_image(image_path):
    """
    Validate the uploaded image file.

    Parameters:
        image_path (str): Path to the image file

    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    # Check file exists
    if not os.path.exists(image_path):
        return False, f"File not found: {image_path}"

    # Check file extension
    _, ext = os.path.splitext(image_path)
    if ext.lower() not in SUPPORTED_FORMATS:
        return False, (
            f"Unsupported file format: '{ext}'. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    # Check file size
    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, (
            f"File too large: {file_size_mb:.1f} MB. "
            f"Maximum allowed: {MAX_FILE_SIZE_MB} MB."
        )

    # Try to load the image
    try:
        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        if img_array.shape != (IMG_SIZE, IMG_SIZE, 3):
            return False, f"Invalid image shape: {img_array.shape}. Expected RGB image."
    except Exception as e:
        return False, f"Could not read image: {str(e)}"

    return True, None


# =============================================================================
# Prediction
# =============================================================================
def preprocess_image(image_path, model_type='efficientnet'):
    """
    Load and preprocess an image for model inference.

    Parameters:
        image_path (str): Path to the image file
        model_type (str): 'efficientnet' or 'custom_cnn' (affects normalisation)

    Returns:
        tuple: (preprocessed_array, display_image)
            - preprocessed_array: (1, 224, 224, 3) ready for model.predict()
            - display_image: (224, 224, 3) uint8 for display
    """
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    display_image = img_array.copy().astype(np.uint8)

    if model_type == 'custom_cnn':
        # Normalise to [0, 1]
        processed = img_array.astype(np.float32) / 255.0
    else:
        # EfficientNet: keep [0, 255] range
        processed = img_array.astype(np.float32)

    # Add batch dimension
    processed = np.expand_dims(processed, axis=0)

    return processed, display_image


def predict(model, image_path, label_map, model_type='efficientnet', top_k=3):
    """
    Make a prediction on an uploaded image.

    Parameters:
        model: Trained Keras model
        image_path (str): Path to the image file
        label_map (dict): Class index to name mapping
        model_type (str): 'efficientnet' or 'custom_cnn'
        top_k (int): Number of top predictions to return

    Returns:
        dict: {
            'predictions': [{'class': str, 'confidence': float, 'index': int}, ...],
            'display_image': np.ndarray,
            'raw_probabilities': np.ndarray
        }
    """
    # Validate input
    is_valid, error_msg = validate_image(image_path)
    if not is_valid:
        raise ValueError(error_msg)

    # Preprocess
    processed, display_image = preprocess_image(image_path, model_type)

    # Predict
    probabilities = model.predict(processed, verbose=0)[0]

    # Get top-k predictions
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    predictions = []
    for idx in top_indices:
        class_name = label_map.get(str(idx), f"Class {idx}")
        # Format class name for display
        readable_name = class_name.replace('___', ' — ').replace('_', ' ')
        predictions.append({
            'class': readable_name,
            'confidence': float(probabilities[idx]),
            'index': int(idx)
        })

    return {
        'predictions': predictions,
        'display_image': display_image,
        'raw_probabilities': probabilities
    }


# =============================================================================
# Grad-CAM
# =============================================================================
def generate_gradcam(model, image_path, model_type='efficientnet'):
    """
    Generate a Grad-CAM heatmap overlay for the predicted class.

    Parameters:
        model: Trained Keras model
        image_path (str): Path to the image file
        model_type (str): 'efficientnet' or 'custom_cnn'

    Returns:
        np.ndarray: (224, 224, 3) uint8 image with Grad-CAM overlay
    """
    processed, display_image = preprocess_image(image_path, model_type)

    # Find the last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers'):
            # Nested model (e.g., EfficientNetB0 inside Sequential)
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    last_conv_layer = sub_layer
                    # Build gradient model through the nested structure
                    try:
                        grad_model = tf.keras.models.Model(
                            inputs=model.inputs,
                            outputs=[layer.get_layer(sub_layer.name).output,
                                     model.output]
                        )
                    except Exception:
                        return display_image  # Fallback
                    break
            if last_conv_layer:
                break
        elif isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(layer.name).output, model.output]
            )
            break

    if last_conv_layer is None:
        return display_image  # Fallback if no conv layer found

    # Compute Grad-CAM
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(processed)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize heatmap
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], (IMG_SIZE, IMG_SIZE)
    ).numpy().squeeze()

    # Apply colourmap
    heatmap_coloured = cm.jet(heatmap_resized)[:, :, :3] * 255

    # Overlay on original image
    overlay = (display_image * 0.6 + heatmap_coloured * 0.4).astype(np.uint8)

    return overlay


# =============================================================================
# Confidence Bar Chart
# =============================================================================
def create_confidence_chart(predictions, top_k=5):
    """
    Create a horizontal bar chart of prediction confidences.

    Parameters:
        predictions (list): List of prediction dicts with 'class' and 'confidence'
        top_k (int): Number of predictions to display

    Returns:
        matplotlib.figure.Figure: The confidence bar chart
    """
    fig, ax = plt.subplots(figsize=(8, 3))

    classes = [p['class'][:35] for p in predictions[:top_k]]
    confidences = [p['confidence'] * 100 for p in predictions[:top_k]]

    # Colour coding: green for high confidence, red for low
    colours = ['#2ecc71' if c > 50 else '#e67e22' if c > 20 else '#e74c3c'
               for c in confidences]

    bars = ax.barh(range(len(classes)), confidences, color=colours,
                   edgecolor='white', height=0.6)

    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel('Confidence (%)', fontsize=11)
    ax.set_xlim(0, 105)
    ax.invert_yaxis()

    # Add percentage labels
    for bar, conf in zip(bars, confidences):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{conf:.1f}%', va='center', fontsize=10, fontweight='bold')

    ax.set_title('Prediction Confidence', fontsize=13, fontweight='bold')
    plt.tight_layout()

    return fig
