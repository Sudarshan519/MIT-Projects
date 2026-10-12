"""
Plant Disease Detection — Interactive Application
INFO813 Artificial Intelligence | Trimester 3, 2026

This application uses a trained deep learning model (EfficientNetB0) to detect
plant diseases from leaf images. Users can upload an image of a plant leaf and
receive a diagnosis with confidence scores and Grad-CAM visualisation.

Usage:
    python app.py

This will start a local Gradio web server (default: http://localhost:7860).

Requirements:
    pip install gradio tensorflow numpy matplotlib Pillow
"""

import os
import sys
import tempfile

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from predict import (
    load_model,
    load_class_labels,
    predict,
    generate_gradcam,
    create_confidence_chart,
    validate_image,
    SUPPORTED_FORMATS,
    MAX_FILE_SIZE_MB
)


# =============================================================================
# Global Model Loading (loaded once at startup)
# =============================================================================
print("=" * 60)
print("Plant Disease Detection System — Starting Up")
print("=" * 60)

# Try to load models
models = {}
try:
    models['efficientnet'] = load_model('efficientnet')
    print("✓ EfficientNetB0 model loaded successfully")
except FileNotFoundError as e:
    print(f"⚠ EfficientNetB0 model not found: {e}")

try:
    models['custom_cnn'] = load_model('custom_cnn')
    print("✓ Custom CNN model loaded successfully")
except FileNotFoundError as e:
    print(f"⚠ Custom CNN model not found: {e}")

if not models:
    print("ERROR: No models found! Please train the models first using the notebooks.")
    print("Expected model files in: ../models/")
    sys.exit(1)

# Load class labels
try:
    label_map = load_class_labels()
    print(f"✓ Loaded {len(label_map)} class labels")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# Default model
default_model = 'efficientnet' if 'efficientnet' in models else 'custom_cnn'
print(f"\nDefault model: {default_model}")
print(f"Available models: {list(models.keys())}")
print("=" * 60)


# =============================================================================
# Prediction Function for Gradio
# =============================================================================
def predict_disease(image, model_choice):
    """
    Main prediction function called by the Gradio interface.

    Parameters:
        image: Uploaded image (numpy array from Gradio)
        model_choice (str): Selected model name

    Returns:
        tuple: (original_image, gradcam_overlay, confidence_chart, result_text)
    """
    if image is None:
        return None, None, None, "⚠️ Please upload an image of a plant leaf."

    # Map model choice to internal name
    model_key = 'efficientnet' if 'EfficientNet' in model_choice else 'custom_cnn'
    if model_key not in models:
        return None, None, None, f"⚠️ Model '{model_key}' is not available."

    model = models[model_key]

    try:
        # Save uploaded image to temp file (Gradio provides numpy array)
        temp_path = os.path.join(tempfile.gettempdir(), 'plant_disease_input.jpg')
        plt.imsave(temp_path, image)

        # Validate
        is_valid, error_msg = validate_image(temp_path)
        if not is_valid:
            return None, None, None, f"⚠️ Validation Error: {error_msg}"

        # Make prediction
        result = predict(
            model=model,
            image_path=temp_path,
            label_map=label_map,
            model_type=model_key,
            top_k=5
        )

        # Generate Grad-CAM overlay
        gradcam_img = generate_gradcam(model, temp_path, model_type=model_key)

        # Create confidence chart
        chart_fig = create_confidence_chart(result['predictions'], top_k=5)

        # Format result text
        top_pred = result['predictions'][0]
        result_text = format_result_text(result['predictions'], model_key)

        # Convert chart to image
        chart_fig.canvas.draw()
        chart_img = np.frombuffer(chart_fig.canvas.tostring_rgb(), dtype=np.uint8)
        chart_img = chart_img.reshape(chart_fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(chart_fig)

        return result['display_image'], gradcam_img, chart_img, result_text

    except Exception as e:
        return None, None, None, f"❌ Error during prediction: {str(e)}"


def format_result_text(predictions, model_name):
    """
    Format prediction results as readable text.

    Parameters:
        predictions (list): List of prediction dicts
        model_name (str): Name of the model used

    Returns:
        str: Formatted result text
    """
    top = predictions[0]

    # Extract crop and disease
    parts = top['class'].split(' — ')
    crop = parts[0] if len(parts) > 0 else 'Unknown'
    disease = parts[1] if len(parts) > 1 else top['class']

    # Determine if healthy or diseased
    is_healthy = 'healthy' in disease.lower()

    if is_healthy:
        status_emoji = "✅"
        status = "HEALTHY"
        recommendation = "No treatment required. Continue regular care."
    else:
        status_emoji = "🔴"
        status = "DISEASE DETECTED"
        recommendation = (
            f"Disease identified: {disease}\n"
            f"Consult an agricultural expert for treatment options."
        )

    result = f"""
{status_emoji} {status}

📋 Diagnosis Results (Model: {model_name.replace('_', ' ').title()})
{'─' * 50}

🌿 Crop: {crop}
🔬 Condition: {disease}
📊 Confidence: {top['confidence']:.1%}

💊 Recommendation:
{recommendation}

📊 Top Predictions:
"""
    for i, pred in enumerate(predictions[:5], 1):
        bar_length = int(pred['confidence'] * 20)
        bar = '█' * bar_length + '░' * (20 - bar_length)
        result += f"  {i}. {pred['class'][:40]}\n     {bar} {pred['confidence']:.1%}\n"

    return result


# =============================================================================
# Gradio Interface
# =============================================================================
def create_interface():
    """Build and return the Gradio interface."""

    # Model choices (only show available models)
    model_choices = []
    if 'efficientnet' in models:
        model_choices.append("EfficientNetB0 (Recommended)")
    if 'custom_cnn' in models:
        model_choices.append("Custom CNN (Baseline)")

    with gr.Blocks(
        title="Plant Disease Detection System",
        theme=gr.themes.Soft(
            primary_hue="green",
            secondary_hue="emerald"
        )
    ) as app:

        # Header
        gr.Markdown("""
        # 🌿 Plant Disease Detection System
        ### INFO813 Artificial Intelligence — Trimester 3, 2026

        Upload a photo of a plant leaf to identify potential diseases.
        The system uses deep learning (EfficientNetB0) trained on the
        [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
        covering **38 classes** across **14 crop species**.

        ---
        """)

        with gr.Row():
            # Left column: Input
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Image")
                image_input = gr.Image(
                    label="Upload a leaf image",
                    type="numpy",
                    sources=["upload", "clipboard"],
                    height=300
                )

                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value=model_choices[0] if model_choices else None,
                    label="🤖 Select Model",
                    info="EfficientNetB0 is recommended for best accuracy"
                )

                predict_btn = gr.Button(
                    "🔍 Analyse Leaf",
                    variant="primary",
                    size="lg"
                )

                gr.Markdown(f"""
                **Supported formats:** {', '.join(SUPPORTED_FORMATS)}
                **Max file size:** {MAX_FILE_SIZE_MB} MB
                **Supported crops:** Apple, Blueberry, Cherry, Corn, Grape,
                Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash,
                Strawberry, Tomato
                """)

            # Right column: Results
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")

                result_text = gr.Textbox(
                    label="Diagnosis",
                    lines=15,
                    interactive=False,
                    placeholder="Upload an image and click 'Analyse Leaf' to see results..."
                )

                with gr.Row():
                    original_output = gr.Image(
                        label="Original Image",
                        height=250
                    )
                    gradcam_output = gr.Image(
                        label="Grad-CAM (Disease Focus Areas)",
                        height=250
                    )

                confidence_output = gr.Image(
                    label="Confidence Chart",
                    height=200
                )

        # Connect button
        predict_btn.click(
            fn=predict_disease,
            inputs=[image_input, model_dropdown],
            outputs=[original_output, gradcam_output, confidence_output, result_text]
        )

        # Also trigger on image upload
        image_input.change(
            fn=predict_disease,
            inputs=[image_input, model_dropdown],
            outputs=[original_output, gradcam_output, confidence_output, result_text]
        )

        # Footer
        gr.Markdown("""
        ---
        **Note:** This system is for educational purposes. Always consult a qualified
        agricultural expert for definitive plant disease diagnosis and treatment plans.

        **Dataset:** PlantVillage (Hughes & Salathé, 2015) |
        **Model:** EfficientNetB0 with Transfer Learning (Tan & Le, 2019)
        """)

    return app


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    app = create_interface()
    print("\n🚀 Starting Plant Disease Detection System...")
    print("   Open your browser at: http://localhost:7860\n")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
