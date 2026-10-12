# ============================================================================
# INFO813 Project 3 — Grad-CAM Explainability Module
# ============================================================================
# Computes Gradient-weighted Class Activation Mapping (Grad-CAM)
# highlights which regions of a dermoscopic image influenced the model's decision.
# ============================================================================

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation for PyTorch CNNs.
    Hooks into the final convolutional layer to extract gradients and activations.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
        
    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generate a 2D Grad-CAM heatmap normalized to [0, 1].
        input_tensor: (1, 3, H, W)
        target_class: integer index (if None, uses predicted class)
        """
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = F.relu(cam)  # Apply ReLU to keep only positive influence
        
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, target_class


def overlay_heatmap_on_image(original_image, heatmap, alpha=0.55, colormap=cv2.COLORMAP_JET):
    """
    Superimposes a Grad-CAM heatmap onto an RGB PIL Image or NumPy array.
    """
    if isinstance(original_image, Image.Image):
        img_np = np.array(original_image)
    else:
        img_np = original_image.copy()
        
    h, w = img_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    superimposed = np.uint8(alpha * heatmap_colored + (1 - alpha) * img_np)
    return superimposed, heatmap_resized
