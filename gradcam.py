"""
gradcam.py - Grad-CAM Visualization

Shows which regions of an image the model focuses on when making a decision.
Visualizes them as heatmaps.

A highly effective visualization method for the graduation project presentation.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob
import random

# --- SETTINGS ---
MODEL_PATH = "results/best_model_fold1.pth"
CLASS_NAMES = ['Normal', 'Steatosis']
RESULTS_DIR = "results/gradcam"

os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- TRANSFORMS ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class GradCAM:
    """Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

    Uses the gradients of the last convolutional layer to compute which
    regions the model focuses on.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Generate a Grad-CAM heatmap.

        Args:
            input_tensor: Normalized image tensor [1, 3, 224, 224]
            target_class: Target class (uses the predicted class if None)

        Returns:
            numpy array: normalized heatmap of size [224, 224]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # Global Average Pooling
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)  # ReLU - only positive contributions

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, target_class, output


def apply_heatmap(image_np, heatmap, alpha=0.4):
    """Overlay the heatmap on top of the original image.

    Args:
        image_np: Original image [H, W, 3] (range 0-1)
        heatmap: Grad-CAM output [H, W] (range 0-1)
        alpha: Opacity ratio

    Returns:
        numpy array: Superimposed image
    """
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

    superimposed = alpha * heatmap_colored + (1 - alpha) * image_np
    superimposed = np.clip(superimposed, 0, 1)

    return superimposed


def visualize_gradcam(model, image_path, gradcam_obj, save_path=None):
    """Produce a Grad-CAM visualization for a single image.

    Shows 3 panels: Original | Heatmap | Superimposed
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((224, 224))
    image_np = np.array(image_resized) / 255.0

    # Preprocess
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Generate Grad-CAM
    heatmap, pred_class, output = gradcam_obj.generate(input_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)

    # Superimposed image
    superimposed = apply_heatmap(image_np, heatmap)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Original
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # 2. Heatmap
    im = axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Superimposed
    axes[2].imshow(superimposed)
    pred_name = CLASS_NAMES[pred_class]
    pred_prob = probs[0, pred_class].item()
    color = '#4CAF50' if pred_name == 'Normal' else '#F44336'
    axes[2].set_title(
        f'Prediction: {pred_name} ({100*pred_prob:.1f}%)',
        fontsize=12, fontweight='bold', color=color
    )
    axes[2].axis('off')

    filename = os.path.basename(image_path)
    plt.suptitle(f'Grad-CAM Analysis - {filename}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return pred_name, pred_prob


# ============================================================
#                         MAIN PROGRAM
# ============================================================
if __name__ == "__main__":
    # Load model
    print("Loading model...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print("Run train.py first to train the model.")
        exit(1)

    model = model.to(device)
    model.eval()

    # Create the Grad-CAM object (ResNet18's last convolutional layer: layer4)
    gradcam = GradCAM(model, model.layer4[-1])

    # --- Select random samples from each class ---
    print("\nGenerating Grad-CAM visualizations...\n")

    for class_name in ['normal', 'steatosis']:
        class_dir = f"dataset/train/{class_name}"
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found, skipping.")
            continue

        all_images = glob.glob(os.path.join(class_dir, "*.png"))
        if not all_images:
            continue

        # Select 4 random samples
        samples = random.sample(all_images, min(4, len(all_images)))

        for i, img_path in enumerate(samples):
            save_path = os.path.join(RESULTS_DIR, f"gradcam_{class_name}_{i+1}.png")
            pred, prob = visualize_gradcam(model, img_path, gradcam, save_path=save_path)
            print(f"  [{class_name.upper()}] {os.path.basename(img_path)} -> "
                  f"Prediction: {pred} ({100*prob:.1f}%) | Saved: {save_path}")

    # --- Show all samples in a single grid ---
    print("\nCreating summary grid...")

    normal_imgs = glob.glob(os.path.join("dataset/train/normal", "*.png"))
    steatosis_imgs = glob.glob(os.path.join("dataset/train/steatosis", "*.png"))

    if normal_imgs and steatosis_imgs:
        n_samples = 3
        selected = (random.sample(normal_imgs, min(n_samples, len(normal_imgs))) +
                    random.sample(steatosis_imgs, min(n_samples, len(steatosis_imgs))))

        fig, axes = plt.subplots(len(selected), 3, figsize=(15, 4 * len(selected)))

        for row, img_path in enumerate(selected):
            image = Image.open(img_path).convert("RGB")
            image_resized = image.resize((224, 224))
            image_np = np.array(image_resized) / 255.0

            input_tensor = preprocess(image).unsqueeze(0).to(device)
            heatmap, pred_class, output = gradcam.generate(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)

            superimposed = apply_heatmap(image_np, heatmap)

            # Original
            axes[row, 0].imshow(image_np)
            actual = "Normal" if "normal" in img_path else "Steatosis"
            axes[row, 0].set_ylabel(f'True: {actual}', fontsize=11, fontweight='bold')
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])

            # Heatmap
            axes[row, 1].imshow(heatmap, cmap='jet')
            axes[row, 1].axis('off')

            # Superimposed
            axes[row, 2].imshow(superimposed)
            pred_name = CLASS_NAMES[pred_class]
            color = '#4CAF50' if pred_name == actual else '#F44336'
            axes[row, 2].set_title(
                f'{pred_name} ({100*probs[0, pred_class].item():.1f}%)',
                color=color, fontsize=11
            )
            axes[row, 2].axis('off')

        axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Grad-CAM', fontsize=12, fontweight='bold')
        axes[0, 2].set_title('Prediction', fontsize=12, fontweight='bold')

        plt.suptitle('Grad-CAM Summary - Where is the model looking?',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'gradcam_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Summary saved: {RESULTS_DIR}/gradcam_summary.png")

    print("\nGrad-CAM visualizations complete!")
