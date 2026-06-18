"""
test_model.py - Prediction on a Single Image or Folder

Uses the trained model to:
- Predict on a single patch
- Test all images in a folder
- Visualize the results
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse

# --- SETTINGS ---
MODEL_PATH = "results/best_model_fold1.pth"
CLASS_NAMES = ['Normal', 'Steatosis']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD MODEL ---
def load_model(model_path):
    """Load the trained model."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run train.py first to train the model."
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# --- TRANSFORMS ---
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_single(model, image_path):
    """Predict on a single image.

    Returns:
        tuple: (class name, confidence score, all probabilities)
    """
    image = Image.open(image_path).convert("RGB")
    input_tensor = test_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    pred_class = CLASS_NAMES[pred.item()]
    return pred_class, confidence.item(), probs[0].cpu().numpy()


def predict_folder(model, folder_path, max_show=12):
    """Predict on all images in a folder and visualize them."""
    image_files = glob.glob(os.path.join(folder_path, "*.png"))
    if not image_files:
        print(f"Error: No PNG files found in {folder_path}!")
        return

    print(f"\nRunning predictions on {len(image_files)} images...\n")

    results = {'Normal': 0, 'Steatosis': 0}
    predictions = []

    for img_path in image_files:
        pred_class, confidence, probs = predict_single(model, img_path)
        results[pred_class] += 1
        predictions.append({
            'path': img_path,
            'pred': pred_class,
            'confidence': confidence,
            'probs': probs
        })

    # Show statistics
    total = len(image_files)
    print(f"Results:")
    print(f"  Normal:    {results['Normal']} ({100*results['Normal']/total:.1f}%)")
    print(f"  Steatosis: {results['Steatosis']} ({100*results['Steatosis']/total:.1f}%)")

    # Visualize random samples
    import random
    show_indices = random.sample(range(len(predictions)), min(max_show, len(predictions)))

    n_cols = min(4, len(show_indices))
    n_rows = (len(show_indices) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, idx in enumerate(show_indices):
        pred = predictions[idx]
        img = Image.open(pred['path']).convert("RGB")

        axes[i].imshow(img)
        color = '#4CAF50' if pred['pred'] == 'Normal' else '#F44336'
        axes[i].set_title(
            f"{pred['pred']}\n({100*pred['confidence']:.1f}% confidence)",
            color=color, fontsize=11, fontweight='bold'
        )
        axes[i].axis('off')

    # Hide empty subplots
    for i in range(len(show_indices), len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Model Predictions - {os.path.basename(folder_path)}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join("results", "test_predictions.png"), dpi=150, bbox_inches='tight')
    plt.show()

    return predictions


# ============================================================
#                         MAIN PROGRAM
# ============================================================
if __name__ == "__main__":
    print("Loading model...")
    model = load_model(MODEL_PATH)
    print("Model ready!\n")

    # By default, test samples from the dataset/train folder
    # Show mixed samples from Normal and Steatosis
    print("=== TEST ON NORMAL SAMPLES ===")
    predict_folder(model, "dataset/train/normal", max_show=6)

    print("\n=== TEST ON STEATOSIS SAMPLES ===")
    predict_folder(model, "dataset/train/steatosis", max_show=6)
