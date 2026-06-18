"""
evaluate.py - Comprehensive Evaluation of the Liver Steatosis Model

Evaluates the performance of the trained model with detailed metrics:
- Confusion Matrix (visual)
- Classification Report (Precision, Recall, F1)
- ROC Curve & AUC Score
- Sensitivity / Specificity (medical standard)
- Visualization of misclassified samples
"""

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, ConfusionMatrixDisplay
)
import os
import json

# --- SETTINGS ---
MODEL_PATH = "results/best_model_fold1.pth"  # Path to the best model
DATA_DIR = "dataset/train"
CLASS_NAMES = ['Normal', 'Steatosis']
RESULTS_DIR = "results/evaluation"
BATCH_SIZE = 32

os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- LOAD MODEL ---
print("Loading model...")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print(f"Model loaded: {MODEL_PATH}")

# --- DATA PREPARATION ---
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Total test samples: {len(dataset)}")
print(f"Class distribution: {dict(zip(CLASS_NAMES, [dataset.targets.count(i) for i in range(2)]))}")

# --- RUN PREDICTIONS ---
print("\nRunning predictions...")
all_labels = []
all_preds = []
all_probs = []  # Probability scores for ROC
all_paths = []

softmax = nn.Softmax(dim=1)

with torch.no_grad():
    for inputs, labels in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = softmax(outputs)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Steatosis probability

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# ============================================================
#               1. CONFUSION MATRIX
# ============================================================
print("\n" + "=" * 50)
print("  CONFUSION MATRIX")
print("=" * 50)

cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

print(f"\n  True Negative  (Normal->Normal):     {tn}")
print(f"  False Positive (Normal->Steatosis):   {fp}")
print(f"  False Negative (Steatosis->Normal):   {fn}  <- Most dangerous error!")
print(f"  True Positive  (Steatosis->Steatosis):{tp}")

# Confusion Matrix visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count CM
disp1 = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
disp1.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title('Confusion Matrix (Count)', fontsize=14, fontweight='bold')

# Percentage CM
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
disp2 = ConfusionMatrixDisplay(cm_norm, display_labels=CLASS_NAMES)
disp2.plot(ax=axes[1], cmap='Oranges', values_format='.2%')
axes[1].set_title('Confusion Matrix (Percentage)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: {RESULTS_DIR}/confusion_matrix.png")


# ============================================================
#            2. CLASSIFICATION REPORT
# ============================================================
print("\n" + "=" * 50)
print("  CLASSIFICATION REPORT")
print("=" * 50)

report = classification_report(all_labels, all_preds,
                                target_names=CLASS_NAMES,
                                digits=4)
print(report)

# Save the report to a file
with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w') as f:
    f.write(report)


# ============================================================
#            3. MEDICAL METRICS
# ============================================================
print("=" * 50)
print("  MEDICAL METRICS")
print("=" * 50)

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall = Sensitivity
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0           # Positive Predictive Value = Precision
npv = tn / (tn + fn) if (tn + fn) > 0 else 0           # Negative Predictive Value
accuracy = (tp + tn) / (tp + tn + fp + fn)

print(f"\n  Accuracy:     {100*accuracy:.2f}%")
print(f"  Sensitivity:  {100*sensitivity:.2f}%  (rate of catching steatosis)")
print(f"  Specificity:  {100*specificity:.2f}%  (rate of correctly identifying normal)")
print(f"  PPV:          {100*ppv:.2f}%  (rate of being correct when predicting steatosis)")
print(f"  NPV:          {100*npv:.2f}%  (rate of being correct when predicting normal)")

medical_metrics = {
    'accuracy': round(accuracy, 4),
    'sensitivity': round(sensitivity, 4),
    'specificity': round(specificity, 4),
    'ppv': round(ppv, 4),
    'npv': round(npv, 4),
    'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
}

with open(os.path.join(RESULTS_DIR, 'medical_metrics.json'), 'w') as f:
    json.dump(medical_metrics, f, indent=2)


# ============================================================
#             4. ROC CURVE & AUC
# ============================================================
print("\n" + "=" * 50)
print("  ROC CURVE & AUC")
print("=" * 50)

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
print(f"\n  AUC Score: {roc_auc:.4f}")

# Find the optimal threshold (Youden's J index)
j_scores = tpr - fpr
best_threshold_idx = np.argmax(j_scores)
best_threshold = thresholds[best_threshold_idx]
print(f"  Optimal Threshold: {best_threshold:.4f}")
print(f"  At this threshold -> Sensitivity: {100*tpr[best_threshold_idx]:.1f}%, "
      f"Specificity: {100*(1-fpr[best_threshold_idx]):.1f}%")

# Plot ROC Curve
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr, tpr, color='#2196F3', lw=2.5,
        label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guess')

# Mark the optimal threshold point
ax.scatter(fpr[best_threshold_idx], tpr[best_threshold_idx],
           color='red', s=100, zorder=5,
           label=f'Optimal Threshold ({best_threshold:.3f})')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title('ROC Curve - Liver Steatosis Detection', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {RESULTS_DIR}/roc_curve.png")


# ============================================================
#       5. MISCLASSIFIED SAMPLES
# ============================================================
print("\n" + "=" * 50)
print("  MISCLASSIFIED SAMPLES")
print("=" * 50)

# Reload the original images without normalization
raw_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
raw_dataset = datasets.ImageFolder(DATA_DIR, transform=raw_transforms)

# Find the misclassified indices
wrong_indices = np.where(all_labels != all_preds)[0]
print(f"\n  Total wrong: {len(wrong_indices)} / {len(all_labels)} "
      f"({100*len(wrong_indices)/len(all_labels):.1f}%)")

# Show the first 12 misclassified samples
n_show = min(12, len(wrong_indices))
if n_show > 0:
    fig, axes = plt.subplots(2, min(6, n_show), figsize=(18, 7))
    if n_show <= 6:
        axes = axes.reshape(1, -1) if n_show <= 6 else axes

    for i in range(n_show):
        idx = wrong_indices[i]
        img, _ = raw_dataset[idx]
        img_np = img.numpy().transpose(1, 2, 0)

        row = i // 6
        col = i % 6
        if n_show > 6:
            ax = axes[row, col]
        else:
            ax = axes[0, i] if len(axes.shape) > 1 else axes[i]

        ax.imshow(img_np)
        ax.set_title(
            f"True: {CLASS_NAMES[all_labels[idx]]}\n"
            f"Pred: {CLASS_NAMES[all_preds[idx]]}\n"
            f"P(steatosis): {all_probs[idx]:.3f}",
            color='red', fontsize=9
        )
        ax.axis('off')

    # Hide empty subplots
    for i in range(n_show, (2 if n_show > 6 else 1) * 6):
        row = i // 6
        col = i % 6
        if n_show > 6:
            axes[row, col].axis('off')

    plt.suptitle('Misclassified Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'misclassified_samples.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {RESULTS_DIR}/misclassified_samples.png")


# ============================================================
#         6. SUMMARY
# ============================================================
print("\n" + "=" * 50)
print("  EVALUATION COMPLETE")
print("=" * 50)
print(f"\n  Outputs saved to the {RESULTS_DIR}/ folder:")
print(f"    - confusion_matrix.png")
print(f"    - roc_curve.png")
print(f"    - misclassified_samples.png")
print(f"    - classification_report.txt")
print(f"    - medical_metrics.json")
