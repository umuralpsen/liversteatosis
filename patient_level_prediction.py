"""
patient_level_prediction.py - Patient-Level Prediction

Aggregates patch-level predictions to the patient level:
- Majority Voting: majority vote of each patient's patches
- Mean Probability: average of steatosis probabilities
- Confidence Analysis: detailed statistics per patient

The clinically meaningful result is the patient-level outcome, not the patch-level one.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import glob
import json
from collections import defaultdict
import matplotlib.pyplot as plt

# --- SETTINGS ---
MODEL_PATH = "results/best_model_fold1.pth"
PATCHES_DIR = "dataset/train"  # Contains the normal/ and steatosis/ folders
CLASS_NAMES = ['Normal', 'Steatosis']
RESULTS_DIR = "results/patient_level"
BATCH_SIZE = 64

os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- TRANSFORMS ---
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- LOAD MODEL ---
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
print("Model ready!\n")

# --- GROUP ALL PATCHES BY PATIENT ---
print("Grouping patches by patient...")
patient_patches = defaultdict(list)
patient_labels = {}  # Each patient's blob-based ground-truth label (folder based)

for class_name in ['normal', 'steatosis']:
    class_dir = os.path.join(PATCHES_DIR, class_name)
    if not os.path.exists(class_dir):
        continue

    label = 0 if class_name == 'normal' else 1

    for img_path in glob.glob(os.path.join(class_dir, "*.png")):
        filename = os.path.basename(img_path)
        patient_id = filename.split('_')[0]
        patient_patches[patient_id].append({
            'path': img_path,
            'label': label
        })

print(f"Found {len(patient_patches)} patients in total.\n")

# --- PREDICT FOR EACH PATIENT ---
print("Running patient-level predictions...\n")
softmax = nn.Softmax(dim=1)

patient_results = {}

for patient_id, patches in patient_patches.items():
    print(f"  Patient: {patient_id} ({len(patches)} patches)")

    all_probs = []
    all_preds = []

    # Predict in batches
    for i in range(0, len(patches), BATCH_SIZE):
        batch_patches = patches[i:i+BATCH_SIZE]
        batch_tensors = []

        for p in batch_patches:
            try:
                img = Image.open(p['path']).convert("RGB")
                tensor = test_transforms(img)
                batch_tensors.append(tensor)
            except Exception as e:
                continue

        if not batch_tensors:
            continue

        batch_input = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            outputs = model(batch_input)
            probs = softmax(outputs)
            _, preds = torch.max(outputs, 1)

            all_probs.extend(probs[:, 1].cpu().numpy())  # Steatosis probabilities
            all_preds.extend(preds.cpu().numpy())

    if not all_probs:
        continue

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # --- PATIENT-LEVEL DECISION ---
    # Method 1: Majority Voting
    n_steatosis = (all_preds == 1).sum()
    n_normal = (all_preds == 0).sum()
    majority_vote = "Steatosis" if n_steatosis > n_normal else "Normal"

    # Method 2: Mean Probability
    mean_prob = all_probs.mean()
    prob_decision = "Steatosis" if mean_prob > 0.5 else "Normal"

    # Method 3: Steatosis patch ratio
    steatosis_ratio = n_steatosis / len(all_preds)

    # Ground-truth label distribution (blob based)
    patch_labels = [p['label'] for p in patches]
    actual_steatosis_ratio = sum(patch_labels) / len(patch_labels)

    patient_results[patient_id] = {
        'total_patches': len(all_preds),
        'n_pred_normal': int(n_normal),
        'n_pred_steatosis': int(n_steatosis),
        'steatosis_ratio': round(float(steatosis_ratio), 4),
        'mean_steatosis_prob': round(float(mean_prob), 4),
        'majority_vote': majority_vote,
        'prob_decision': prob_decision,
        'actual_steatosis_ratio': round(float(actual_steatosis_ratio), 4),
        'prob_std': round(float(all_probs.std()), 4),
        'prob_min': round(float(all_probs.min()), 4),
        'prob_max': round(float(all_probs.max()), 4),
    }

    print(f"    -> Majority Vote: {majority_vote} "
          f"(Steatosis: {n_steatosis}/{len(all_preds)} = {100*steatosis_ratio:.1f}%)")
    print(f"    -> Mean Probability: {mean_prob:.3f} -> {prob_decision}")


# ============================================================
#              VISUALIZE RESULTS
# ============================================================
print("\n\nVisualizing results...")

patients = list(patient_results.keys())
n_patients = len(patients)

if n_patients > 0:
    # --- 1. Per-Patient Steatosis Probability Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart: steatosis patch ratio for each patient
    steatosis_ratios = [patient_results[p]['steatosis_ratio'] for p in patients]
    mean_probs = [patient_results[p]['mean_steatosis_prob'] for p in patients]

    colors = ['#F44336' if r > 0.5 else '#4CAF50' for r in steatosis_ratios]

    axes[0].bar(patients, steatosis_ratios, color=colors, edgecolor='white', linewidth=1.5)
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Threshold (50%)')
    axes[0].set_xlabel('Patient ID', fontsize=12)
    axes[0].set_ylabel('Steatosis Patch Ratio', fontsize=12)
    axes[0].set_title('Majority Voting Results', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    # Bar chart: mean probability
    colors2 = ['#F44336' if p > 0.5 else '#4CAF50' for p in mean_probs]

    axes[1].bar(patients, mean_probs, color=colors2, edgecolor='white', linewidth=1.5)
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Threshold (0.5)')
    axes[1].set_xlabel('Patient ID', fontsize=12)
    axes[1].set_ylabel('Average Steatosis Probability', fontsize=12)
    axes[1].set_title('Probability-Based Results', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].set_ylim(0, 1)

    plt.suptitle('Patient-Level Liver Steatosis Analysis',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'patient_level_results.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # --- 2. Detailed Table ---
    fig, ax = plt.subplots(figsize=(14, 3 + n_patients * 0.5))
    ax.axis('off')

    table_data = []
    for p in patients:
        r = patient_results[p]
        table_data.append([
            p,
            r['total_patches'],
            f"{r['n_pred_normal']} / {r['n_pred_steatosis']}",
            f"{100*r['steatosis_ratio']:.1f}%",
            f"{r['mean_steatosis_prob']:.3f}",
            r['majority_vote'],
            r['prob_decision']
        ])

    headers = ['Patient ID', 'Total\nPatches', 'Normal /\nSteatosis',
               'Steatosis\nRatio', 'Avg.\nProb', 'Majority\nVote', 'Prob.\nDecision']

    table = ax.table(cellText=table_data, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Color the header row
    for i in range(len(headers)):
        table[0, i].set_facecolor('#2196F3')
        table[0, i].set_text_props(color='white', fontweight='bold')

    # Color the decision cells
    for row in range(1, n_patients + 1):
        for col in [5, 6]:  # Majority Vote and Probability Decision columns
            cell_text = table[row, col].get_text().get_text()
            if 'Steatosis' in cell_text:
                table[row, col].set_facecolor('#FFCDD2')
            else:
                table[row, col].set_facecolor('#C8E6C9')

    plt.title('Patient-Level Prediction Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'patient_summary_table.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

# --- Save results to JSON ---
with open(os.path.join(RESULTS_DIR, 'patient_results.json'), 'w') as f:
    json.dump(patient_results, f, indent=2)

print(f"\nOutputs saved to the {RESULTS_DIR}/ folder:")
print(f"  - patient_level_results.png")
print(f"  - patient_summary_table.png")
print(f"  - patient_results.json")
print("\nPatient-level analysis complete!")
