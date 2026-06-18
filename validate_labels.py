"""
validate_labels.py - Labeling Method Ablation Study

Analyzes the effect of the blob count threshold on labeling.
By relabeling with different threshold values, it compares the resulting
class distributions of each candidate.

This analysis answers the question "why did we choose this threshold?" with
quantitative evidence for the graduation project thesis.
"""

import cv2
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import json

# --- SETTINGS ---
PATCHES_DIR = "dataset/patches"  # Raw patches (not yet labeled)
RESULTS_DIR = "results/label_analysis"

os.makedirs(RESULTS_DIR, exist_ok=True)

SELECTED_THRESHOLD = 5  # The threshold used in label.py, selected via this ablation


def count_blobs(img_path):
    """Count lipid droplet-like blobs in a patch.

    This function is consistent with the one in label.py.
    """
    img = cv2.imread(img_path)
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 2000:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > 0.6:
                count += 1
    return count


# ============================================================
#        1. COMPUTE BLOB COUNTS FOR ALL PATCHES
# ============================================================
print("Computing blob counts for all patches...")
print("(This may take a while)\n")

all_patches = glob.glob(os.path.join(PATCHES_DIR, "*", "*.png"))

if not all_patches:
    print("ERROR: No patch files found! Run patch.py first.")
    exit(1)

print(f"{len(all_patches)} patches will be analyzed.")

blob_counts = {}
patient_blobs = defaultdict(list)

for i, path in enumerate(all_patches):
    if (i + 1) % 500 == 0:
        print(f"  Processing: {i+1}/{len(all_patches)}...")

    blob_n = count_blobs(path)
    blob_counts[path] = blob_n

    filename = os.path.basename(path)
    patient_id = filename.split('_')[0]
    patient_blobs[patient_id].append(blob_n)

print(f"\nAnalysis complete. {len(all_patches)} patches processed.\n")


# ============================================================
#          2. BLOB DISTRIBUTION ANALYSIS
# ============================================================
all_blob_values = list(blob_counts.values())

print("=" * 50)
print("  BLOB COUNT STATISTICS")
print("=" * 50)
print(f"  Min:    {np.min(all_blob_values)}")
print(f"  Max:    {np.max(all_blob_values)}")
print(f"  Mean:   {np.mean(all_blob_values):.2f}")
print(f"  Median: {np.median(all_blob_values):.1f}")
print(f"  Std:    {np.std(all_blob_values):.2f}")

# Histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of all values
axes[0].hist(all_blob_values, bins=50, color='#2196F3', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Blob Count', fontsize=12)
axes[0].set_ylabel('Patch Count', fontsize=12)
axes[0].set_title('Blob Count Distribution (All Patches)', fontsize=13, fontweight='bold')
axes[0].axvline(x=SELECTED_THRESHOLD, color='red', linestyle='--', linewidth=2,
                label=f'Selected Threshold ({SELECTED_THRESHOLD})')
axes[0].legend(fontsize=10)

# Per-patient mean blob counts
patient_means = {pid: np.mean(blobs) for pid, blobs in patient_blobs.items()}
axes[1].bar(patient_means.keys(), patient_means.values(),
            color='#FF9800', edgecolor='white')
axes[1].set_xlabel('Patient ID', fontsize=12)
axes[1].set_ylabel('Mean Blob Count', fontsize=12)
axes[1].set_title('Per-Patient Mean Blob Count', fontsize=13, fontweight='bold')
axes[1].axhline(y=SELECTED_THRESHOLD, color='red', linestyle='--', linewidth=2,
                label=f'Selected Threshold ({SELECTED_THRESHOLD})')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'blob_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: {RESULTS_DIR}/blob_distribution.png")


# ============================================================
#        3. LABELING WITH DIFFERENT THRESHOLDS
# ============================================================
print("\n" + "=" * 50)
print("  THRESHOLD ABLATION STUDY")
print("=" * 50)

thresholds = [3, 5, 7, 10, 15, 20, 25, 30]
threshold_results = {}

for threshold in thresholds:
    n_steatosis = sum(1 for b in all_blob_values if b >= threshold)
    n_normal = sum(1 for b in all_blob_values if b < threshold)
    ratio = n_steatosis / len(all_blob_values) if len(all_blob_values) > 0 else 0

    threshold_results[threshold] = {
        'n_normal': n_normal,
        'n_steatosis': n_steatosis,
        'steatosis_ratio': round(ratio, 4),
        'balance_ratio': round(n_steatosis / n_normal if n_normal > 0 else 0, 4)
    }

    print(f"  Threshold={threshold:2d}: Normal={n_normal:6d} | "
          f"Steatosis={n_steatosis:6d} | "
          f"Steatosis Ratio={100*ratio:.1f}% | "
          f"Balance Ratio={n_steatosis/n_normal if n_normal > 0 else 0:.3f}")

# Threshold effect plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

normal_counts = [threshold_results[t]['n_normal'] for t in thresholds]
steatosis_counts = [threshold_results[t]['n_steatosis'] for t in thresholds]
ratios = [threshold_results[t]['steatosis_ratio'] for t in thresholds]

# Stacked bar chart
width = 1.5
axes[0].bar(thresholds, normal_counts, width, label='Normal', color='#4CAF50', alpha=0.8)
axes[0].bar(thresholds, steatosis_counts, width, bottom=normal_counts,
            label='Steatosis', color='#F44336', alpha=0.8)
axes[0].set_xlabel('Blob Threshold', fontsize=12)
axes[0].set_ylabel('Patch Count', fontsize=12)
axes[0].set_title('Class Distribution by Threshold', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].axvline(x=SELECTED_THRESHOLD, color='blue', linestyle=':', linewidth=2, alpha=0.5,
                label=f'Selected ({SELECTED_THRESHOLD})')

# Ratio line plot
axes[1].plot(thresholds, [r * 100 for r in ratios], 'o-', color='#9C27B0',
             linewidth=2, markersize=8)
axes[1].set_xlabel('Blob Threshold', fontsize=12)
axes[1].set_ylabel('Steatosis Ratio (%)', fontsize=12)
axes[1].set_title('Steatosis Ratio by Threshold', fontsize=13, fontweight='bold')
axes[1].axvline(x=SELECTED_THRESHOLD, color='blue', linestyle=':', linewidth=2, alpha=0.5,
                label=f'Selected ({SELECTED_THRESHOLD})')
axes[1].axhline(y=50, color='green', linestyle='--', alpha=0.3, label='Ideal Balance (50%)')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'threshold_ablation.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: {RESULTS_DIR}/threshold_ablation.png")


# ============================================================
#     4. PER-PATIENT ANALYSIS - blob profile for each patient
# ============================================================
print("\n" + "=" * 50)
print("  PER-PATIENT BLOB PROFILES")
print("=" * 50)

n_patients = len(patient_blobs)
fig, axes = plt.subplots(2, (n_patients + 1) // 2, figsize=(6 * ((n_patients + 1) // 2), 10))
axes = axes.flatten()

for i, (patient_id, blobs) in enumerate(sorted(patient_blobs.items())):
    if i >= len(axes):
        break
    axes[i].hist(blobs, bins=30, color='#2196F3', edgecolor='white', alpha=0.8)
    axes[i].axvline(x=SELECTED_THRESHOLD, color='red', linestyle='--', linewidth=2,
                     label=f'Threshold={SELECTED_THRESHOLD}')
    axes[i].set_title(f'Patient {patient_id}\n({len(blobs)} patches, mean={np.mean(blobs):.1f})',
                      fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Blob Count')
    axes[i].legend(fontsize=8)

# Hide empty subplots
for i in range(n_patients, len(axes)):
    axes[i].axis('off')

plt.suptitle('Per-Patient Blob Count Distributions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'patient_blob_profiles.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {RESULTS_DIR}/patient_blob_profiles.png")


# --- Save results to JSON ---
summary = {
    'total_patches': len(all_patches),
    'blob_statistics': {
        'min': int(np.min(all_blob_values)),
        'max': int(np.max(all_blob_values)),
        'mean': round(float(np.mean(all_blob_values)), 2),
        'median': round(float(np.median(all_blob_values)), 1),
        'std': round(float(np.std(all_blob_values)), 2)
    },
    'threshold_analysis': {str(k): v for k, v in threshold_results.items()},
    'patient_statistics': {
        pid: {
            'n_patches': len(blobs),
            'mean_blobs': round(float(np.mean(blobs)), 2),
            'median_blobs': round(float(np.median(blobs)), 1),
            'std_blobs': round(float(np.std(blobs)), 2)
        }
        for pid, blobs in patient_blobs.items()
    }
}

with open(os.path.join(RESULTS_DIR, 'label_analysis.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n  Results saved: {RESULTS_DIR}/label_analysis.json")
print("\nLabel analysis complete!")
