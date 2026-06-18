"""
label.py - Automated Labeling via Morphological Blob Detection

Second stage of the pipeline. Automatically classifies the raw patches
extracted by patch.py without any manual annotation. Binary thresholding
followed by morphological opening detects lipid vacuole-like circular
structures (blobs); patches containing 5 or more blobs are labeled
"steatosis", the rest "normal". The threshold (5) was selected through the
ablation study in validate_labels.py and yields a near 50-50 class balance.
"""

import cv2
import os
import shutil
import numpy as np
import glob

# --- SETTINGS ---
SOURCE_ROOT = "dataset/patches"  # Raw patches produced by patch.py
DEST_ROOT = "dataset/train"      # Labeled output, read by the model (ImageFolder compatible)
BLOB_THRESHOLD = 5               # Minimum blob count for steatosis (chosen via ablation)

# Create the output folders
os.makedirs(os.path.join(DEST_ROOT, "steatosis"), exist_ok=True)
os.makedirs(os.path.join(DEST_ROOT, "normal"), exist_ok=True)


def count_blobs(img_path):
    """Return the number of lipid vacuole-like blobs in a patch.

    Converts to grayscale, applies thresholding and morphological opening,
    then counts structures matching the contour area (50-2000 px) and
    circularity (>0.6) criteria. The same algorithm is used in validate_labels.py.
    """
    img = cv2.imread(img_path)
    if img is None: return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Ignore very small dots and huge tears
        if 50 < area < 2000:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > 0.6: count += 1
    return count


# Find all PNGs in every subfolder
all_patches = glob.glob(os.path.join(SOURCE_ROOT, "*", "*.png"))
print(f"Analyzing {len(all_patches)} patches in total...")

stats = {"normal": 0, "steatosis": 0}

for path in all_patches:
    blob_n = count_blobs(path)
    filename = os.path.basename(path)  # e.g. TCGA-XY-1234_100_100.png

    if blob_n >= BLOB_THRESHOLD:
        shutil.copy(path, os.path.join(DEST_ROOT, "steatosis", filename))
        stats["steatosis"] += 1
    else:
        shutil.copy(path, os.path.join(DEST_ROOT, "normal", filename))
        stats["normal"] += 1

print(f"Done! Statistics: {stats}")
