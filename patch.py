"""
patch.py - Patch Extraction from Whole-Slide Images

First stage of the pipeline. Uses the OpenSlide library to tile whole-slide
images (WSI) in SVS format into non-overlapping 256x256 pixel patches. A
tissue detection filter discards background regions. Accepted patches are
saved into per-patient folders with the patient ID and spatial coordinates
encoded in the filename (e.g. TCGA-XY-1234_512_768.png).
"""

import openslide
import numpy as np
import os
import glob

# --- SETTINGS ---
SOURCE_DIR = "data"             # Folder containing the .svs files
OUTPUT_ROOT = "dataset/patches" # Folder where extracted raw patches are saved
PATCH_SIZE = 256
STEP_SIZE = 256                 # No overlap between patches


def is_tissue(region_np, threshold=210):
    """Check whether a patch contains tissue rather than background.

    A patch is accepted if its mean grayscale intensity is below the threshold
    (i.e. dark enough to be tissue). The threshold of 210 was found empirically
    to give the most balanced separation.
    """
    gray = np.mean(region_np, axis=2)
    return np.mean(gray) < threshold


svs_files = glob.glob(os.path.join(SOURCE_DIR, "*.svs"))
print(f"Found {len(svs_files)} slides in total.")

for slide_path in svs_files:
    filename = os.path.basename(slide_path).split('.')[0]  # File name (Patient ID)
    patient_dir = os.path.join(OUTPUT_ROOT, filename)
    os.makedirs(patient_dir, exist_ok=True)

    print(f"Processing: {filename}...")

    try:
        slide = openslide.OpenSlide(slide_path)
        w, h = slide.dimensions
    except:
        print(f"ERROR: {filename} is corrupted, skipping.")
        continue

    count = 0
    for y in range(0, h, STEP_SIZE):
        for x in range(0, w, STEP_SIZE):
            if x+PATCH_SIZE > w or y+PATCH_SIZE > h: continue

            try:
                patch = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
                if is_tissue(np.array(patch)):
                    # Encode Patient ID + coordinates in the filename (key for patient-level split)
                    patch.save(os.path.join(patient_dir, f"{filename}_{x}_{y}.png"))
                    count += 1
            except:
                continue
    print(f"-> {count} patches extracted.")
