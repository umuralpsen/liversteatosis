"""
visualize_samples.py - Visualize Dataset Samples

Selects random samples from each class in the labeled training data
(dataset/train) and displays them side by side. Useful for a quick visual
inspection of the dataset and a sanity check of the labeling quality.
"""

import matplotlib.pyplot as plt
import os
import random
import cv2

# Folder paths
steatosis_dir = "dataset/train/steatosis"
normal_dir = "dataset/train/normal"

# Select 5 random samples from each class
steatosis_samples = random.sample(os.listdir(steatosis_dir), 5)
normal_samples = random.sample(os.listdir(normal_dir), 5)

# Visualization
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, img_name in enumerate(steatosis_samples):
    img_path = os.path.join(steatosis_dir, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0, i].imshow(img)
    axes[0, i].set_title("Steatosis (Fatty)")
    axes[0, i].axis('off')

for i, img_name in enumerate(normal_samples):
    img_path = os.path.join(normal_dir, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[1, i].imshow(img)
    axes[1, i].set_title("Normal")
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()

print("You can inspect the images in the window that opens.")
