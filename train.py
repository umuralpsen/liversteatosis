"""
train.py - Model Training (ResNet18 + Patient-Level Cross-Validation)

Third stage of the pipeline. Fine-tunes an ImageNet-pretrained ResNet18 using
patient-level 3-fold GroupKFold cross-validation. All patches from a given
patient always belong to a single fold (training or validation), preventing
data leakage. Class imbalance is handled with weighted cross-entropy loss and
a WeightedRandomSampler; data augmentation and early stopping prevent
overfitting. The best model and training logs for each fold are saved to the
results/ folder.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupKFold
import numpy as np
import os
import glob
import csv
import time
from PIL import Image
from collections import Counter

# --- SETTINGS ---
DATA_DIR = "dataset/train"       # Location of the labeled data
BATCH_SIZE = 32
EPOCHS = 10                      # Optimized for CPU (early stopping enabled)
LEARNING_RATE = 0.0001
N_FOLDS = 3                      # 3 folds for CPU (5 can be used on GPU)
PATIENCE = 3                     # Early stopping patience (stop after 3 epochs w/o improvement)
LOG_DIR = "results"              # Where training logs are saved

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(LOG_DIR, exist_ok=True)

# --- CUSTOM DATASET CLASS ---
class LiverDataset(Dataset):
    """Liver pathology image dataset.
    Extracts the Patient ID from file paths and pairs it with labels.
    """
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# --- DATA PREPARATION ---
print("Scanning data...")
normal_files = glob.glob(os.path.join(DATA_DIR, "normal", "*.png"))
steatosis_files = glob.glob(os.path.join(DATA_DIR, "steatosis", "*.png"))

all_files = normal_files + steatosis_files
# 0: Normal, 1: Steatosis
all_labels = [0] * len(normal_files) + [1] * len(steatosis_files)

# Extract Patient IDs (part of the filename up to the first "_")
groups = []
for f in all_files:
    filename = os.path.basename(f)
    patient_id = filename.split('_')[0]
    groups.append(patient_id)

print(f"Total Data: {len(all_files)}")
print(f"  Normal: {len(normal_files)} | Steatosis: {len(steatosis_files)}")
print(f"  Ratio: Normal {100*len(normal_files)/len(all_files):.1f}% | Steatosis {100*len(steatosis_files)/len(all_files):.1f}%")
print(f"Total Number of Patients: {len(set(groups))}")

# --- COMPUTE CLASS WEIGHTS ---
label_counts = Counter(all_labels)
total = len(all_labels)
class_weights = torch.tensor([
    total / (2 * label_counts[0]),  # Normal weight (decreases)
    total / (2 * label_counts[1])   # Steatosis weight (increases)
], dtype=torch.float32).to(device)
print(f"Class Weights: Normal={class_weights[0]:.3f}, Steatosis={class_weights[1]:.3f}")


# --- TRANSFORMS (AUGMENTATION) ---
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def create_weighted_sampler(labels):
    """Create a WeightedRandomSampler to address class imbalance.
    Samples steatosis examples more frequently to keep each batch balanced.
    """
    label_counts = Counter(labels)
    weights = [1.0 / label_counts[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    return sampler


# --- TRAINING FUNCTION ---
def train_one_fold(fold, train_files, train_labels, val_files, val_labels):
    """Training and validation loop for a single fold.

    Returns:
        dict: Fold results (best accuracy, epoch info, etc.)
    """
    print(f"\n{'='*60}")
    print(f"  STARTING FOLD {fold}/{N_FOLDS}")
    print(f"  Train: {len(train_files)} | Validation: {len(val_files)}")
    print(f"{'='*60}")

    # Create datasets
    train_ds = LiverDataset(train_files, train_labels, transform=train_transforms)
    val_ds = LiverDataset(val_files, val_labels, transform=val_transforms)

    # Balanced sampling with WeightedRandomSampler
    train_sampler = create_weighted_sampler(train_labels)

    # DataLoader (shuffle must be False when using a sampler)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- MODEL ---
    # ResNet18 + Transfer Learning (ImageNet weights)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # Class-weighted Cross-Entropy Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning Rate Scheduler - reduces LR when validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # --- TRAINING LOOP ---
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    fold_log = []

    for epoch in range(EPOCHS):
        start_time = time.time()

        # === Training ===
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # === Validation ===
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_tp = 0  # True Positive (steatosis correctly found)
        val_fp = 0  # False Positive
        val_fn = 0  # False Negative (steatosis missed)
        val_tn = 0  # True Negative

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

                # Confusion matrix elements
                for p, l in zip(preds, labels):
                    if p == 1 and l == 1: val_tp += 1
                    elif p == 1 and l == 0: val_fp += 1
                    elif p == 0 and l == 1: val_fn += 1
                    else: val_tn += 1

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        # Precision, Recall, F1
        precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
        recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  Epoch {epoch+1:02d}/{EPOCHS} ({elapsed:.0f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
              f"P: {precision:.3f} R: {recall:.3f} F1: {f1:.3f} | "
              f"LR: {current_lr:.6f}")

        # Save log
        fold_log.append({
            'fold': fold, 'epoch': epoch + 1,
            'train_loss': round(train_loss, 4), 'train_acc': round(train_acc, 2),
            'val_loss': round(val_loss, 4), 'val_acc': round(val_acc, 2),
            'precision': round(precision, 4), 'recall': round(recall, 4),
            'f1': round(f1, 4), 'lr': current_lr,
            'tp': val_tp, 'fp': val_fp, 'fn': val_fn, 'tn': val_tn
        })

        # Update LR scheduler
        scheduler.step(val_loss)

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(LOG_DIR, f"best_model_fold{fold}.pth"))
            print(f"  [OK] New best model saved! (Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= PATIENCE:
            print(f"  [STOP] No improvement for {PATIENCE} epochs. Early stopping!")
            break

    # Write fold results to CSV
    log_path = os.path.join(LOG_DIR, f"training_log_fold{fold}.csv")
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fold_log[0].keys())
        writer.writeheader()
        writer.writerows(fold_log)
    print(f"  Training log saved: {log_path}")

    return {
        'fold': fold,
        'best_acc': best_val_acc,
        'best_loss': best_val_loss,
        'final_precision': precision,
        'final_recall': recall,
        'final_f1': f1,
        'epochs_trained': epoch + 1
    }


# ============================================================
#                     MAIN TRAINING LOOP
# ============================================================
print("\n" + "=" * 60)
print(f"  STARTING {N_FOLDS}-FOLD CROSS-VALIDATION")
print("=" * 60)

gkf = GroupKFold(n_splits=N_FOLDS)
all_files_np = np.array(all_files)
all_labels_np = np.array(all_labels)
groups_np = np.array(groups)

fold_results = []
fold = 1

for train_idx, val_idx in gkf.split(all_files, all_labels, groups):
    # Split the data
    train_files = [all_files[i] for i in train_idx]
    train_labels_fold = [all_labels[i] for i in train_idx]
    val_files = [all_files[i] for i in val_idx]
    val_labels_fold = [all_labels[i] for i in val_idx]

    # Show the patient distribution of the fold
    train_patients = set([groups[i] for i in train_idx])
    val_patients = set([groups[i] for i in val_idx])
    print(f"\n  Training patients: {train_patients}")
    print(f"  Validation patients: {val_patients}")

    # Run training
    result = train_one_fold(fold, train_files, train_labels_fold, val_files, val_labels_fold)
    fold_results.append(result)
    fold += 1

# --- OVERALL RESULTS ---
print("\n" + "=" * 60)
print("  ALL FOLD RESULTS")
print("=" * 60)

accs = [r['best_acc'] for r in fold_results]
f1s = [r['final_f1'] for r in fold_results]
recalls = [r['final_recall'] for r in fold_results]
precisions = [r['final_precision'] for r in fold_results]

for r in fold_results:
    print(f"  Fold {r['fold']}: Acc={r['best_acc']:.2f}% | "
          f"P={r['final_precision']:.3f} | R={r['final_recall']:.3f} | "
          f"F1={r['final_f1']:.3f} | Epochs={r['epochs_trained']}")

print(f"\n  MEAN Accuracy:  {np.mean(accs):.2f}% ± {np.std(accs):.2f}")
print(f"  MEAN Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
print(f"  MEAN Recall:    {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
print(f"  MEAN F1-Score:  {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

# Save summary CSV
summary_path = os.path.join(LOG_DIR, "fold_summary.csv")
with open(summary_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fold_results[0].keys())
    writer.writeheader()
    writer.writerows(fold_results)
print(f"\n  Summary saved: {summary_path}")
print("\n  Training complete!")
