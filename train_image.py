"""
Optimized Dermatology Disorder Detection (PyTorch Version - Shape Fixed)

Key Fixes:
- T.Resize((224, 224)) : Fixed size resize to ensure uniform 224x224 tensors (no shape mismatch)
- Removed CenterCrop from test (unnecessary with fixed resize)
- Simplified robust_collate (no longer needed with fixed shapes, but kept for safety)
- Added try-except in __getitem__ to skip/flag bad images during loading

This ensures all tensors are exactly [B, 3, 224, 224]. Run now—should complete epoch in ~2-4 min.

Author: Jahanzaib Farooq (Improved by Grok)
Date: October 2025
"""

import os
import cv2
import time
import numpy as np
import pandas as pd
import seaborn as sns
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from fvcore.nn.precise_bn import update_bn_stats  # Optional

import argparse
import sys
import datetime
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import torch.distributed as dist
import torch.backends.cudnn as cudnn

# Speed Optimizations
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

# Config
DIR = "/content/ProjectFlask_internship_Assignment/dermatology_dataset"
DIR_TRAIN = "/content/ProjectFlask_internship_Assignment/dermatology_dataset/train/"
DIR_TEST = "/content/ProjectFlask_internship_Assignment/dermatology_dataset/test/"
BATCH_SIZE = 32
NUM_EPOCHS = 4  # Test; set to 50 for full
LEARNING_RATE = 1e-4
NUM_WORKERS = 2  # Safe start; set to 2 after test
PIN_MEMORY = False  # With workers=0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# AMP Scaler (Fixed)
scaler = torch.amp.GradScaler('cuda')

# Exploring Dataset (Same)
classes = sorted(os.listdir(DIR_TRAIN))
print("Total Classes: ", len(classes))

train_count = 0
test_count = 0
classes_df = []
for _class in classes:
    class_dict = {}
    train_count += len(os.listdir(DIR_TRAIN + _class))
    test_count += len(os.listdir(DIR_TEST + _class))
    class_dict.update({'Class': _class, 'Train': len(os.listdir(DIR_TRAIN + _class)), 'Test': len(os.listdir(DIR_TEST + _class))})
    classes_df.append(class_dict)

print("Total train images: ", train_count)
print("Total test images: ", test_count)
print(pd.DataFrame(classes_df))

# Image lists and mappings (Same)
train_imgs = []
test_imgs = []
for _class in classes:
    for img in os.listdir(DIR_TRAIN + _class):
        train_imgs.append(DIR_TRAIN + _class + "/" + img)
    for img in os.listdir(DIR_TEST + _class):
        test_imgs.append(DIR_TEST + _class + "/" + img)

class_to_int = {classes[i]: i for i in range(len(classes))}
int_to_class = dict(map(reversed, class_to_int.items()))

# Fixed Transforms: Resize to exact (224, 224) for uniform shapes
def robust_transform(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

train_transform = T.Compose([
    T.Lambda(robust_transform),
    T.RandomRotation(5),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    T.RandomHorizontalFlip(p=0.5),
    T.Resize((224, 224)),  # Fixed size: ensures [3, 224, 224]
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = T.Compose([
    T.Lambda(robust_transform),
    T.Resize((224, 224)),  # Fixed size, no crop needed
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset: Skip bad images
class RobustImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target
        except Exception as e:
            print(f"Skipping bad image at index {index} ({self.samples[index][0]}): {e}")
            # Dummy: black image + original target (or random; here keep target)
            dummy_img = torch.zeros(3, 224, 224)
            return dummy_img, target  # Use original target to avoid bias

# Datasets
train_dataset = RobustImageFolder(root=DIR_TRAIN, transform=train_transform)
test_dataset_full = RobustImageFolder(root=DIR_TEST, transform=test_transform)
test_size = int(0.5 * len(test_dataset_full))
valid_size = len(test_dataset_full) - test_size
valid_dataset, test_dataset = torch.utils.data.random_split(test_dataset_full, [valid_size, test_size])

# Class weights + Sampler (Same)
class_counts = np.bincount(train_dataset.targets)
class_weights = 1. / class_counts
sample_weights = [class_weights[label] for label in train_dataset.targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# Collate (Simplified, as shapes now uniform)
def robust_collate(batch):
    try:
        return torch.utils.data.dataloader.default_collate(batch)
    except RuntimeError as e:
        if "stack expects each tensor" in str(e):
            print("Warning: Shape mismatch in batch; using padded dummy")
            # Pad to max height/width or dummy
            max_h = max(img.shape[1] for img, _ in batch)
            max_w = max(img.shape[2] for img, _ in batch)
            padded = []
            targets = []
            for img, tgt in batch:
                pad_h = max_h - img.shape[1]
                pad_w = max_w - img.shape[2]
                padded_img = F.pad(img, (0, pad_w, 0, pad_h), value=0)
                padded.append(padded_img)
                targets.append(tgt)
            return torch.stack(padded), torch.stack(targets)
        raise e

# Dataloaders
dataloaders_dict = {
    'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, 
                        pin_memory=PIN_MEMORY, collate_fn=robust_collate),
    'val': DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, 
                      pin_memory=PIN_MEMORY, drop_last=False, collate_fn=robust_collate)
}
dataloader_test = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, 
                             pin_memory=PIN_MEMORY, drop_last=False, collate_fn=robust_collate)

# Visualize (Same)
src_folder = DIR_TRAIN
fig = plt.figure(figsize=(15, 35))
image_num = 0
num_folders = len(os.listdir(src_folder))
for root, folders, filenames in os.walk(src_folder):
    for folder in sorted(folders):
        image_num += 1
        file_name = os.listdir(os.path.join(root, folder))[0]
        file_path = os.path.join(root, folder, file_name)
        image = mpimg.imread(file_path)
        a = fig.add_subplot(num_folders, 3, image_num)
        plt.imshow(image)
        a.set_title(folder)
fig.subplots_adjust(hspace=1, wspace=1)
plt.show()

# Focal Loss (Same)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# Model (Same)
num_classes = 23
model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(DEVICE)

# Optimizer + Criterion + Scheduler (Same)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
criterion = FocalLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# Train Function (Same as previous)
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS, patience=7):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    val_acc_history = []
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        epoch_start = time.time()

        for phase in ['train', 'val']:
            phase_start = time.time()
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            batch_count = 0

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                batch_start = time.time()
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                batch_count += 1

                if batch_idx % 50 == 0 and batch_idx > 0:
                    batch_time = time.time() - batch_start
                    print(f'  {phase} Batch {batch_idx}/{len(dataloaders[phase])}: {batch_time:.2f}s/step')

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            phase_time = time.time() - phase_start
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} (Time: {phase_time:.1f}s)')

            if phase == 'val':
                val_acc_history.append(epoch_acc)
                prev_lr = scheduler.get_last_lr()
                scheduler.step(epoch_acc)
                new_lr = scheduler.get_last_lr()
                if new_lr[0] < prev_lr[0]:
                    print(f"LR reduced from {prev_lr[0]:.6f} to {new_lr[0]:.6f}")

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        epoch_time = time.time() - epoch_start
        print(f'Epoch time: {epoch_time:.1f}s')

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, val_acc_history

# Test function (Same)
def test_model(model, dl):
    model.eval()
    true_labels = []
    predictions = []
    total = 0
    num_correct = 0
    with torch.no_grad():
        for images, labels in dl:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            predicted = torch.argmax(outputs.data, -1)
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            total += labels.size(0)
            num_correct += (predicted == labels).sum()
    print(f"Test Accuracy: {float(num_correct)/float(total)*100:.2f}%")
    print(classification_report(true_labels, predictions, target_names=classes))
    return true_labels, predictions

# Visualization function (Same)
def visualize_predictions(model, dl, true_class_idx, pred_class_idx, num_samples=5):
    model.eval()
    true_labels = []
    predictions = []
    images_list = []
    with torch.no_grad():
        for images, labels in dl:
            images_list.append(images)
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            predicted = torch.argmax(outputs.data, -1)
            true_labels.append(labels.cpu().numpy())
            predictions.append(predicted.cpu().numpy())

    found = 0
    for batch_idx in range(len(true_labels)):
        for i in range(len(true_labels[batch_idx])):
            if true_labels[batch_idx][i] == true_class_idx and predictions[batch_idx][i] == pred_class_idx and found < num_samples:
                img = images_list[batch_idx][i].cpu().permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                plt.figure()
                plt.imshow(img)
                plt.title(f'True: {int_to_class[true_class_idx]}, Pred: {int_to_class[pred_class_idx]}')
                plt.show()
                found += 1
                if found >= num_samples:
                    return

# Train
print("🚀 Training EfficientNetV2-S (Shape Fixed)...")
model, hist = train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)

# Enhanced Saving
checkpoint = {
    'epoch': NUM_EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_acc': max(hist) if hist else 0.0,
    'class_to_int': class_to_int,
    'int_to_class': int_to_class,
    'transforms': str(train_transform),
    'classes': classes
}
torch.save(checkpoint, 'best_derma_model_efficientnetv2s.pth')
print("✅ Model checkpoint saved as 'best_derma_model_efficientnetv2s.pth'")

# Evaluate
print("\n🎯 Evaluating...")
true_labels, predictions = test_model(model, dataloader_test)

# Confusion Matrix
c_matrix = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(12, 12))
plt.title("Confusion Matrix")
sns.heatmap(c_matrix, cmap='Blues', annot=True, xticklabels=classes, yticklabels=classes, fmt='g')
plt.xlabel('Predictions')
plt.ylabel('True Labels')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Examples
eczema_idx = class_to_int.get('Eczema Photos', 5)
psoriasis_idx = class_to_int.get('Psoriasis pictures Lichen Planus and related diseases', 14)
visualize_predictions(model, dataloader_test, eczema_idx, psoriasis_idx)  # False
visualize_predictions(model, dataloader_test, eczema_idx, eczema_idx)     # Correct

print("\n✅ Done! Shapes now uniform—expect stable run. For full, set NUM_EPOCHS=50.")
