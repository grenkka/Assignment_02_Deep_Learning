"""
Full Validation Script matching mode_training.ipynb:
- Image resolution 384x384 (identical to training CFG.img_size)
- Re-optimizes thresholds dynamically on out-of-fold predictions
"""
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms
from sklearn.metrics import f1_score, precision_score, recall_score

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
VAL_DIR = os.path.join(BASE_DIR, 'validation')
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')
FOLDS_CSV = os.path.join(VAL_DIR, 'train_folds.csv')
WEIGHTS_PATH = os.path.join(BASE_DIR, 'convnext_tiny_fold0_best.pth')

LABEL_COLS = ['Cat', 'Dog', 'Moris', 'Motya', 'Biatrix']

class PetValDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['image_id']}.jpg")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = row[LABEL_COLS].values.astype(np.float32)
        return image, torch.tensor(labels)

def find_optimal_thresholds(y_true, y_pred_probs, steps=100):
    num_classes = y_true.shape[1]
    best_thresholds = np.full(num_classes, 0.5)
    threshold_candidates = np.linspace(0.01, 0.99, steps)
    for c in range(num_classes):
        best_c_th = 0.5
        best_c_f1 = -1.0
        for th in threshold_candidates:
            preds_c = (y_pred_probs[:, c] >= th).astype(int)
            score = f1_score(y_true[:, c], preds_c, zero_division=0)
            if score > best_c_f1:
                best_c_f1 = score
                best_c_th = th
        best_thresholds[c] = best_c_th
    return best_thresholds

def apply_postprocessing(preds):
    preds[:, 0] = np.logical_or(preds[:, 0], preds[:, 2]).astype(int)
    preds[:, 0] = np.logical_or(preds[:, 0], preds[:, 3]).astype(int)
    preds[:, 1] = np.logical_or(preds[:, 1], preds[:, 4]).astype(int)
    return preds

def run_evaluation(img_size=384, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Eval resolution: {img_size}x{img_size}")

    df = pd.read_csv(FOLDS_CSV)
    val_df = df[df['fold'] == 0].reset_index(drop=True)

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_loader = DataLoader(PetValDataset(val_df, TRAIN_IMG_DIR, val_transforms), 
                            batch_size=batch_size, shuffle=False, num_workers=0)

    model = timm.create_model('convnext_tiny', pretrained=False, num_classes=5)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    # Threshold search
    best_th = find_optimal_thresholds(all_targets, all_probs)
    val_preds_bin = (all_probs >= best_th).astype(int)
    val_preds_bin = apply_postprocessing(val_preds_bin)

    macro_f1 = f1_score(all_targets, val_preds_bin, average='macro', zero_division=0)
    f1_per_class = f1_score(all_targets, val_preds_bin, average=None, zero_division=0)
    prec_per_class = precision_score(all_targets, val_preds_bin, average=None, zero_division=0)
    rec_per_class = recall_score(all_targets, val_preds_bin, average=None, zero_division=0)

    print("\n" + "="*55)
    print(f"  EXACT EVALUATION (Resolution {img_size}x{img_size})")
    print(f"  Overall Validation Macro F1: {macro_f1:.4f}")
    print(f"  Optimal Thresholds: {np.round(best_th, 3)}")
    print("="*55)
    print(f"{'Class':<10} {'F1-Score':<10} {'Precision':<10} {'Recall':<10} {'Support'}")
    print("-" * 55)
    for i, col in enumerate(LABEL_COLS):
        support = int(all_targets[:, i].sum())
        print(f"{col:<10} {f1_per_class[i]:<10.4f} {prec_per_class[i]:<10.4f} {rec_per_class[i]:<10.4f} {support}")

if __name__ == '__main__':
    run_evaluation(img_size=384)
