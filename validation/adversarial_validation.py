"""
Adversarial Validation:
Train vs Test distribution similarity check using a CNN backbone.
Optimized for 4GB VRAM (GTX 1650):
- Batch size = 32
- Image size = 224x224
- Mixed precision (torch.amp.autocast)
- Pretrained MobileNetV3 / EfficientNet-B0
"""
import os
import sys
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import timm

sys.stdout.reconfigure(encoding='utf-8')

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
VAL_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')
TEST_IMG_DIR = os.path.join(DATA_DIR, 'test')

class AdversarialDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        label = row['is_test']
        
        with Image.open(img_path) as img:
            img = img.convert('RGB').resize((224, 224))
            
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        tensor = torch.tensor(arr).permute(2, 0, 1) # (C, H, W)
        
        return tensor, torch.tensor(label, dtype=torch.float32)

def run_adversarial_validation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    train_files = [os.path.join(TRAIN_IMG_DIR, f) for f in os.listdir(TRAIN_IMG_DIR) if f.endswith('.jpg')]
    test_files = [os.path.join(TEST_IMG_DIR, f) for f in os.listdir(TEST_IMG_DIR) if f.endswith('.jpg')]
    
    print(f"Total train images: {len(train_files)}")
    print(f"Total test images:  {len(test_files)}")
    
    # Subsample 2500 train and 2500 test for fast convergence and safe memory usage
    N_SAMPLE = 2500
    train_sample = random.sample(train_files, min(N_SAMPLE, len(train_files)))
    test_sample = random.sample(test_files, min(N_SAMPLE, len(test_files)))
    
    records = []
    for p in train_sample:
        records.append({'path': p, 'is_test': 0})
    for p in test_sample:
        records.append({'path': p, 'is_test': 1})
        
    df = pd.DataFrame(records)
    
    train_df, val_df = train_test_split(df, test_size=0.25, random_state=SEED, stratify=df['is_test'])
    print(f"Adv Train: {len(train_df)}, Adv Val: {len(val_df)}")
    
    # Use batch_size 32 and num_workers 0 for Windows stability
    train_loader = DataLoader(AdversarialDataset(train_df), batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(AdversarialDataset(val_df), batch_size=32, shuffle=False, num_workers=0)
    
    print("Loading pretrained backbone: efficientnet_b0...")
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1)
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    epochs = 2
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(x).squeeze(-1)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x).squeeze(-1)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item() * len(y)
            
        train_loss = total_loss / len(train_df)
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                if scaler:
                    with torch.amp.autocast('cuda'):
                        logits = model(x).squeeze(-1)
                else:
                    logits = model(x).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(y.numpy())
                
        val_auc = roc_auc_score(val_targets, val_preds)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Validation ROC-AUC: {val_auc:.4f}")
        
    print("\n=== Result Interpretation ===")
    if val_auc < 0.60:
        conclusion = f"ROC-AUC = {val_auc:.4f} (~0.50). Train and Test sets have virtually IDENTICAL distributions! Local CV will strongly correlate with Public LB."
    elif val_auc < 0.75:
        conclusion = f"ROC-AUC = {val_auc:.4f}. Slight domain shift detected between Train and Test."
    else:
        conclusion = f"ROC-AUC = {val_auc:.4f}. Significant distribution shift detected between Train and Test."
    print(conclusion)

    out_file = os.path.join(VAL_DIR, 'adversarial_validation_result.txt')
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(f"Model: efficientnet_b0 (pretrained ImageNet)\n")
        f.write(f"Adversarial Validation ROC-AUC: {val_auc:.4f}\n")
        f.write(f"Conclusion: {conclusion}\n")
    print(f"Result saved to {out_file}")

if __name__ == '__main__':
    run_adversarial_validation()
