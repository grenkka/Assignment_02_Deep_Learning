import os
import sys
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
VAL_DIR = os.path.join(BASE_DIR, 'validation')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
OUTPUT_SPLIT_CSV = os.path.join(VAL_DIR, 'train_folds.csv')

LABEL_COLS = ['Cat', 'Dog', 'Moris', 'Motya', 'Biatrix']

def make_folds(n_splits=5, seed=42):
    print(f"Creating {n_splits}-fold Multilabel Stratified Split (Seed={seed})...")
    df = pd.read_csv(TRAIN_CSV)
    
    # Initialize MultilabelStratifiedKFold
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    X = df['image_id'].values
    y = df[LABEL_COLS].values
    
    df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(mskf.split(X, y)):
        df.loc[val_idx, 'fold'] = fold
        
    # Save folds to CSV
    df.to_csv(OUTPUT_SPLIT_CSV, index=False)
    print(f"Folds saved to: {OUTPUT_SPLIT_CSV}")
    
    print("\n--- Distribution of Labels Across Folds ---")
    summary = []
    for f in range(n_splits):
        sub = df[df['fold'] == f]
        row = {'fold': f, 'total_samples': len(sub)}
        for c in LABEL_COLS:
            row[c] = sub[c].sum()
        row['zero_labels'] = (sub[LABEL_COLS].sum(axis=1) == 0).sum()
        summary.append(row)
        
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

if __name__ == '__main__':
    make_folds(n_splits=5, seed=42)

