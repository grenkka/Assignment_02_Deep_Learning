import os
import sys
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')
TEST_IMG_DIR = os.path.join(DATA_DIR, 'test')

LABEL_COLS = ['Cat', 'Dog', 'Moris', 'Motya', 'Biatrix']

def main():
    print("=== Step 1: Loading Data & Basic Stats ===")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples:  {len(test_df)}")

    print("\n=== Step 2: Label Anomaly & Contradiction Checks ===")
    # 1. Both Cat and Dog
    both_cat_dog = train_df[(train_df['Cat'] == 1) & (train_df['Dog'] == 1)]
    print(f"Samples with BOTH Cat=1 and Dog=1: {len(both_cat_dog)}")
    if len(both_cat_dog) > 0:
        print(both_cat_dog[['image_id'] + LABEL_COLS].head())

    # 2. Identity without species
    # Let's check associations
    for name in ['Moris', 'Motya', 'Biatrix']:
        has_name = train_df[train_df[name] == 1]
        cat_cnt = (has_name['Cat'] == 1).sum()
        dog_cnt = (has_name['Dog'] == 1).sum()
        neither_cnt = ((has_name['Cat'] == 0) & (has_name['Dog'] == 0)).sum()
        print(f"{name} (total {len(has_name)}): Cat={cat_cnt}, Dog={dog_cnt}, Neither={neither_cnt}")

    # 3. Multiple individual names at once
    multiple_names = train_df[train_df[['Moris', 'Motya', 'Biatrix']].sum(axis=1) > 1]
    print(f"Samples with multiple individual pet names: {len(multiple_names)}")

    print("\n=== Step 3: Checking Image Integrity on Disk ===")
    # Check train images
    corrupt_train = []
    train_sizes = set()
    train_modes = set()
    for img_id in train_df['image_id']:
        path = os.path.join(TRAIN_IMG_DIR, f"{img_id}.jpg")
        if not os.path.exists(path):
            corrupt_train.append((img_id, "Missing file"))
            continue
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                train_sizes.add(img.size)
                train_modes.add(img.mode)
        except Exception as e:
            corrupt_train.append((img_id, str(e)))

    print(f"Corrupt/Missing Train images: {len(corrupt_train)}")
    print(f"Train Image Modes: {train_modes}")
    print(f"Number of distinct Train Image Sizes: {len(train_sizes)}")
    sample_sizes = list(train_sizes)[:5]
    print(f"Sample Train Sizes (W x H): {sample_sizes}")

    # Quick check on test images
    corrupt_test = []
    for img_id in test_df['image_id']:
        path = os.path.join(TEST_IMG_DIR, f"{img_id}.jpg")
        if not os.path.exists(path):
            corrupt_test.append((img_id, "Missing file"))
            break
    print(f"Missing Test images check: {len(corrupt_test)} missing found")

if __name__ == '__main__':
    main()

