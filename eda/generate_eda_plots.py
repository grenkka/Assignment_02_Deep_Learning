"""
EDA Visualizations for Assignment #2 — Multi-label Pet Classification
Generates all EDA plots and saves them into the eda/ folder.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
EDA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')
TEST_IMG_DIR = os.path.join(DATA_DIR, 'test')

LABEL_COLS = ['Cat', 'Dog', 'Moris', 'Motya', 'Biatrix']
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

df = pd.read_csv(TRAIN_CSV)

# ──────────────────────────────────────────────────────────────────────────────
# 1. CLASS DISTRIBUTION  (bar chart)
# ──────────────────────────────────────────────────────────────────────────────
def plot_class_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute counts
    counts = df[LABEL_COLS].sum()
    axes[0].bar(LABEL_COLS, counts, color=COLORS)
    axes[0].set_title('Positive Samples per Class', fontsize=13)
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts):
        axes[0].text(i, v + 50, str(int(v)), ha='center', fontweight='bold')

    # Prevalence (fraction)
    prevalence = df[LABEL_COLS].mean()
    axes[1].bar(LABEL_COLS, prevalence * 100, color=COLORS)
    axes[1].set_title('Class Prevalence (%)', fontsize=13)
    axes[1].set_ylabel('% of training set')
    for i, v in enumerate(prevalence * 100):
        axes[1].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '01_class_distribution.png'), dpi=150)
    plt.close()
    print("[✓] 01_class_distribution.png")


# ──────────────────────────────────────────────────────────────────────────────
# 2. LABELS PER IMAGE  (histogram)
# ──────────────────────────────────────────────────────────────────────────────
def plot_labels_per_image():
    n_labels = df[LABEL_COLS].sum(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = n_labels.value_counts().sort_index()
    ax.bar(vals.index.astype(str), vals.values, color='#4C72B0')
    ax.set_title('Number of Active Labels per Image', fontsize=13)
    ax.set_xlabel('# labels')
    ax.set_ylabel('# images')
    for i, (k, v) in enumerate(vals.items()):
        pct = v / len(df) * 100
        ax.text(i, v + 30, f'{v}\n({pct:.1f}%)', ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '02_labels_per_image.png'), dpi=150)
    plt.close()
    print("[✓] 02_labels_per_image.png")


# ──────────────────────────────────────────────────────────────────────────────
# 3. LABEL CO-OCCURRENCE MATRIX  (heatmap)
# ──────────────────────────────────────────────────────────────────────────────
def plot_cooccurrence():
    cooc = df[LABEL_COLS].T.dot(df[LABEL_COLS])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cooc, annot=True, fmt='d', cmap='YlOrRd', square=True,
                xticklabels=LABEL_COLS, yticklabels=LABEL_COLS, ax=ax,
                linewidths=0.5, linecolor='white')
    ax.set_title('Label Co-occurrence Matrix', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '03_label_cooccurrence.png'), dpi=150)
    plt.close()
    print("[✓] 03_label_cooccurrence.png")


# ──────────────────────────────────────────────────────────────────────────────
# 4. PET IDENTITY → SPECIES MAPPING  (stacked bar)
# ──────────────────────────────────────────────────────────────────────────────
def plot_identity_species():
    fig, ax = plt.subplots(figsize=(8, 5))
    pets = ['Moris', 'Motya', 'Biatrix']
    cat_counts = []
    dog_counts = []
    both_counts = []
    for p in pets:
        sub = df[df[p] == 1]
        both = ((sub['Cat'] == 1) & (sub['Dog'] == 1)).sum()
        cat_only = ((sub['Cat'] == 1) & (sub['Dog'] == 0)).sum()
        dog_only = ((sub['Dog'] == 1) & (sub['Cat'] == 0)).sum()
        cat_counts.append(cat_only)
        dog_counts.append(dog_only)
        both_counts.append(both)

    x = np.arange(len(pets))
    w = 0.5
    ax.bar(x, cat_counts, w, label='Cat only', color='#4C72B0')
    ax.bar(x, dog_counts, w, bottom=cat_counts, label='Dog only', color='#DD8452')
    ax.bar(x, both_counts, w, bottom=[c+d for c,d in zip(cat_counts, dog_counts)],
           label='Both Cat & Dog', color='#C44E52')
    ax.set_xticks(x)
    ax.set_xticklabels(pets)
    ax.set_title('Pet Identity → Species Association', fontsize=13)
    ax.set_ylabel('Count')
    ax.legend()

    # Annotate totals
    for i, p in enumerate(pets):
        total = cat_counts[i] + dog_counts[i] + both_counts[i]
        ax.text(i, total + 0.5, f'n={total}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '04_identity_species.png'), dpi=150)
    plt.close()
    print("[✓] 04_identity_species.png")


# ──────────────────────────────────────────────────────────────────────────────
# 5. SAMPLE IMAGES GRID  (4 rows: Cat, Dog, Moris/Motya/Biatrix, Zero-label)
# ──────────────────────────────────────────────────────────────────────────────
def plot_sample_images():
    random.seed(42)
    categories = {
        'Cat (only)': df[(df['Cat']==1) & (df['Dog']==0) & (df[['Moris','Motya','Biatrix']].sum(axis=1)==0)],
        'Dog (only)': df[(df['Dog']==1) & (df['Cat']==0) & (df[['Moris','Motya','Biatrix']].sum(axis=1)==0)],
        'Named pets (Moris/Motya/Biatrix)': df[df[['Moris','Motya','Biatrix']].sum(axis=1) > 0],
        'Zero-label (background)': df[df[LABEL_COLS].sum(axis=1) == 0],
    }

    n_cols = 5
    n_rows = len(categories)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

    for row_idx, (cat_name, sub_df) in enumerate(categories.items()):
        sample = sub_df.sample(n=min(n_cols, len(sub_df)), random_state=42)
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if col_idx < len(sample):
                img_id = sample.iloc[col_idx]['image_id']
                img_path = os.path.join(TRAIN_IMG_DIR, f'{img_id}.jpg')
                img = Image.open(img_path)
                ax.imshow(img)
                # Show labels beneath image
                labels = [c for c in LABEL_COLS if sample.iloc[col_idx][c] == 1]
                label_str = ', '.join(labels) if labels else '(none)'
                ax.set_xlabel(label_str, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(cat_name, fontsize=10, fontweight='bold')

    plt.suptitle('Sample Images by Category', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '05_sample_images.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[✓] 05_sample_images.png")


# ──────────────────────────────────────────────────────────────────────────────
# 6. TRAIN vs TEST SIZE COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
def plot_train_vs_test_size():
    train_count = len(df)
    test_count = len(os.listdir(TEST_IMG_DIR))
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(['Train', 'Test'], [train_count, test_count], color=['#4C72B0', '#DD8452'])
    ax.set_title('Dataset Size: Train vs Test', fontsize=13)
    ax.set_ylabel('Number of images')
    for bar, v in zip(bars, [train_count, test_count]):
        ax.text(bar.get_x() + bar.get_width()/2, v + 200, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '06_train_vs_test_size.png'), dpi=150)
    plt.close()
    print("[✓] 06_train_vs_test_size.png")


# ──────────────────────────────────────────────────────────────────────────────
# 7. MEAN IMAGE per category (visual fingerprint)
# ──────────────────────────────────────────────────────────────────────────────
def plot_mean_images():
    categories = {
        'Cat': df[(df['Cat']==1) & (df['Dog']==0)],
        'Dog': df[(df['Dog']==1) & (df['Cat']==0)],
        'Zero-label': df[df[LABEL_COLS].sum(axis=1) == 0],
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for idx, (name, sub_df) in enumerate(categories.items()):
        # Sample up to 200 for speed
        sample = sub_df.sample(n=min(200, len(sub_df)), random_state=42)
        acc = np.zeros((375, 450, 3), dtype=np.float64)
        count = 0
        for _, row in sample.iterrows():
            img_path = os.path.join(TRAIN_IMG_DIR, f"{row['image_id']}.jpg")
            img = np.array(Image.open(img_path).resize((450, 375)))
            acc += img.astype(np.float64)
            count += 1
        mean_img = (acc / count).astype(np.uint8)
        axes[idx].imshow(mean_img)
        axes[idx].set_title(f'Mean Image: {name}\n(n={count})', fontsize=11)
        axes[idx].axis('off')

    plt.suptitle('Average Image per Category', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '07_mean_images.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[✓] 07_mean_images.png")


# ──────────────────────────────────────────────────────────────────────────────
# 8. PIXEL INTENSITY DISTRIBUTION per channel (train sample)
# ──────────────────────────────────────────────────────────────────────────────
def plot_pixel_distributions():
    sample = df.sample(n=300, random_state=42)
    r_vals, g_vals, b_vals = [], [], []
    for _, row in sample.iterrows():
        img = np.array(Image.open(os.path.join(TRAIN_IMG_DIR, f"{row['image_id']}.jpg")))
        r_vals.extend(img[:,:,0].flatten()[::50])  # subsample for speed
        g_vals.extend(img[:,:,1].flatten()[::50])
        b_vals.extend(img[:,:,2].flatten()[::50])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(r_vals, bins=50, alpha=0.5, color='red', label='R', density=True)
    ax.hist(g_vals, bins=50, alpha=0.5, color='green', label='G', density=True)
    ax.hist(b_vals, bins=50, alpha=0.5, color='blue', label='B', density=True)
    ax.set_title('RGB Pixel Intensity Distribution (300 sample images)', fontsize=13)
    ax.set_xlabel('Pixel value')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '08_pixel_distributions.png'), dpi=150)
    plt.close()
    print("[✓] 08_pixel_distributions.png")


# ──────────────────────────────────────────────────────────────────────────────
# RUN ALL
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating EDA visualizations...\n")
    plot_class_distribution()
    plot_labels_per_image()
    plot_cooccurrence()
    plot_identity_species()
    plot_sample_images()
    plot_train_vs_test_size()
    plot_mean_images()
    plot_pixel_distributions()
    print(f"\nAll plots saved to: {EDA_DIR}")

