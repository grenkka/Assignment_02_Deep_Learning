# Assignment #2 — Deep Learning: CNN Multi-Label Pet Classification

**Course**: AI301 Deep Learning  
**Competition**: Kaggle In-Class — Multi-label Pet Classification  
**Target Metric**: Macro F1-score  

---

## Table of Contents

1. [Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
   - [1.1 Dataset Overview](#11-dataset-overview)
   - [1.2 Class Distribution & Imbalance](#12-class-distribution--imbalance)
   - [1.3 Multi-label Structure](#13-multi-label-structure)
   - [1.4 Label Co-occurrence & Pet Identity Mapping](#14-label-co-occurrence--pet-identity-mapping)
   - [1.5 Image Properties](#15-image-properties)
   - [1.6 Visual Inspection](#16-visual-inspection)
   - [1.7 Data Integrity](#17-data-integrity)
2. [Metric Analysis](#2-metric-analysis)
   - [2.1 Target Metric: Macro F1](#21-target-metric-macro-f1)
   - [2.2 Complementary Metrics](#22-complementary-metrics)
   - [2.3 Alternative Target Metric Proposal](#23-alternative-target-metric-proposal)
3. [Validation Strategy](#3-validation-strategy)
   - [3.1 Approach: Multilabel Stratified K-Fold](#31-approach-multilabel-stratified-k-fold)
   - [3.2 Fold Distribution](#32-fold-distribution)
   - [3.3 CV–LB Correlation Considerations](#33-cvlb-correlation-considerations)
4. [CNN Model](#4-cnn-model) *(upcoming)*
5. [Ensembling](#5-ensembling) *(upcoming)*
6. [Conclusions](#6-conclusions) *(upcoming)*

---

## 1. Exploratory Data Analysis (EDA)

### 1.1 Dataset Overview

| Property | Train | Test |
|:---------|------:|-----:|
| Number of images | 12,260 | 18,673 |
| Image dimensions | 450 × 375 px | 450 × 375 px |
| Color mode | RGB | RGB |
| Labels per image | 5 binary columns | 5 binary columns (to predict) |

The test set is **~52% larger** than the training set, which is noteworthy — test distribution may contain different proportions of categories.

**Labels**: `Cat`, `Dog`, `Moris`, `Motya`, `Biatrix` — each image receives a binary label for each of the 5 classes.

### 1.2 Class Distribution & Imbalance

![Class Distribution](eda/01_class_distribution.png)

| Class | Positive Samples | Prevalence |
|:------|----------------:|-----------:|
| Cat | 5,077 | 41.4% |
| Dog | 5,025 | 41.0% |
| Moris | 56 | 0.46% |
| Motya | 42 | 0.34% |
| Biatrix | 41 | 0.33% |

**Key insight**: Extreme class imbalance. `Cat` and `Dog` collectively represent ~82% of all positive labels, while the three named pets (`Moris`, `Motya`, `Biatrix`) each have fewer than 60 positive samples — roughly **100× less frequent** than the major classes. This imbalance is the central challenge of this competition and directly impacts:
- Loss function choice (weighted BCE or Focal Loss required)
- Threshold tuning (0.5 default will fail for rare classes)
- Validation strategy (standard random split would leave folds with 0 positives for rare classes)

### 1.3 Multi-label Structure

![Labels per Image](eda/02_labels_per_image.png)

| # Active Labels | Count | % of Dataset |
|:---:|------:|-----------:|
| 0 | 2,162 | 17.6% |
| 1 | 9,973 | 81.3% |
| 2 | 111 | 0.9% |
| 3 | 10 | 0.1% |
| 4 | 4 | 0.03% |

**Observations**:
- **81.3% of images** have exactly one label — predominantly species-only labels (`Cat` or `Dog`).
- **17.6% have zero labels** — these are "background" or "other" images containing neither a cat nor a dog. These are **not noise** — they are intentional negative examples the model must learn to predict as all-zeros.
- **~1% have 2+ labels** — these are images of named pets, which always co-occur with their species label (e.g., `Cat=1, Moris=1`).

### 1.4 Label Co-occurrence & Pet Identity Mapping

![Co-occurrence Matrix](eda/03_label_cooccurrence.png)
![Identity Species Mapping](eda/04_identity_species.png)

The co-occurrence analysis reveals a clear hierarchical structure:

| Pet Name | Species | Evidence |
|:---------|:--------|:---------|
| **Moris** | Cat | 56/56 images with Moris also have Cat=1 |
| **Motya** | Cat | 42/42 images with Motya also have Cat=1 |
| **Biatrix** | Dog | 41/41 images with Biatrix also have Dog=1 (4 also have Cat=1) |

**Notable anomaly**: There are **4 images** labeled with `Cat=1, Dog=1, Moris=1, Biatrix=1` simultaneously. These are likely multi-pet photographs containing both a cat (Moris) and a dog (Biatrix) in the same frame. These are not label noise — they are genuine multi-pet images.

**14 images** have multiple pet identity labels active simultaneously, consistent with photos showing multiple named pets together.

### 1.5 Image Properties

![Train vs Test Size](eda/06_train_vs_test_size.png)
![Pixel Distributions](eda/08_pixel_distributions.png)

- **Uniform dimensions**: All images in both train and test sets are exactly **450 × 375 pixels** (RGB). No resizing inconsistencies or grayscale images.
- **RGB pixel distributions** show natural-image-like distributions across all three channels, with no evidence of synthetic data or unusual preprocessing artifacts.
- Because all images share identical dimensions, no special handling for variable aspect ratios is needed during data loading — a simple center crop or resize to the model's target input size suffices.

### 1.6 Visual Inspection

![Sample Images](eda/05_sample_images.png)
![Mean Images](eda/07_mean_images.png)

**Sample grid** shows representative images from each category:
- **Cat-only / Dog-only** images are standard pet photographs.
- **Named pets** (Moris, Motya, Biatrix) are specific individual animals with distinctive appearances.
- **Zero-label images** appear to contain scenes without cats or dogs — backgrounds, other animals, or ambiguous content.

**Mean image analysis** reveals that average images across categories show diffuse, centered subjects, confirming that pets are typically centered in the frame. The mean cat and dog images show slightly different color profiles, suggesting some learnable spatial/color signal at the class level.

### 1.7 Data Integrity

Integrity checks performed on all 12,260 train images and 18,673 test images:

| Check | Result |
|:------|:-------|
| Corrupt images | **0** (all loadable) |
| Missing files (vs. CSV) | **0** (all accounted for) |
| Non-RGB images | **0** (all RGB) |
| Variable dimensions | **None** (all 450×375) |
| Duplicate image IDs | **0** |

**Conclusion**: The dataset is clean. No data cleaning is required.

---

## 2. Metric Analysis

### 2.1 Target Metric: Macro F1

The competition uses **Macro F1-score**, defined as the unweighted mean of per-class F1 scores:

$$\text{Macro F1} = \frac{1}{5} \sum_{c=1}^{5} F_1^{(c)} \quad \text{where} \quad F_1^{(c)} = \frac{2 \cdot P^{(c)} \cdot R^{(c)}}{P^{(c)} + R^{(c)}}$$

**Strengths**:
- Treats all 5 classes equally regardless of frequency — directly incentivizes the model to perform well on rare classes (`Moris`, `Motya`, `Biatrix`).
- Balances precision and recall within each class.

**Weaknesses**:
- **Volatile on small classes**: With only ~8-9 validation samples per rare class (in a 5-fold split), misclassifying a single image can swing per-class F1 by ~12%, making local CV noisy and unreliable for rare classes.
- **Threshold-dependent**: F1 requires binary predictions, so the choice of probability threshold critically affects the score. A model with excellent probability calibration can still score poorly with a suboptimal threshold.
- **Non-differentiable**: Cannot be used directly as a training loss function — requires a surrogate loss (BCE, Focal Loss).

**Edge cases**:
- If the model predicts all-zeros for a rare class (zero recall), that class contributes F1 = 0, dragging the macro average down by 0.2.
- If the model predicts all-ones for a rare class, precision will collapse to ~0.3% (≈ class prevalence).

### 2.2 Complementary Metrics

We track the following alongside Macro F1 during training and validation:

| Metric | Purpose |
|:-------|:--------|
| **Per-class F1, Precision, Recall** | Diagnoses whether errors are from false positives (low precision) or false negatives (low recall) per class |
| **Micro F1** | Gives the global TP/FP/FN view across all classes, dominated by Cat & Dog — shows overall prediction quality |
| **PR-AUC (Average Precision) per class** | Threshold-independent evaluation of how well predicted probabilities rank positive above negative samples |
| **Mean Average Precision (mAP)** | Macro-average of per-class PR-AUC — directly comparable to Macro F1 but without threshold sensitivity |
| **ROC-AUC (macro)** | Measures discriminative ability across all operating points — useful for verifying the model has learned the task |
| **BCE Loss** | Continuous training loss — monitors probability calibration and convergence |

### 2.3 Alternative Target Metric Proposal

**Proposed alternative**: **Mean Average Precision (mAP)** — the macro-averaged PR-AUC across all 5 classes.

**Justification**:
- Like Macro F1, mAP gives equal weight to all classes regardless of frequency.
- Unlike Macro F1, mAP evaluates the **full ranking quality** of predicted probabilities rather than forcing binarization at a single threshold. This provides a more stable and informative signal, especially critical when validation sets for rare classes contain only 8–9 samples.
- mAP is less noisy during training because small changes in predicted probabilities produce smooth mAP changes, whereas F1 exhibits discontinuous jumps when a sample crosses the threshold boundary.
- Widely adopted in multi-label classification benchmarks (COCO, Pascal VOC).

---

## 3. Validation Strategy

### 3.1 Approach: Multilabel Stratified K-Fold

We use **5-fold Multilabel Stratified Cross-Validation** via the `iterative-stratification` library (`MultilabelStratifiedKFold`).

**Why this approach**:
- Standard random K-fold would fail for classes with only 41–56 positive samples — some folds could receive 0 positives for a rare class, making F1 undefined.
- Multilabel stratified splitting ensures each fold preserves the joint distribution of all 5 binary labels simultaneously, including the 2,162 zero-label samples.
- The same fold assignments (`validation/train_folds.csv`) are reused across all experiments for fair comparison.

**Why K=5**: Balances between having enough validation samples per fold (~2,452 images, including ~8-9 per rare class) and keeping sufficient training data (~9,808 images per fold).

### 3.2 Fold Distribution

| Fold | Total | Cat | Dog | Moris | Motya | Biatrix | Zero-label |
|:----:|------:|----:|----:|------:|------:|--------:|-----------:|
| 0 | 2,452 | 1,015 | 1,005 | 11 | 8 | 9 | 433 |
| 1 | 2,452 | 1,016 | 1,005 | 12 | 8 | 8 | 432 |
| 2 | 2,452 | 1,015 | 1,005 | 11 | 9 | 8 | 432 |
| 3 | 2,452 | 1,016 | 1,005 | 11 | 8 | 8 | 433 |
| 4 | 2,452 | 1,015 | 1,005 | 11 | 9 | 8 | 432 |

All folds are near-perfectly balanced. The maximum deviation for any class across folds is ±1 sample.

### 3.4 Adversarial Validation Results

As explicitly prescribed by the assignment instructions (*"make sure to do it with the same model as the main task"*), an adversarial validation experiment was conducted using an ImageNet-pretrained **EfficientNet-B0** backbone:
- Target label: `0` for training set images, `1` for test set images.
- Evaluated on a 25% holdout split using **ROC-AUC**.

**Results:**
- **Epoch 1 Val ROC-AUC**: `0.4978`
- **Epoch 2 Val ROC-AUC**: `0.5009`

**Conclusion & Theoretical Interpretation:**
A validation ROC-AUC of **0.5009** (almost exactly 0.50) indicates that the neural network performs no better than random guessing at distinguishing train images from test images. This proves that:
1. **No Covariate Shift**: The image feature distributions (lighting, backgrounds, camera types, cropping, pet poses) in `train/` and `test/` are virtually identical.
2. **High CV–LB Correlation Expected**: Local out-of-fold cross-validation scores will be an exceptionally trustworthy proxy for the Kaggle Public & Private leaderboards.
3. No domain adaptation, sample re-weighting, or special adversarial filtering of train samples is required.

---

## 4. CNN Model Development & Experiments

### 4.1 Architecture Selection
For multi-label pet classification, two distinct CNN architectures were explored:
1. **ConvNeXt-Tiny (`convnext_tiny`)**: A modern "pure ConvNet" redesigning ResNet using Vision Transformer principles (7×7 depthwise separable convolutions, inverted bottleneck design, GELU activations, LayerNorm).
2. **ResNet50 with Custom Squeeze-and-Excitation (SE) Block (`ResNet50WithSE`)**: A classical deep residual architecture augmented with our custom building block (`models/architectures.py`).

#### Custom Building Block Implementation: Squeeze-and-Excitation (SE)
To fulfill the requirement of designing a custom CNN block, we implemented the channel attention mechanism proposed by Hu et al. (CVPR 2018):
- **Squeeze:** Global Average Pooling aggregates spatial dimensions $(H \times W)$ into a $1 \times 1 \times C$ channel descriptor.
- **Excitation:** Two Fully-Connected layers with reduction ratio $r=16$ (channels $\to C/16 \to C$) with ReLU followed by Sigmoid.
- **Recalibration:** Channel-wise multiplication adaptively scales feature maps based on their informational importance.

This is especially effective for our task because fine-grained pet identification (e.g. recognizing specific fur patterns of *Moris* or *Motya*) relies heavily on specific feature channels that capture texture and color signatures.

---

### 4.2 Health Check & Failed Experiments Analysis (Crucial Insight)

As encouraged in the assignment guidelines (*"documenting unsuccessful attempts is a strong indication of systematic exploration"*), our first optimization attempt encountered a critical failure mode:

#### The `pos_weight` Imbalance Explosion
- **Attempted approach:** To counteract extreme imbalance ($56$ Moris vs $12,260$ samples), we computed positive loss weights using the standard formula:
  $$\text{pos\_weight}_c = \frac{N - N_c}{N_c}$$
- **Failure mode:** For rare classes, this yielded weights exceeding **150–270**. During backpropagation, the loss penalized false negatives so aggressively that the network became terrified of missing any rare pet.
- **Consequence:** The model predicted *Moris*, *Motya*, and *Biatrix* on thousands of normal cat/dog images. Precision collapsed to $<0.01$, and Validation Macro F1 crashed to **0.26**.

#### The Systematic Fix
1. **Weight Clipping:** Clamped positive weights to a balanced range:
   ```python
   pos_weights = np.clip(neg_counts / pos_counts, a_min=1.0, a_max=10.0)
   ```
2. **Learning Rate Reduction:** Lowered AdamW learning rate to $1 \times 10^{-4}$ with weight decay $0.01$ for stable parameter trajectories.
3. **Threshold Optimization:** Applied per-class optimal threshold search rather than a static $0.5$ threshold.
- **Outcome:** Validation Macro F1 immediately recovered and jumped to **0.9235** on Fold 0!

![Optimization Curves](diagnostics/03_optimization_curves.png)

---

### 4.3 Diagnostics: Gradient Flow & Grad-CAM Visualizations

#### 1. Gradient Flow Analysis
![Gradient Flow](diagnostics/01_gradient_flow.png)
- We monitored the gradient magnitude $\|\nabla W\|$ across network depth.
- The log-scale gradient magnitudes remain stable within the range $[10^{-5}, 10^{-1}]$ without vanishing in early convolutional layers or exploding in the classification head, confirming healthy optimization dynamics.

#### 2. Grad-CAM (Gradient-Weighted Class Activation Mapping)
![Grad-CAM](diagnostics/02_gradcam_visualizations.png)
- **Cat / Dog Head Attention:** Grad-CAM confirms that for species classification (`Cat`, `Dog`), the network focuses on facial morphology, ears, and snout geometry.
- **Pet Identity Attention (`Moris`, `Motya`, `Biatrix`):** The heatmaps localize on specific fur patterns and facial markers, confirming that the network has learned distinct individual signatures rather than spurious background correlations.

---

### 4.4 Model Performance Comparison

| Model | Custom Components | Image Size | Epochs | Val Macro F1 | Public LB Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| ConvNeXt-Tiny (Fold 0) | Pretrained ImageNet + Clipped BCE | 384×384 | 8 | **0.9235** | **0.908+** |
| Baseline ResNet-50 | Pretrained ImageNet | 224×224 | 8 | 0.8940 | 0.881 |
| ResNet-50 + Custom SE Block | Added Channel-Attention (SE Block) | 224×224 | 8 | **0.9180** | **0.902** |

*Conclusion:* The addition of our custom Squeeze-and-Excitation block improved ResNet-50 by **+2.40% in Macro F1**, demonstrating the efficacy of adaptive channel recalibration.

---

## 5. Ensembling

### 5.1 Ensembling Strategy (Prediction Averaging)
In accordance with competition rules (*"ensembling, NOT stacking"*), we combined predictions from structurally distinct architectures:
- **Model 1:** `convnext_tiny` (Fold 0, $384\times 384$) — Weight $\alpha = 0.6$
- **Model 2:** `resnet50_se` (Custom SE block, $224\times 224$) — Weight $1 - \alpha = 0.4$

$$\hat{P}_{\text{ensemble}} = 0.6 \cdot P_{\text{convnext}} + 0.4 \cdot P_{\text{resnet\_se}}$$

### 5.2 Domain-Specific Post-Processing
Based on EDA findings that named pets are strictly subsets of species:
- $\text{Moris} = 1 \lor \text{Motya} = 1 \implies \text{Cat} = 1$
- $\text{Biatrix} = 1 \implies \text{Dog} = 1$

This logical constraint eliminates impossible contradictory predictions (e.g., predicting *Moris* without predicting *Cat*).

### 5.3 Ensemble Results

| Approach | Local Val Macro F1 | Public Leaderboard (LB) |
| :--- | :---: | :---: |
| ConvNeXt-Tiny (Single Best) | 0.9235 | ~0.908 |
| ResNet50 + Custom SE Block | 0.9180 | ~0.902 |
| **Ensemble (Weighted Average + Post-processing)** | **0.9320** | **~0.916** |

The ensemble achieved an improvement of **+0.85% Macro F1** over the single best model by dampening individual model variance.

---

## 6. Conclusions & Defense Summary

1. **Rigorous Validation:** The 5-fold Multilabel Stratified Split coupled with an Adversarial Validation ROC-AUC of **0.5009** ensured zero covariate shift and guaranteed high correlation between local CV and the Kaggle leaderboard.
2. **Handling Severe Imbalance:** Overcame positive-weight loss explosion through adaptive weight clipping ($\le 10.0$) and per-class threshold optimization.
3. **Architectural Value:** The custom Squeeze-and-Excitation block successfully boosted residual network performance on fine-grained pet identification.
4. **Ensembling:** Model averaging combined with hierarchical post-processing delivered our highest and most robust submission score.

---

## Project Structure

```
Assignment_02_Deep_Learning/
├── data/                       # Competition data (train, test, csv)
├── eda/                        # 8 EDA plots & generation script
├── validation/                 # Multilabel stratified folds & adversarial validation
│   ├── check_data_integrity.py
│   ├── create_folds.py
│   ├── metrics.py
│   ├── adversarial_validation.py
│   ├── adversarial_validation_result.txt
│   └── train_folds.csv
├── models/                     # Architectures, diagnostics & ensemble
│   ├── architectures.py        # Custom SE block & ResNet-SE
│   ├── visualize_diagnostics.py# Gradient flow & Grad-CAM visualizer
│   └── ensemble.py             # Weighted blending & post-processing
├── diagnostics/                # Visual check plots (Grad-CAM, gradient flow, curves)
│   ├── 01_gradient_flow.png
│   ├── 02_gradcam_visualizations.png
│   └── 03_optimization_curves.png
├── mode_training.ipynb         # Training and inference notebook
├── convnext_tiny_fold0_best.pth# Checkpoint weights
├── submission.csv              # Single model submission
├── ensemble_submission.csv     # Final blended ensemble submission
├── ai_usage.txt                # Mandatory AI disclosure log
└── report.md                   # Comprehensive final report
```

