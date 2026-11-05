# 🛡️ Шпаргалка до захисту — Assignment #2 (CNN)

## 📁 Структура проєкту та зв'язок між файлами

```
Assignment_02_Deep_Learning/
│
├── data/                          ← Дані Kaggle (НЕ в git, в .gitignore)
│   ├── train/                     ← 12,260 зображень (450×375, RGB)
│   ├── test/                      ← 18,673 зображень
│   ├── train.csv                  ← Мітки: Cat, Dog, Moris, Motya, Biatrix
│   ├── test.csv                   ← image_id тестових зображень
│   └── sample_submission.csv      ← Шаблон сабмішну
│
├── eda/                           ← Крок 1: Розвідувальний аналіз
│   ├── generate_eda_plots.py      ← Генерує всі 8 графіків EDA
│   ├── 01_class_distribution.png  ← Розподіл класів (дисбаланс 100×)
│   ├── 02_labels_per_image.png    ← Кількість міток на зображення
│   ├── 03_label_cooccurrence.png  ← Матриця ко-окуренції міток
│   ├── 04_identity_species.png    ← Зв'язок: Moris→Cat, Biatrix→Dog
│   ├── 05_sample_images.png       ← Приклади зображень з кожного класу
│   ├── 06_train_vs_test_size.png  ← Порівняння розмірів train/test
│   └── 08_pixel_distributions.png ← RGB-розподіл пікселів
│
├── validation/                    ← Крок 2: Валідація та спліти
│   ├── create_folds.py            ← Створює 5-fold MultilabelStratifiedKFold
│   ├── train_folds.csv            ← Результат: кожен image_id → його fold (0-4)
│   ├── check_data_integrity.py    ← Перевірка цілісності (0 corrupt, 0 missing)
│   ├── adversarial_validation.py  ← Adversarial validation: train vs test
│   ├── adversarial_validation_result.txt ← Результат: ROC-AUC = 0.5009
│   ├── metrics.py                 ← Обчислення Macro F1, mAP, ROC-AUC, per-class F1
│   └── evaluate_model.py          ← Валідація чекпоінту на Fold 0 (відтворює CV)
│
├── models/                        ← Крок 3: Архітектури та діагностика
│   ├── architectures.py           ← SqueezeAndExcitation блок + ResNet50WithSE
│   ├── visualize_diagnostics.py   ← Gradient Flow + Grad-CAM візуалізації
│   └── ensemble.py                ← Зважене усереднення + пост-обробка
│
├── diagnostics/                   ← Крок 4: Результати діагностики
│   ├── 01_gradient_flow.png       ← Норми градієнтів по шарах (немає vanishing)
│   ├── 02_gradcam_visualizations.png ← Grad-CAM: де дивиться модель
│   └── 03_optimization_curves.png ← Криві навчання (loss, F1 по епохах)
│
├── mode_training.ipynb            ← Перший ноутбук: single-fold тренування на Colab T4
├── run_ensemble_in_colab.ipynb    ← Другий ноутбук: 3-fold ансамблевий пайплайн (Colab)
├── convnext_tiny_fold0_best.pth   ← Чекпоінт ваг моделі Fold 0 (111MB, в .gitignore)
├── report.md                      ← Звіт (англійською)
├── ai_usage.txt                   ← Обов'язковий лог використання AI
├── github_url_hrynkiv.txt         ← URL GitHub-репозиторію
├── .gitignore                     ← Виключає data/, *.pth, __pycache__
└── [2026] AI301 Deep Learning - Assignment #2.pdf  ← Завдання (PDF)
```

---

## 🔄 Хронологічний порядок виконання (пайплайн)

```
1. Data Integrity     →  validation/check_data_integrity.py
   (0 corrupt, 0 missing, все 450×375 RGB)
       │
       ▼
2. EDA                →  eda/generate_eda_plots.py
   (8 графіків: дисбаланс 100×, ієрархія Moris=Cat, Biatrix=Dog)
       │
       ▼
3. Validation Split   →  validation/create_folds.py
   (5-fold MultilabelStratifiedKFold → train_folds.csv)
       │
       ▼
4. Adversarial Valid. →  validation/adversarial_validation.py
   (ROC-AUC=0.5009 → НЕМАЄ зсуву розподілу між train і test)
       │
       ▼
5. Тренування (Colab) →  mode_training.ipynb (перший запуск, single-fold)
                      →  run_ensemble_in_colab.ipynb (3-fold ансамбль)
   (ConvNeXt-Tiny, 384×384, AdamW lr=1e-4, Clipped BCE pos_weight≤10)
       │
       ▼
6. Custom Block       →  models/architectures.py
   (SE Block: squeeze→excite→recalibrate каналів)
       │
       ▼
7. Діагностика        →  models/visualize_diagnostics.py
   (Gradient Flow, Grad-CAM → diagnostics/*.png)
       │
       ▼
8. Ансамбль + Інфер.  →  models/ensemble.py + run_ensemble_in_colab.ipynb
   (avg ймовірностей 3 фолдів → пороги → пост-обробка → submission.csv)
       │
       ▼
9. Валідація          →  validation/evaluate_model.py
   (Відтворення Val Macro F1 = 0.9235 на локальному GPU)
       │
       ▼
10. Звіт              →  report.md
```

---

## 🏋️ Що саме тренували та як

### Архітектура
- **ConvNeXt-Tiny** (pretrained ImageNet) — основна модель для Kaggle сабмішнів
- **ResNet50 + Custom SE Block** — для ablation study (демонстрація кастомного блоку)
- **Baseline ResNet50** — для порівняння (без SE блоку)

### Параметри тренування (Colab)
| Параметр | Значення |
|:---------|:---------|
| Модель | `convnext_tiny` (timm pretrained) |
| Зображення | 384×384 px |
| Batch size | 64 |
| Epochs | 6 |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-2) |
| Loss | BCEWithLogitsLoss з pos_weight (clipped ≤ 10.0) |
| AMP | torch.amp.autocast + GradScaler |
| Augmentations | HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate, Resize, Normalize |

### Результати по фолдах (Colab, ConvNeXt-Tiny)

| Fold | Best Val Macro F1 | Досягнуто на епосі |
|:----:|:-----------------:|:-------------------:|
| 0 | **0.9875** | Epoch 5 |
| 1 | **0.9663** | Epoch 6 |
| 2 | **0.9975** | Epoch 4 |

### ResNet-50 (ablation study, НЕ на Kaggle)
| Модель | Val Macro F1 |
|:-------|:-----------:|
| Baseline ResNet-50 (без SE) | 0.8940 |
| ResNet-50 + Custom SE Block | **0.9180** (+2.4%) |

---

## 🏆 Kaggle Leaderboard — Історія сабмішнів

| # | Submission | Підхід | Public LB |
|:-:|:-----------|:-------|:---------:|
| 1 | `submission.csv` | Single Fold 0, пороги з CV | **0.82544** |
| 2 | `submission_ensemble_multifold.csv` | 3-fold ансамбль, пороги [0.95, 0.20, 0.85, 0.95, 0.45] | **0.86351** |
| 3 | `submission_balanced_boost.csv` | Ті ж 3 фолди, збалансовані пороги [0.70, 0.50, 0.85, 0.90, 0.45] | **0.88346** ← Best |
| 4 | `submission_topk_perfection.csv` | Top-K hard cap (88/65/65) | **0.77345** (regression!) |

**Solid baseline**: 0.90624 — ми не побили його (наш найкращий: 0.88346).

---

## 🧠 Ключові відповіді для захисту

### 1. Чому Macro F1 — складна метрика?
- Однаково важить усі 5 класів, хоча Moris/Motya/Biatrix мають лише 41-56 зразків
- Помилка на 1 зразку рідкісного класу ≈ зміна F1 цього класу на 12%
- Потрібно оптимізувати пороги для кожного класу окремо

### 2. Що таке SE Block і навіщо він потрібен?
- **Squeeze**: Global Average Pooling (H×W → 1×1×C)
- **Excitation**: 2 FC шари (C → C/16 → C) з ReLU + Sigmoid
- **Recalibrate**: Множення карт ознак на отримані ваги каналів
- Допомагає моделі "фокусуватися" на важливих каналах (текстура шерсті, колір)
- Дав **+2.4% Macro F1** порівняно з чистим ResNet-50

### 3. Чому pos_weight > 150 зламало модель?
- Формула: pos_weight = (N - N_positive) / N_positive
- Для Moris (56 з 12260): pos_weight ≈ 218
- Модель стала "панічно" боятися пропустити рідкісний клас
- Результат: тисячі false positives, Macro F1 = 0.26
- **Фікс**: np.clip(pos_weight, 1.0, 10.0) → F1 = 0.9235

### 4. Adversarial Validation — що і навіщо?
- Тренуємо модель відрізняти train від test зображень
- ROC-AUC = 0.5009 ≈ випадкове вгадування
- **Висновок**: розподіли train і test ідентичні → CV надійний proxy для LB

### 5. Чому LB < CV? (0.88 vs 0.97)
- CV оптимізує пороги на тих самих ~8-9 зразках рідкісних класів — overfitting порогів
- Тест має ~18,673 зображень — інші пропорції рідкісних класів
- Порогі, оптимальні на 8 зразках, не переносяться на 18K зображень

### 6. Чому наш скор нижче solid_baseline (0.906)?
- Baseline мав 1 сабмішн і ймовірно краще калібровані пороги
- Наша основна проблема — **threshold overfitting на малій кількості validation зразків рідкісних класів**
- З більшою кількістю сабмішнів та більш агресивним threshold search могли б покращити

### 7. Що таке ансамблювання і як ми його робили?
- **Prediction averaging**: усереднення ймовірностей з 3 незалежно тренованих ConvNeXt-Tiny (Fold 0, 1, 2)
- Зменшує варіативність предсказань (variance reduction)
- Плюс post-processing: Moris=1 → Cat=1, Biatrix=1 → Dog=1

### 8. Альтернативна метрика — mAP
- Mean Average Precision (macro-averaged PR-AUC)
- Не залежить від порогу (threshold-free)
- Більш стабільна на малих вибірках
- Стандарт для multi-label (COCO, VOC)

### 9. Grad-CAM — що показує?
- Для Cat/Dog: модель дивиться на морду, вуха, ніс
- Для Moris/Motya/Biatrix: на специфічні патерни шерсті та маркери обличчя
- Підтверджує: модель навчилася реальним ознакам, а не фоновим кореляціям

### 10. Gradient Flow — здоров'я тренування
- Норми градієнтів стабільні в діапазоні [1e-5, 1e-1]
- Немає vanishing (рання частина мережі) і exploding (класифікаційна голова)
- Оптимізація здорова
