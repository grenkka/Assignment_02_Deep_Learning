import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, average_precision_score

sys.stdout.reconfigure(encoding='utf-8')

LABEL_COLS = ['Cat', 'Dog', 'Moris', 'Motya', 'Biatrix']

def compute_multilabel_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, thresholds=None):
    """
    Computes a comprehensive suite of multi-label classification metrics.
    
    Args:
        y_true: np.ndarray of shape (N, num_classes) with binary 0/1 values
        y_pred_probs: np.ndarray of shape (N, num_classes) with predicted probabilities [0, 1]
        thresholds: float or list/array of floats per class (defaults to 0.5)
    """
    if thresholds is None:
        thresholds = np.array([0.5] * len(LABEL_COLS))
    elif isinstance(thresholds, (int, float)):
        thresholds = np.array([thresholds] * len(LABEL_COLS))
    else:
        thresholds = np.array(thresholds)
        
    y_pred_bin = (y_pred_probs >= thresholds).astype(int)
    
    metrics = {}
    
    # Macro and Micro F1
    metrics['f1_macro'] = f1_score(y_true, y_pred_bin, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred_bin, average='micro', zero_division=0)
    
    # Per-class F1, Precision, Recall
    f1_per_class = f1_score(y_true, y_pred_bin, average=None, zero_division=0)
    prec_per_class = precision_score(y_true, y_pred_bin, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred_bin, average=None, zero_division=0)
    
    for i, col in enumerate(LABEL_COLS):
        metrics[f'{col}_f1'] = f1_per_class[i]
        metrics[f'{col}_prec'] = prec_per_class[i]
        metrics[f'{col}_rec'] = rec_per_class[i]
        
    # ROC-AUC (handling cases where a class might not appear, though with our stratified folds it will)
    try:
        metrics['roc_auc_macro'] = roc_auc_score(y_true, y_pred_probs, average='macro')
    except Exception:
        metrics['roc_auc_macro'] = np.nan
        
    # Mean Average Precision (mAP)
    try:
        metrics['map_macro'] = average_precision_score(y_true, y_pred_probs, average='macro')
    except Exception:
        metrics['map_macro'] = np.nan
        
    return metrics

def find_optimal_thresholds(y_true: np.ndarray, y_pred_probs: np.ndarray, metric='f1_macro', steps=100):
    """
    Optimizes per-class decision thresholds independently to maximize Macro F1.
    Crucial for rare classes like Moris, Motya, Biatrix!
    """
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

if __name__ == '__main__':
    # Self-test with dummy data
    N = 1000
    y_dummy = np.random.binomial(1, [0.4, 0.4, 0.005, 0.003, 0.003], size=(N, 5))
    probs_dummy = np.clip(y_dummy * 0.7 + np.random.uniform(0, 0.3, size=(N, 5)), 0, 1)
    
    base_metrics = compute_multilabel_metrics(y_dummy, probs_dummy)
    print("Baseline metrics (th=0.5): Macro F1 =", round(base_metrics['f1_macro'], 4))
    
    best_th = find_optimal_thresholds(y_dummy, probs_dummy)
    opt_metrics = compute_multilabel_metrics(y_dummy, probs_dummy, thresholds=best_th)
    print("Optimal thresholds:", np.round(best_th, 3))
    print("Optimized metrics: Macro F1 =", round(opt_metrics['f1_macro'], 4))

