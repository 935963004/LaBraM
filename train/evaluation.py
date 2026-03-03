from typing import Dict, List, Optional, Any

import numpy as np
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def find_best_threshold_for_f1(pred_probs: np.ndarray,
                               true_labels: np.ndarray,
                               thresholds: Optional[np.ndarray] = None,
                               labels: List[Any] = None) -> Dict[str, float]:
    """Find the threshold that maximizes the F1 score for binary classification.

    Args:
        pred_probs: Predicted probabilities, shape (N,) or (N, 1).
        true_labels: Ground-truth binary labels, shape (N,) or (N, 1).
        thresholds: Array of candidate thresholds to evaluate.
                    Defaults to ``np.arange(0.01, 1.0, 0.01)``.
        labels: List of label values for confusion matrix. Defaults to [0, 1].

    Returns:
        A dict with ``best_threshold``, ``best_f1``, ``best_accuracy``,
        and ``best_conf_matrix`` (normalized, shape 2×2).
    """
    # pred_probs = pred_probs.ravel()
    # true_labels = true_labels.ravel()

    if labels is None:
        labels = [0, 1]
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    best_f1 = -1.0
    best_threshold = 0.5
    best_accuracy = 0.0
    best_conf_matrix = None
    tn_norm, fp_norm, fn_norm, tp_norm = .0, .0, .0, .0
    tn, fp, fn, tp = 0, 0, 0, 0
    for thr in thresholds:
        preds_bin = (pred_probs >= thr).astype(float)
        f1 = f1_score(true_labels, preds_bin, zero_division=0, labels=labels)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thr)
            best_accuracy = accuracy_score(true_labels, preds_bin)
            best_conf_matrix_norm = confusion_matrix(
                true_labels.astype(int), preds_bin.astype(int), labels=labels, normalize='pred'
            )
            tn_norm, fp_norm, fn_norm, tp_norm = best_conf_matrix_norm.ravel().tolist()
            best_conf_matrix = confusion_matrix(
                true_labels.astype(int), preds_bin.astype(int), labels=labels, normalize=None
            )
            tn, fp, fn, tp  = best_conf_matrix.ravel().tolist()
    return {
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'best_accuracy': best_accuracy,
        'best_tn_norm': tn_norm,
        'best_fp_norm': fp_norm,
        'best_fn_norm': fn_norm,
        'best_tp_norm': tp_norm,
        'best_tn':tn,
        'best_fp':fp,
        'best_fn':fn,
        'best_tp':tp,
    }


def get_metrics(pred_probs, true_labels, metrics: List[str], is_binary: bool, threshold=0.5) -> Dict[str, float]:
    if is_binary:
        if 'roc_auc' not in metrics or sum(true_labels) * (
                len(true_labels) - sum(true_labels)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            results = binary_metrics_fn(
                true_labels,
                pred_probs,
                metrics=metrics,
                threshold=threshold,
            )
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
    else:
        results = multiclass_metrics_fn(
            true_labels, pred_probs, metrics=metrics
        )
    return results


def get_eval_metrics(pred_probs: np.ndarray,
                     true_labels: np.ndarray,
                     metrics: List[str],
                     is_binary: bool,
                     threshold: float = 0.5,
                     labels: List[Any] = None) -> Dict[str, float]:
    eval_metrics = get_metrics(pred_probs, true_labels, metrics, is_binary=is_binary)
    if is_binary:
        labels = [0, 1] if labels is None else labels
        prob_class_batch = (pred_probs > threshold).astype(float)
        conf_matrix = confusion_matrix(true_labels.astype(int),
                                       prob_class_batch.astype(int),labels=labels, normalize=None)
        conf_matrix_norm = confusion_matrix(true_labels.astype(int),
                                       prob_class_batch.astype(int),labels=labels, normalize='pred')
        # best_conf_matrix = confusion_matrix(
        #     true_labels.astype(int), preds_bin.astype(int), labels=labels
        # )
        tn, fp, fn, tp = conf_matrix.ravel().tolist()
        tn_norm, fp_norm, fn_norm, tp_norm = conf_matrix_norm.ravel().tolist()
        eval_metrics['tn'] = tn
        eval_metrics['fp'] = fp
        eval_metrics['fn'] = fn
        eval_metrics['tp'] = tp

        eval_metrics['tn_norm'] = tn_norm
        eval_metrics['fp_norm'] = fp_norm
        eval_metrics['fn_norm'] = fn_norm
        eval_metrics['tp_norm'] = tp_norm

        best_f1_metrics = find_best_threshold_for_f1(pred_probs, true_labels, labels=labels)
        eval_metrics.update(best_f1_metrics)
    return eval_metrics
