from typing import Dict

from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn


def get_metrics(output, target, metrics, is_binary, threshold=0.5) -> Dict[str, float]:
    if is_binary:
        if 'roc_auc' not in metrics or sum(target) * (
                len(target) - sum(target)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            results = binary_metrics_fn(
                target,
                output,
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
            target, output, metrics=metrics
        )
    return results
