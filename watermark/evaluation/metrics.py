"""
Watermark Metrics Module

Implements evaluation metrics:
- Clip-level AUC
- TPR @ FPR
- Payload Accuracy

Implementation follows WATERMARK_PROJECT_PLAN.md v17, section 6.1.
"""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    
    Args:
        y_true: (N,) binary labels (0 or 1)
        y_score: (N,) prediction scores/probabilities
        
    Returns:
        AUC score
    """
    if len(np.unique(y_true)) < 2:
        return 0.5 # Undefined if only one class
    return roc_auc_score(y_true, y_score)

def compute_tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_target: float = 0.01) -> float:
    """
    Compute True Positive Rate at a specific False Positive Rate threshold.
    
    Args:
        y_true: (N,) binary labels
        y_score: (N,) prediction scores
        fpr_target: Target FPR (e.g., 0.01 for 1%)
        
    Returns:
        TPR at the threshold where FPR <= fpr_target
    """
    if len(np.unique(y_true)) < 2:
        return 0.0
        
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # Find threshold where FPR is just below/at target
    # fpr is increasing, we want max index where fpr <= target
    valid_indices = np.where(fpr <= fpr_target)[0]
    if len(valid_indices) == 0:
        return 0.0
        
    idx = valid_indices[-1]
    return tpr[idx]

def compute_payload_accuracy(
    decoded_model_ids: np.ndarray, 
    true_model_ids: np.ndarray,
    detected_mask: np.ndarray
) -> float:
    """
    Compute accuracy of payload recovery ONLY on samples correctly detected as watermarked.
    
    Args:
        decoded_model_ids: (N,) predicted model IDs
        true_model_ids: (N,) ground truth model IDs
        detected_mask: (N,) boolean mask where detector said "positive"
        
    Returns:
        Accuracy fraction (0.0 to 1.0)
    """
    # Filter only detected samples
    if detected_mask.sum() == 0:
        return 0.0
        
    correct = (decoded_model_ids[detected_mask] == true_model_ids[detected_mask])
    return correct.mean()
