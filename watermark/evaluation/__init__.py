"""
Watermark Evaluation Package
"""
from watermark.evaluation.attacks import apply_attack_safe, ATTACKS, register_attack
from watermark.evaluation.metrics import compute_auc, compute_tpr_at_fpr, compute_payload_accuracy

__all__ = [
    "apply_attack_safe",
    "ATTACKS",
    "register_attack",
    "compute_auc",
    "compute_tpr_at_fpr",
    "compute_payload_accuracy",
]
