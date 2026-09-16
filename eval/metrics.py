"""Metrics for all three sub-systems: intent classification, reply quality
(judge + human agreement), escalation policy."""
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, cohen_kappa_score, precision_recall_fscore_support,
)

# cost-weighted escalation eval: a missed safety issue is far worse than a
# missed ordinary one, which is worse than an unnecessary escalation.
# Stated explicitly here, not buried - this is the number that should
# actually drive the auto-handle threshold, not raw accuracy.
COST_FALSE_AUTO_HANDLE_SAFETY = 10
COST_FALSE_AUTO_HANDLE_OTHER = 3
COST_FALSE_ESCALATE = 1


def intent_metrics(y_true, y_pred, labels):
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {"report": report, "confusion_matrix": cm.tolist(), "labels": labels}


def judge_human_agreement(human_scores, judge_scores, dims=("grounded", "tone", "factual", "actionable")):
    """Both args: list of dicts with the same dims, same order, same items.
    Weighted (linear) kappa per dimension - these are 1-5 ordinal scores, an
    unweighted kappa would punish a judge that's off by one point as hard as
    one that's off by four."""
    out = {}
    for dim in dims:
        h = [s[dim] for s in human_scores if s[dim] is not None]
        j = [s[dim] for s in judge_scores if s[dim] is not None]
        n = min(len(h), len(j))
        if n < 2:
            out[dim] = None
            continue
        out[dim] = round(cohen_kappa_score(h[:n], j[:n], weights="linear"), 3)
    return out


def escalation_metrics(true_escalate, pred_escalate, true_reason):
    p, r, f1, _ = precision_recall_fscore_support(
        true_escalate, pred_escalate, average="binary", zero_division=0
    )

    cost = 0
    for t, p_, reason in zip(true_escalate, pred_escalate, true_reason):
        if t and not p_:  # missed an escalation
            is_safety = "safety" in str(reason).lower() or "harm" in str(reason).lower()
            cost += COST_FALSE_AUTO_HANDLE_SAFETY if is_safety else COST_FALSE_AUTO_HANDLE_OTHER
        elif not t and p_:  # unnecessary escalation
            cost += COST_FALSE_ESCALATE

    return {
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f1, 3),
        "total_cost": cost,
        "avg_cost_per_example": round(cost / max(len(true_escalate), 1), 3),
    }
