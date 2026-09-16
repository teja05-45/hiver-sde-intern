"""
Classification evaluation metrics.

Implements accuracy, macro/weighted precision-recall-F1, per-intent F1,
and a confusion matrix from scratch on top of plain Python + numpy
(rather than assuming sklearn.metrics is available under a specific
version) -- also makes every formula auditable in one place, which matters
for a project whose central claim is "trust the numbers."
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_intent: dict[str, dict[str, float]]
    confusion_matrix: dict[str, dict[str, int]]
    labels: list[str]
    support: dict[str, int]
    n_examples: int

    def as_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "per_intent": {k: {m: round(v, 4) for m, v in d.items()} for k, d in self.per_intent.items()},
            "confusion_matrix": self.confusion_matrix,
            "labels": self.labels,
            "support": self.support,
            "n_examples": self.n_examples,
        }


def compute_classification_metrics(y_true: list[str], y_pred: list[str]) -> ClassificationMetrics:
    assert len(y_true) == len(y_pred), "y_true/y_pred length mismatch"
    n = len(y_true)
    labels = sorted(set(y_true) | set(y_pred))

    confusion: dict[str, dict[str, int]] = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    support = {label: sum(confusion[label].values()) for label in labels}

    per_intent: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_intent[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support[label]}

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n if n else 0.0
    macro_precision = sum(d["precision"] for d in per_intent.values()) / len(labels) if labels else 0.0
    macro_recall = sum(d["recall"] for d in per_intent.values()) / len(labels) if labels else 0.0
    macro_f1 = sum(d["f1"] for d in per_intent.values()) / len(labels) if labels else 0.0
    weighted_f1 = (
        sum(d["f1"] * support[label] for label, d in per_intent.items()) / n if n else 0.0
    )

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_intent=per_intent,
        confusion_matrix=confusion,
        labels=labels,
        support=support,
        n_examples=n,
    )


def rare_intent_f1(metrics: ClassificationMetrics, rarity_threshold: int) -> dict:
    """Average F1 restricted to intents with training support below a threshold.

    Reported separately per the assignment's explicit requirement for
    "rare-intent F1" -- macro F1 can look fine while the system is
    actually unreliable on infrequent-but-important intents (e.g.
    payment_or_billing_issue in this dataset).
    """
    rare = {k: v for k, v in metrics.per_intent.items() if v["support"] < rarity_threshold}
    if not rare:
        return {"rare_intents": [], "avg_f1": None, "threshold": rarity_threshold}
    avg_f1 = sum(v["f1"] for v in rare.values()) / len(rare)
    return {
        "rare_intents": list(rare.keys()),
        "avg_f1": round(avg_f1, 4),
        "per_intent": {k: round(v["f1"], 4) for k, v in rare.items()},
        "threshold": rarity_threshold,
    }
