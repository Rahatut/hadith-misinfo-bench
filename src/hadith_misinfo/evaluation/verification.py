"""Verification evaluation metrics (Strict/Selective Accuracy, Macro-F1, Abstention Rate, Δ_BN)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from hadith_misinfo.schemas import GroundTruthLabel, PredictionLabel, VerificationResult
from hadith_misinfo.verification.policies import map_verdict_to_prediction


@dataclass
class VerificationMetrics:
    system: str
    language: str
    n: int
    strict_accuracy: float
    coverage: float
    selective_accuracy: float
    macro_f1: float
    abstention_rate: float
    per_class_f1: dict[str, float] = field(default_factory=dict)
    verdict_counts: dict[str, int] = field(default_factory=dict)


def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def compute_verification_metrics(
    gold_labels: dict[str, GroundTruthLabel],
    predictions: dict[str, PredictionLabel],
    system: str = "",
    language: str = "",
) -> VerificationMetrics:
    """Compute verification evaluation metrics."""
    claim_ids = list(gold_labels.keys())
    n = len(claim_ids)
    if n == 0:
        return VerificationMetrics(
            system=system, language=language, n=0,
            strict_accuracy=0.0, coverage=0.0, selective_accuracy=0.0,
            macro_f1=0.0, abstention_rate=0.0,
        )

    n_abstain = sum(1 for cid in claim_ids if predictions.get(cid) == "abstain")
    n_correct_strict = sum(
        1 for cid in claim_ids if predictions.get(cid) == gold_labels[cid]
    )

    decided = [cid for cid in claim_ids if predictions.get(cid) != "abstain"]
    n_correct_decided = sum(1 for cid in decided if predictions[cid] == gold_labels[cid])

    per_class_f1: dict[str, float] = {}
    for cls in ("authentic", "fabricated"):
        tp = sum(1 for cid in decided if predictions[cid] == cls and gold_labels[cid] == cls)
        fp = sum(1 for cid in decided if predictions[cid] == cls and gold_labels[cid] != cls)
        fn = sum(1 for cid in decided if predictions[cid] != cls and gold_labels[cid] == cls)
        per_class_f1[cls] = _f1(tp, fp, fn)

    macro_f1 = sum(per_class_f1.values()) / len(per_class_f1) if per_class_f1 else 0.0
    coverage = len(decided) / n
    verdict_counts = dict(Counter(predictions.values()))

    return VerificationMetrics(
        system=system,
        language=language,
        n=n,
        strict_accuracy=n_correct_strict / n,
        coverage=coverage,
        selective_accuracy=n_correct_decided / len(decided) if decided else 0.0,
        macro_f1=macro_f1,
        abstention_rate=n_abstain / n,
        per_class_f1=per_class_f1,
        verdict_counts=verdict_counts,
    )


def results_to_predictions(
    results: list[VerificationResult],
) -> dict[str, PredictionLabel]:
    """Convert a list of VerificationResults to claim_id -> prediction mapping."""
    return {
        r.claim_id: map_verdict_to_prediction(r.verdict, r.contradicts_claim)
        for r in results
    }


def compute_delta_bn(
    en_metrics: VerificationMetrics,
    bn_metrics: VerificationMetrics,
) -> dict[str, float]:
    """Compute Δ_BN = EN_metric - BN_metric."""
    return {
        "Δ_BN strict_accuracy": en_metrics.strict_accuracy - bn_metrics.strict_accuracy,
        "Δ_BN macro_f1": en_metrics.macro_f1 - bn_metrics.macro_f1,
        "Δ_BN abstention_rate": bn_metrics.abstention_rate - en_metrics.abstention_rate,
    }
