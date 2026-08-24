"""Cross-lingual performance evaluation and language degradation metrics (Δ_BN)."""

from __future__ import annotations

from dataclasses import dataclass

from hadith_misinfo.evaluation.retrieval import RetrievalMetrics
from hadith_misinfo.evaluation.verification import VerificationMetrics


@dataclass
class CrossLingualAnalysis:
    # Parametric language gap: S1 (EN) vs S2 (BN)
    parametric_delta_strict_acc: float
    parametric_delta_macro_f1: float
    parametric_delta_abstention: float

    # RAG language gap: S3 (EN) vs S4 (BN)
    rag_delta_strict_acc: float
    rag_delta_macro_f1: float
    rag_delta_abstention: float

    # Language gap reduction by RAG: (S1-S2) - (S3-S4)
    # Positive means RAG helped close the cross-lingual performance gap
    gap_reduction_strict_acc: float
    gap_reduction_macro_f1: float


def compute_crosslingual_analysis(
    s1: VerificationMetrics,
    s2: VerificationMetrics,
    s3: VerificationMetrics,
    s4: VerificationMetrics,
) -> CrossLingualAnalysis:
    """Analyze whether cross-lingual RAG mitigates the Bangla performance penalty (RQ2b)."""
    p_delta_acc = s1.strict_accuracy - s2.strict_accuracy
    p_delta_f1 = s1.macro_f1 - s2.macro_f1
    p_delta_abs = s2.abstention_rate - s1.abstention_rate

    r_delta_acc = s3.strict_accuracy - s4.strict_accuracy
    r_delta_f1 = s3.macro_f1 - s4.macro_f1
    r_delta_abs = s4.abstention_rate - s3.abstention_rate

    gap_red_acc = p_delta_acc - r_delta_acc
    gap_red_f1 = p_delta_f1 - r_delta_f1

    return CrossLingualAnalysis(
        parametric_delta_strict_acc=p_delta_acc,
        parametric_delta_macro_f1=p_delta_f1,
        parametric_delta_abstention=p_delta_abs,
        rag_delta_strict_acc=r_delta_acc,
        rag_delta_macro_f1=r_delta_f1,
        rag_delta_abstention=r_delta_abs,
        gap_reduction_strict_acc=gap_red_acc,
        gap_reduction_macro_f1=gap_red_f1,
    )


def compute_retrieval_delta_bn(
    en_retrieval: RetrievalMetrics,
    bn_retrieval: RetrievalMetrics,
) -> dict[str, float]:
    """Compute Δ_BN for retrieval metrics: Recall@1, Recall@5, MRR."""
    return {
        "Δ_BN Recall@1": en_retrieval.recall_at_1 - bn_retrieval.recall_at_1,
        "Δ_BN Recall@5": en_retrieval.recall_at_5 - bn_retrieval.recall_at_5,
        "Δ_BN MRR": en_retrieval.mrr - bn_retrieval.mrr,
    }
