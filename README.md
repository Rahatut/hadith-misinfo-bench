# HadithMisinfoBench

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**HadithMisinfoBench** is an evidence-grounded, cross-lingual benchmark and mitigation framework for evaluating how language models detect, verify, and respond to religious misinformation circulating on social media. The primary case study is Hadith-related claims in Bangla and English, evaluated against canonical Arabic/English Hadith evidence (*Sahih al-Bukhari*, *Sahih Muslim*).

---

## 🎯 Conceptual Scope & Methodological Boundary

> **We are not attempting to determine whether every Hadith is theologically authentic in the full scholarly sense. We are testing whether an NLP system can detect, verify, and mitigate social-media claims presented as Hadith by grounding them in an explicit canonical evidence corpus.**

Absence from a specific collection (e.g. Bukhari & Muslim) cannot establish theological fabrication. Therefore, the system enforces a strict three-way verification schema:
- **`SUPPORTED`**: The claim is directly corroborated by retrieved canonical evidence.
- **`NOT_SUPPORTED`**: The retrieved canonical evidence explicitly contradicts the claim or attribution.
- **`INSUFFICIENT_EVIDENCE`**: The indexed corpus cannot verify or refute the claim (calibrated abstention).

---

## 🔬 Research Questions

- **RQ1 (Retrieval Benefit)**: Does evidence retrieval improve LLM-based verification of Hadith misinformation compared with parametric-only verification ($S_3 - S_1$ and $S_4 - S_2$)?
- **RQ2 (Cross-Lingual Degradation)**: How does presenting Hadith misinformation claims in Bangla affect retrieval and verification compared with English ($S_1 \text{ vs } S_2$ and $S_3 \text{ vs } S_4$)?
- **RQ2b (Mitigation of Language Gap)**: Can cross-lingual evidence retrieval reduce the performance penalty associated with low/medium-resource Bangla claims?
- **RQ3 (Real-World Robustness)**: How well does the controlled verification pipeline generalize to naturally occurring, noisy Bangladeshi social-media discussions (Dataset C)?
- **RQ4 (Mitigation & Explanation Quality)**: Does evidence-grounded explanation provide more verifiable, calibrated, and actionable misinformation interventions than unsupported LLM responses?

---

## 📖 4-Layer System Architecture

```text
                    ┌──────────────────────────┐
                    │   SOCIAL MEDIA CONTENT   │
                    │   Bangla / English       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  1. CLAIM EXTRACTION     │
                    │                          │
                    │ Extract alleged Hadith   │
                    │ claim from noisy post    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ 2. EVIDENCE RETRIEVAL    │
                    │                          │
                    │ BM25                     │
                    │ BGE-M3 Dense             │
                    │ Hybrid / RRF + Reranker  │
                    └────────────┬─────────────┘
                                 │
                         Top-k canonical
                            evidence
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ 3. VERIFICATION          │
                    │                          │
                    │ SUPPORTED                │
                    │ NOT_SUPPORTED            │
                    │ INSUFFICIENT_EVIDENCE    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ 4. MITIGATION            │
                    │                          │
                    │ Evidence-backed response │
                    │ Canonical reference      │
                    │ Confidence / abstention  │
                    └──────────────────────────┘
```

### Layer 4: Structured Mitigation Output

Rather than returning a bare classification label, the mitigation layer generates an actionable intervention and structured JSON payload:

```json
{
  "claim": "রাসূল (সাঃ) বলেছেন: কাজগুলো তাদের নিয়তের উপর নির্ভর করে।",
  "verdict": "SUPPORTED",
  "confidence": 0.85,
  "evidence": [
    {
      "evidence_id": "bukhari_1",
      "collection": "Sahih al-Bukhari",
      "reference": "Hadith 1",
      "arabic_matn": "إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ...",
      "english_text": "Actions are judged by intentions..."
    }
  ],
  "explanation": "The claim accurately reflects the established narration in Sahih al-Bukhari, Hadith 1.",
  "recommended_action": "The claim is consistent with the retrieved canonical Hadith source(s).",
  "abstained": false
}
```

For unsupported or unindexed claims:
```json
{
  "claim": "...",
  "verdict": "INSUFFICIENT_EVIDENCE",
  "confidence": 0.40,
  "evidence": [],
  "explanation": "The indexed canonical collections (Bukhari & Muslim) do not contain matching evidence for this attribution.",
  "recommended_action": "This claim could not be verified against available canonical sources. Avoid circulating as an established Hadith.",
  "abstained": true
}
```

---

## 🧪 Experimental Matrix (S1–S4)

```text
                 ┌───────────────┐
                 │ Same claim    │
                 └───────┬───────┘
                         │
              ┌──────────┴──────────┐
              │                     │
          English                 Bangla
              │                     │
        ┌─────┴─────┐         ┌─────┴─────┐
        │           │         │           │
       S1          S3         S2          S4
     LLM-only     RAG       LLM-only     RAG
        │           │         │           │
        └─────┬─────┘         └─────┬─────┘
              │                     │
              ▼                     ▼
          RAG gain              BN RAG gain
          S3 - S1               S4 - S2

                    S3 vs S4
                       │
                       ▼
              Cross-lingual penalty (Δ_BN)
```

| System | Language | Retrieval | Evidence Corpus | Research Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **S1** | English | None (Parametric) | None | Closed-book baseline |
| **S2** | Bangla | None (Parametric) | None | Parametric language penalty baseline |
| **S3** | English | RAG (BM25 / BGE-M3 / Hybrid) | Bukhari + Muslim | Evidence-grounded verification |
| **S4** | Bangla | Cross-Lingual RAG | Bukhari + Muslim | Cross-lingual mitigation & verification |

---

## 🛡️ Dataset Partitions & Anti-Leakage Separation

```text
Dataset A (MAHADDAT Benchmark)
    ├── 500 Paired Claims (250 Authentic + 250 Fabricated)
    ├── Generated via direct Arabic->EN and Arabic->BN paraphrases
    └── Anti-Leakage Contract: Model receives ONLY InferenceRecord (claim_text, language)

Dataset B (Canonical Evidence Corpus)
    ├── Sahih al-Bukhari + Sahih Muslim (RAG-2 Core)
    ├── Optional 6/17 book extensions (RAG-17 Ablation)
    └── Pre-computed BM25 & BGE-M3 multilingual embeddings

Dataset C (Al-Zaman / Noman)
    └── 50–100 Real-World Bangladeshi Facebook Comments (OOD Robustness / RQ3)
```

---

## 📁 Repository Structure

```text
hadith-misinfo-bench/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── mahaddat/                      # MAHADDAT CSV release
│   │   ├── hadith-json/                   # Bukhari & Muslim JSON corpus
│   │   └── al-zaman/                      # Al-Zaman Facebook comments dataset
│   ├── processed/
│   │   ├── mahaddat/
│   │   ├── evidence/
│   │   └── benchmark/
│   └── indices/
│       ├── bm25/                          # Serialized BM25 index
│       └── dense/                         # BGE-M3 embeddings & ID map
│
├── src/hadith_misinfo/
│   ├── schemas.py                         # Pydantic schemas & anti-leakage models
│   ├── config.py                          # Centralized configuration & path management
│   │
│   ├── ingestion/
│   │   ├── mahaddat.py                    # Ingest MAHADDAT test split
│   │   ├── hadith_json.py                 # Ingest Bukhari / Muslim JSON
│   │   └── al_zaman.py                    # Ingest Al-Zaman Dataset C
│   │
│   ├── preprocessing/
│   │   ├── arabic.py                      # Tashkeel, tatweel, alef normalisation
│   │   ├── bangla.py                      # Bangla Unicode NFC & punctuation normalisation
│   │   ├── english.py                     # English text normalisation
│   │   └── normalize.py                   # Multi-script tokenizer & normalizer
│   │
│   ├── benchmark/
│   │   ├── sampler.py                     # Balanced 250+250 sampling
│   │   ├── paraphraser.py                 # Direct Arabic->EN & Arabic->BN paraphraser
│   │   ├── validator.py                   # Benchmark integrity & anti-leakage checks
│   │   └── builder.py                     # End-to-end benchmark dataset builder
│   │
│   ├── evidence/
│   │   ├── corpus.py                      # RAG-2 vs RAG-17 collection specs
│   │   └── store.py                       # Unified EvidenceStore index
│   │
│   ├── retrieval/
│   │   ├── bm25.py                        # BM25 lexical retriever
│   │   ├── dense.py                       # BGE-M3 dense multilingual retriever
│   │   ├── hybrid.py                      # Reciprocal Rank Fusion (RRF)
│   │   └── reranker.py                    # Cross-encoder reranker
│   │
│   ├── extraction/
│   │   ├── extractor.py                   # Two-stage social media claim extractor (Layer 1)
│   │   └── prompts.py                     # Detection & extraction prompts
│   │
│   ├── verification/
│   │   ├── verifier.py                    # Verification runner for S1–S4 (Layer 2)
│   │   ├── prompts.py                     # Verification prompt templates
│   │   └── policies.py                    # Label mapping & abstention policies
│   │
│   ├── mitigation/
│   │   ├── responder.py                   # Structured JSON & formatted responses (Layer 3)
│   │   └── prompts.py                     # Mitigation templates & recommendations
│   │
│   ├── llm/
│   │   ├── base.py                        # Protocol & exponential back-off wrapper
│   │   ├── openai.py                      # OpenAI client adapter
│   │   ├── anthropic.py                   # Anthropic Claude client adapter
│   │   └── ollama.py                      # Ollama local LLM adapter
│   │
│   └── evaluation/
│       ├── retrieval.py                   # Recall@1, Recall@5, MRR
│       ├── verification.py                # Strict/Selective Acc, Macro-F1, Abstention
│       ├── crosslingual.py                # Δ_BN & language gap mitigation metrics
│       ├── grounding.py                   # Grounded / Ungrounded qualitative audit
│       └── report.py                      # Summary table & export generator
│
├── scripts/
│   ├── inspect_mahaddat.py                # Diagnostic: inspect MAHADDAT schema
│   ├── inspect_hadith_json.py             # Diagnostic: inspect hadith-json schema
│   ├── inspect_al_zaman.py                # Diagnostic: inspect Al-Zaman comments
│   ├── build_evidence_index.py            # Build BM25 / Dense indices
│   ├── build_benchmark.py                 # Generate paired benchmark (Dataset A)
│   ├── run_experiment.py                  # Run S1–S4 verification experiments
│   ├── evaluate.py                        # Compute metrics and generate tables
│   ├── run_social_media.py                # Run Dataset C (Al-Zaman) OOD validation
│   └── demo.py                            # Interactive 4-layer CLI demo
│
├── experiments/ (configs/, pilot/, full/)
├── results/ (retrieval/, verification/, grounding/, tables/)
└── tests/
    ├── conftest.py
    ├── test_schemas.py
    ├── test_preprocessing.py
    ├── test_ingestion.py
    ├── test_retrieval.py
    ├── test_verification.py
    └── test_pipeline.py
```

---

## ⚡ Quickstart

### 1. Installation

```bash
git clone https://github.com/your-org/hadith-misinfo-bench.git
cd hadith-misinfo-bench

# Base install
pip install -e .

# Full install (Dense embeddings + LLM SDKs + Dev tools)
pip install -e ".[all,dev]"
```

### 2. Environment Setup

```bash
cp .env.example .env
# Edit .env and supply OPENAI_API_KEY, ANTHROPIC_API_KEY, or local OLLAMA_BASE_URL
```

### 3. Pipeline Execution

```bash
# 1. Diagnostic schema checks
python scripts/inspect_mahaddat.py
python scripts/inspect_hadith_json.py
python scripts/inspect_al_zaman.py

# 2. Build Evidence Indices (BM25 + BGE-M3 Dense)
python scripts/build_evidence_index.py --retriever both

# 3. Build Paired Benchmark (Dataset A)
python scripts/build_benchmark.py --provider openai --model gpt-4o-mini

# 4. Run S1–S4 Experiments
python scripts/run_experiment.py --system S1
python scripts/run_experiment.py --system S2
python scripts/run_experiment.py --system S3 --retriever dense
python scripts/run_experiment.py --system S4 --retriever dense

# 5. Evaluate & Export Tables
python scripts/evaluate.py --results-dir results/ --export-dir results/tables/
```

### 4. Interactive Demonstration

```bash
python scripts/demo.py --system S4 --retriever dense
```

---

## 📊 Metrics & Formulas

- **Strict Accuracy**: Abstentions counted as incorrect ($N_{\text{correct}} / N$).
- **Selective Accuracy**: Accuracy computed only over non-abstained claims ($N_{\text{correct, decided}} / N_{\text{decided}}$).
- **Coverage**: Fraction of evaluated claims where the system did not abstain ($N_{\text{decided}} / N$).
- **Macro-F1**: Unweighted mean of per-class F1 scores (Authentic / Fabricated) over decided claims.
- **Recall@k & MRR**: Evaluated strictly on authentic claims with ground-truth evidence targets.
- **Cross-Lingual Degradation ($\Delta_{\text{BN}}$)**:
  $$\Delta_{\text{BN}}(M) = M_{\text{EN}} - M_{\text{BN}}$$
- **Grounding Audit**: Manual evaluation of 50 sampled RAG outputs categorised into:
  - *Grounded* (verdict corroborated by retrieved citation)
  - *Partially Grounded* (evidence partially matched / slight drift)
  - *Ungrounded* (hallucinated attribution or citation mismatch)
