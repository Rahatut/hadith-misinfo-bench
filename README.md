# HadithMisinfoBench

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**HadithMisinfoBench** is a controlled benchmark for **cross-lingual canonical Hadith evidence retrieval**, focusing on Bangla claims matched against an Arabic–English Hadith corpus.

The benchmark studies a practical retrieval problem: a Hadith claim may be expressed in Bangla while the canonical evidence corpus is available primarily in Arabic and English. We compare lexical retrieval, Bangla-to-English translation-assisted retrieval, multilingual dense retrieval, and lexical-semantic rank fusion.

The current study evaluates:

* English BM25
* Direct Bangla BM25
* Bangla-to-English translation + BM25
* BGE-M3
* multilingual E5-Large
* LaBSE
* multilingual MiniLM
* BM25 + dense retrieval using Reciprocal Rank Fusion (RRF)

The canonical evidence corpus contains **14,940 Hadith records** from **Sahih al-Bukhari** and **Sahih Muslim**.

> **Important scope:** HadithMisinfoBench evaluates **canonical evidence retrieval**, not theological Hadith authentication. Successful retrieval means that a source-associated canonical record was recovered; retrieval failure does not establish that a claim is fabricated.

---

## Overview

The benchmark is designed around the following problem:

> **Given an English or Bangla representation of a Hadith claim, can a retrieval system recover the corresponding canonical Hadith evidence from an Arabic–English evidence corpus?**

The central challenge is a representation mismatch:

```text
                    Hadith claim
                         │
              ┌──────────┴──────────┐
              │                     │
           English                 Bangla
              │                     │
              ▼                     ▼
          English BM25       Translation / Dense
                                    │
                                    ▼
                         Arabic–English Corpus
                                    │
                                    ▼
                          Canonical Evidence
```

The benchmark separates:

1. **Claim representation** — English or Bangla paraphrases derived from the same Arabic source proposition.
2. **Evidence corpus** — canonical Hadith records from Sahih al-Bukhari and Sahih Muslim.
3. **Retrieval system** — lexical, translation-assisted, multilingual dense, or hybrid retrieval.
4. **Evaluation metadata** — gold evidence identifiers used only by the evaluation harness.

---

## Research Questions

### RQ1 — Cross-Lingual Lexical Degradation

How strongly does retrieval performance degrade when an English query is replaced by a Bangla representation while the evidence corpus remains Arabic–English?

The primary comparison is:

```text
English BM25
       vs.
Direct Bangla BM25
```

### RQ2 — Cross-Lingual Mitigation

Can translation or multilingual semantic retrieval recover evidence that direct Bangla lexical retrieval fails to retrieve?

We compare:

```text
Direct Bangla BM25
        │
        ├── Bangla → English → BM25
        │
        └── Bangla → Multilingual Dense Retrieval
```

### RQ3 — Lexical-Semantic Fusion

Can combining lexical and multilingual semantic rankings improve evidence retrieval?

We evaluate Reciprocal Rank Fusion between English BM25 and each multilingual dense retriever.

---

# Key Results

The reported retrieval evaluation uses **986 authentic claims with established canonical evidence associations**.

### Overall Retrieval Performance

| System                     |       R@1 |       R@5 |      R@10 |       MRR |
| :------------------------- | --------: | --------: | --------: | --------: |
| **RRF: BM25 + LaBSE**      | **0.419** | **0.579** | **0.614** | **0.486** |
| RRF: BM25 + BGE-M3         |     0.390 |     0.572 |     0.600 |     0.466 |
| RRF: BM25 + mE5-Large      |     0.369 |     0.557 |     0.590 |     0.449 |
| English BM25               |     0.348 |     0.516 |     0.553 |     0.417 |
| LaBSE, Direct Bangla       |     0.345 |     0.501 |     0.550 |     0.412 |
| RRF: BM25 + MiniLM         |     0.347 |     0.474 |     0.516 |     0.388 |
| BGE-M3, Direct Bangla      |     0.308 |     0.446 |     0.479 |     0.367 |
| mE5-Large, Direct Bangla   |     0.269 |     0.399 |     0.450 |     0.326 |
| Qwen 2.5-3B → English BM25 |     0.264 |     0.399 |     0.442 |     0.321 |
| MiniLM, Direct Bangla      |     0.000 |     0.000 |     0.004 |     0.000 |
| Direct Bangla BM25         |     0.000 |     0.001 |     0.003 |     0.001 |

The main findings are:

* English BM25 reaches **55.3% Recall@10**.
* Direct Bangla BM25 reaches only **0.3% Recall@10**.
* Translation-assisted BM25 reaches **44.2% Recall@10**.
* BGE-M3 reaches **47.9% Recall@10**.
* LaBSE reaches **55.0% Recall@10**, making it the strongest standalone multilingual retriever.
* English BM25 + LaBSE via RRF reaches **61.4% Recall@10** and **0.486 MRR**, the strongest overall configuration.

### Core Comparison

| System                     |   R@1 |   R@5 |  R@10 |
| :------------------------- | ----: | ----: | ----: |
| Direct Bangla BM25         | 0.000 | 0.001 | 0.003 |
| Qwen 2.5-3B → English BM25 | 0.264 | 0.399 | 0.442 |
| BGE-M3 Direct Bangla       | 0.308 | 0.446 | 0.479 |

BGE-M3 exceeds translation-assisted BM25 by:

* **4.5 percentage points** at Recall@1
* **4.8 percentage points** at Recall@5
* **3.7 percentage points** at Recall@10

The Recall@10 difference between BGE-M3 and translation-assisted BM25 is:

```text
+0.037
95% CI: [0.002, 0.068]
```

using paired claim-level bootstrap resampling.

---

# Benchmark Construction

The benchmark is constructed from the **MAHADDAT train partition**.

Using random seed `42`, we sample:

```text
2,000 total records
├── 1,000 authentic
└── 1,000 fabricated
```

For each selected record, the source Arabic **Matn** is used to generate paired English and Bangla representations:

```text
                Arabic Matn
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
   English paraphrase   Bangla paraphrase
```

The claims are generated using:

```text
Model:       gpt-4o-mini
Temperature: 0.3
Workers:     10 concurrent
Seed:        42
```

The English and Bangla versions correspond to the same underlying source proposition, allowing paired cross-lingual comparisons.

### Retrieval Evaluation Set

Of the 1,000 authentic records:

```text
986 → reliable canonical evidence identifiers
14   → unmatched / excluded from retrieval metrics
```

Therefore:

```text
Primary retrieval evaluation: n = 986
```

The 1,000 fabricated records remain part of the benchmark but are **not included in Recall@k or MRR**, because fabricated claims do not have explicit canonical contradiction-evidence identifiers.

---

# Anti-Leakage Design

Gold information is restricted to the evaluation harness.

Retrieval and representation components do **not** receive:

```text
source_id
benchmark label
canonical Arabic source
gold evidence IDs
expected retrieval rank
```

At inference time, the retrieval system receives only the claim representation and language/condition required by the particular experiment.

This separation prevents gold evidence information from leaking into retrieval.

---

# Evidence Corpus

The canonical evidence corpus is constructed from the public:

```text
AhmedBaset/hadith-json
```

resource and is restricted to:

* **Sahih al-Bukhari**
* **Sahih Muslim**

The resulting corpus contains:

| Collection       |    Records |
| :--------------- | ---------: |
| Sahih al-Bukhari |      7,563 |
| Sahih Muslim     |      7,377 |
| **Total**        | **14,940** |

Each Hadith is indexed as a single evidence document.

The corpus retains fields including:

```text
evidence_id
collection
book
reference
arabic_matn
english_text
grade
```

There are **no Bangla fields** in the canonical corpus.

This creates the controlled cross-lingual retrieval setting studied in the paper.

---

# Retrieval Systems

The benchmark evaluates eleven retrieval configurations.

| ID  | Query            | Retrieval | Condition               |
| :-- | :--------------- | :-------- | :---------------------- |
| S1  | English          | BM25      | English BM25            |
| S2  | Bangla           | BM25      | Direct Bangla BM25      |
| S3  | Bangla → English | BM25      | Qwen 2.5-3B + BM25      |
| S4  | Bangla           | BGE-M3    | Direct Bangla BGE-M3    |
| S5  | Bangla           | mE5-Large | Direct Bangla mE5-Large |
| S6  | Bangla           | LaBSE     | Direct Bangla LaBSE     |
| S7  | Bangla           | MiniLM    | Direct Bangla MiniLM    |
| S8  | English + Bangla | RRF       | BM25 + BGE-M3           |
| S9  | English + Bangla | RRF       | BM25 + mE5-Large        |
| S10 | English + Bangla | RRF       | BM25 + LaBSE            |
| S11 | English + Bangla | RRF       | BM25 + MiniLM           |

All systems return their top ten candidates.

---

# BM25 Retrieval

BM25 operates over the canonical evidence representation containing:

```text
Arabic Matn
English text
```

English BM25 uses the English claim directly.

Direct Bangla BM25 uses the original Bangla claim against the same Arabic–English index.

The dramatic difference between the two conditions demonstrates the effect of script and vocabulary mismatch:

```text
English BM25          R@10 = 55.3%
Direct Bangla BM25   R@10 =  0.3%
```

---

# Translation-Assisted Retrieval

The translation condition uses:

```text
Qwen/Qwen2.5-3B-Instruct
```

with deterministic greedy decoding.

The pipeline is:

```text
Bangla claim
     │
     ▼
Qwen 2.5-3B
     │
     ▼
English query
     │
     ▼
BM25
     │
     ▼
Arabic–English evidence corpus
```

Translation outputs are cached.

The resulting Recall@10 is:

```text
44.2%
```

This substantially recovers the performance lost by direct Bangla BM25, but remains below English BM25 and the strongest multilingual retrievers.

---

# Multilingual Dense Retrieval

Four multilingual encoders are evaluated:

```text
BAAI/bge-m3
intfloat/multilingual-e5-large
sentence-transformers/LaBSE
paraphrase-multilingual-MiniLM-L12-v2
```

Dense retrieval operates directly on the Bangla claims.

All dense indices contain the same:

```text
14,940 evidence records
```

Multilingual E5-Large uses its model-specific query and passage prefixes.

---

# Reciprocal Rank Fusion

RRF combines the English BM25 ranking with a multilingual dense ranking.

The RRF score is:

```text
RRF(d) = Σ 1 / (k_RRF + r_m(d))
```

with:

```text
k_RRF = 60
```

The lexical component receives the English claim, while the dense component receives the corresponding Bangla claim.

The strongest configuration is:

```text
English BM25 + LaBSE
```

with:

```text
Recall@1  = 41.9%
Recall@5  = 57.9%
Recall@10 = 61.4%
MRR        = 0.486
```

This exceeds English BM25 by **6.1 percentage points at Recall@10**, suggesting that lexical and multilingual semantic rankings provide complementary evidence.

---

# Evaluation Metrics

The primary evaluation is performed on the 986 authentic claims with established evidence associations.

### Recall@k

Recall@k measures whether at least one gold evidence document appears within the top `k` retrieved records:

```text
Recall@k =
number of claims with a gold document in top-k
------------------------------------------------
                 total claims
```

The study reports:

```text
Recall@1
Recall@5
Recall@10
```

### Mean Reciprocal Rank

MRR measures the reciprocal rank of the first retrieved gold evidence:

```text
MRR = mean(1 / rank)
```

A query receives zero when no gold evidence is retrieved.

### Statistical Analysis

Pairwise retrieval comparisons use:

```text
2,000 paired bootstrap resamples
seed = 42
unit = claim
confidence level = 95%
```

The paired design is appropriate because English and Bangla claims originate from the same underlying source proposition.

---

# Retrieval Results

The principal results are summarized below.

```text
                         Recall@10

RRF BM25 + LaBSE       ███████████████████████████████ 61.4%
RRF BM25 + BGE-M3      ██████████████████████████████  60.0%
RRF BM25 + mE5         █████████████████████████████   59.0%
English BM25           ███████████████████████████     55.3%
LaBSE                   ███████████████████████████     55.0%
RRF BM25 + MiniLM      █████████████████████████       51.6%
BGE-M3                  ████████████████████████        47.9%
mE5-Large               ██████████████████████          45.0%
Translation + BM25     █████████████████████            44.2%
MiniLM                  0.4%
Direct Bangla BM25      0.3%
```

The main conclusion is not that one multilingual model universally dominates. Rather, retrieval quality depends strongly on the interaction between:

```text
query language
+
surface-form similarity
+
multilingual representation
+
lexical evidence
```

---

# Translation vs. BGE-M3

We examined **272 authentic claims** where translation-assisted BM25 and BGE-M3 disagree at Recall@10.

Results:

| Outcome                                 |       Cases |
| :-------------------------------------- | ----------: |
| BGE-M3 succeeds where Translation fails | 154 (56.6%) |
| Translation succeeds where BGE-M3 fails | 118 (43.4%) |

Neither method dominates every claim.

This suggests that translation-mediated lexical retrieval and multilingual semantic retrieval fail on substantially different subsets of claims.

A second disagreement analysis compares BGE-M3 and LaBSE:

| Outcome                           |       Cases |
| :-------------------------------- | ----------: |
| LaBSE succeeds where BGE-M3 fails | 114 (72.2%) |
| BGE-M3 succeeds where LaBSE fails |  44 (27.8%) |

---

# Qualitative Examples

The repository includes qualitative retrieval analysis for disagreement cases.

One representative case, C0004, concerns the option of rescinding a transaction before the parties separate.

Translation-assisted BM25 fails to retrieve a gold record in its top five, whereas BGE-M3 retrieves:

```text
bukhari_2082  rank 1
bukhari_2112  rank 2  ← gold
bukhari_2111  rank 3  ← gold
```

This illustrates how multilingual semantic retrieval can recover evidence when translated wording differs substantially from the canonical English wording.

The opposite behavior occurs in C0007, where a concrete physical detail produces a strong lexical anchor. Translation-assisted BM25 retrieves the gold record:

```text
bukhari_519  rank 1
```

while BGE-M3's top five contain no gold identifier.

These cases demonstrate complementarity rather than universal superiority.

---

# MiniLM Failure Analysis

Direct multilingual MiniLM achieves only:

```text
Recall@10 = 0.4%
MRR       = 0.000
```

Among the 982 authentic queries for which MiniLM fails to retrieve a gold record in its top ten, the rank-1 predictions are concentrated in only 161 distinct evidence records.

The most frequent record alone accounts for:

```text
170 / 982 = 17.3%
```

of rank-1 predictions.

The five most frequent records account for:

```text
40.3%
```

of rank-1 predictions.

This concentration is reported as a descriptive empirical pattern. The study does not claim a specific architectural cause without direct embedding-space diagnostics.

---

# Repository Structure

```text
hadith-misinfo-bench/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── .env.example
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── mahaddat/
│   │   ├── hadith-json/
│   │   └── al-zaman/
│   │
│   ├── processed/
│   │   ├── mahaddat/
│   │   ├── evidence/
│   │   └── benchmark/
│   │
│   └── indices/
│       ├── bm25/
│       └── dense/
│
├── src/
│   └── hadith_misinfo/
│       ├── schemas.py
│       ├── config.py
│       │
│       ├── ingestion/
│       │   ├── mahaddat.py
│       │   ├── hadith_json.py
│       │   └── al_zaman.py
│       │
│       ├── preprocessing/
│       │   ├── arabic.py
│       │   ├── bangla.py
│       │   ├── english.py
│       │   └── normalize.py
│       │
│       ├── benchmark/
│       │   ├── sampler.py
│       │   ├── paraphraser.py
│       │   ├── validator.py
│       │   └── builder.py
│       │
│       ├── evidence/
│       │   ├── corpus.py
│       │   └── store.py
│       │
│       ├── retrieval/
│       │   ├── bm25.py
│       │   ├── dense.py
│       │   ├── hybrid.py
│       │   └── reranker.py
│       │
│       ├── evaluation/
│       │   ├── retrieval.py
│       │   ├── crosslingual.py
│       │   └── report.py
│       │
│       └── llm/
│           ├── base.py
│           ├── openai.py
│           ├── anthropic.py
│           └── ollama.py
│
├── scripts/
│   ├── inspect_mahaddat.py
│   ├── inspect_hadith_json.py
│   ├── inspect_al_zaman.py
│   ├── build_evidence_index.py
│   ├── build_benchmark.py
│   ├── run_experiment.py
│   ├── evaluate.py
│   └── demo.py
│
├── experiments/
│   ├── configs/
│   ├── pilot/
│   └── full/
│
├── results/
│   ├── retrieval/
│   ├── tables/
│   └── cache/
│
├── figures/
│   ├── pipeline.pdf
│   ├── benchmark_construction.pdf
│   ├── evidence_corpus.pdf
│   ├── fig1_global_recall.pdf
│   └── fig6_rrf_fusion_gain.pdf
│
└── tests/
    ├── conftest.py
    ├── test_schemas.py
    ├── test_preprocessing.py
    ├── test_ingestion.py
    ├── test_retrieval.py
    └── test_pipeline.py
```

---

# Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/your-org/hadith-misinfo-bench.git
cd hadith-misinfo-bench

python -m venv .venv
source .venv/bin/activate

pip install -e .
```

For the complete development and retrieval stack:

```bash
pip install -e ".[all,dev]"
```

---

# Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

If benchmark generation or translation is being reproduced, configure the required provider credentials.

For example:

```text
OPENAI_API_KEY=...
```

The retrieval evaluation itself does not require an LLM when using the already generated and cached benchmark/query artifacts.

---

# Quickstart

## 1. Inspect the source datasets

```bash
python scripts/inspect_mahaddat.py
python scripts/inspect_hadith_json.py
python scripts/inspect_al_zaman.py
```

## 2. Build the canonical evidence index

BM25:

```bash
python scripts/build_evidence_index.py --retriever bm25
```

Dense:

```bash
python scripts/build_evidence_index.py --retriever dense
```

Both:

```bash
python scripts/build_evidence_index.py --retriever both
```

The resulting indices should contain the same 14,940 canonical evidence records.

## 3. Build the benchmark

```bash
python scripts/build_benchmark.py \
    --provider openai \
    --model gpt-4o-mini
```

The benchmark builder should preserve the separation between inference-time fields and evaluation-only metadata.

## 4. Run retrieval experiments

English BM25:

```bash
python scripts/run_experiment.py \
    --system S1
```

Direct Bangla BM25:

```bash
python scripts/run_experiment.py \
    --system S2
```

Translation-assisted BM25:

```bash
python scripts/run_experiment.py \
    --system S3
```

BGE-M3:

```bash
python scripts/run_experiment.py \
    --system S4
```

LaBSE:

```bash
python scripts/run_experiment.py \
    --system S6
```

RRF with LaBSE:

```bash
python scripts/run_experiment.py \
    --system S10
```

The exact system identifiers should be kept consistent with the experiment configuration files.

## 5. Evaluate

```bash
python scripts/evaluate.py \
    --results-dir results/ \
    --export-dir results/tables/
```

The evaluation should report:

```text
Recall@1
Recall@5
Recall@10
MRR
paired bootstrap differences
95% confidence intervals
```

---

# Reproducibility

The principal experimental configuration is:

```text
Benchmark source:        MAHADDAT train partition
Benchmark size:          2,000
Authentic claims:        1,000
Fabricated claims:       1,000
Retrieval evaluation:    986 authentic claims
Evidence corpus:         14,940 records
Sampling seed:           42
Bootstrap resamples:     2,000
Bootstrap seed:          42
Top-k:                   10
RRF k:                   60
```

Dense retrieval models:

```text
BAAI/bge-m3
intfloat/multilingual-e5-large
sentence-transformers/LaBSE
paraphrase-multilingual-MiniLM-L12-v2
```

Translation model:

```text
Qwen/Qwen2.5-3B-Instruct
```

Benchmark claim generation:

```text
gpt-4o-mini
temperature = 0.3
```

Experiments were executed using a Google Colab NVIDIA T4 GPU. Translation outputs and retrieval artifacts are cached to support reproducibility.

---

# What Is Evaluated

The current paper evaluates:

* English BM25
* Direct Bangla BM25
* Bangla-to-English translation + BM25
* BGE-M3
* multilingual E5-Large
* LaBSE
* multilingual MiniLM
* BM25 + BGE-M3 RRF
* BM25 + mE5-Large RRF
* BM25 + LaBSE RRF
* BM25 + MiniLM RRF
* Recall@1
* Recall@5
* Recall@10
* MRR
* paired bootstrap confidence intervals
* translation/dense disagreement analysis
* multilingual retrieval comparison
* MiniLM retrieval-failure concentration analysis
* qualitative claim-level case studies

The following components may exist in the repository but should not be interpreted as experimentally validated by the current paper unless corresponding results are explicitly included:

* cross-encoder reranking
* alternative hybrid retrieval strategies
* additional evidence collections
* domain-adapted retrievers
* downstream claim verification
* structured misinformation mitigation
* large-scale naturally occurring social-media evaluation

---

# Dataset C: Real-World Social Media

The repository may also contain Bangla/Banglish social-media material from the Al-Zaman religious misinformation corpus.

This material is not part of the controlled retrieval benchmark.

It is intended for qualitative investigation of issues such as:

```text
noisy spelling
Banglish transliteration
code-switching
implicit claims
fragmented claims
social-media context
claim extraction
retrieval failures
```

Because it does not provide the same controlled gold evidence associations as the 986-claim retrieval set, it should not be used to report benchmark Recall@k or MRR.

---

# Limitations

The evidence corpus is deliberately restricted to Sahih al-Bukhari and Sahih Muslim. It is therefore not an exhaustive representation of Hadith literature.

Retrieval failure means that the relevant source-associated record was not recovered from the indexed corpus. It does **not** mean that no supporting Hadith exists elsewhere.

The benchmark uses LLM-generated English and Bangla paraphrases rather than naturally occurring misinformation. Generated claims can contain terminology substitution, entity confusion, or semantic drift.

The translation-assisted system introduces an additional model-dependent transformation:

```text
Bangla claim
     ↓
English translation
     ↓
BM25 retrieval
```

Translation errors can therefore affect retrieval independently of the retriever.

The retrieval labels are also asymmetric. Authentic claims have source-associated evidence identifiers, whereas fabricated claims do not have explicit contradiction documents. For this reason, Recall@k and MRR are evaluated only on the 986 authentic claims with established evidence associations.

The dense retrieval study covers four multilingual encoders and one indexing configuration. It does not establish universal superiority of any particular model.

---

# Interpretation Boundary

HadithMisinfoBench is a computational retrieval benchmark, not an automated religious authority.

A high retrieval score means that the system can recover source-associated canonical evidence. It does not establish:

```text
Hadith authenticity
theological correctness
scholarly consensus
historical authenticity
interpretive correctness
```

Similarly:

```text
No retrieved evidence
```

must not be interpreted as:

```text
The Hadith is fabricated.
```

The benchmark should therefore be used to study **cross-lingual evidence retrieval and language-model behavior**, rather than to replace qualified Hadith scholarship.

---

# Citation

If you use HadithMisinfoBench in academic work, cite the accompanying paper:

```bibtex
@inproceedings{hadithmisinfobench2026,
  title     = {Cross-Lingual Canonical Evidence Retrieval for Bangla Hadith Claims:
               A Controlled Evaluation of Lexical, Translation-Assisted,
               Multilingual, and Hybrid Retrieval},
  booktitle = {IEEE International Conference on Computer and Information Technology (ICCIT)},
  year      = {2026}
}
```

---

# License

This project is released under the MIT License.

See [`LICENSE`](LICENSE) for the complete license text.
