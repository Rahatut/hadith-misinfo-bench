# HadithMisinfoBench

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**HadithMisinfoBench** is an evidence-grounded, cross-lingual benchmark and verification framework for evaluating how language models handle religious claims presented as Hadith, particularly claims circulating in Bangla and English social-media environments.

The benchmark evaluates whether language models can make evidence-grounded decisions by retrieving relevant canonical Hadith evidence from *Sahih al-Bukhari* and *Sahih Muslim*. The primary experimental setting compares parametric-only verification with retrieval-augmented verification and evaluates the effect of direct Bangla retrieval and Bangla-to-English query translation.

> **Important methodological boundary:** HadithMisinfoBench does not attempt to determine theological authenticity in the complete scholarly sense. It evaluates whether a system can verify a claim against an explicitly defined canonical evidence corpus. Absence from *Sahih al-Bukhari* or *Sahih Muslim* alone is not treated as proof of fabrication.

---

## Conceptual Scope

Traditional Hadith authentication involves substantially more than textual similarity. Scholarly evaluation may consider both the chain of transmission (*Sanad*) and the narrative text (*Matn*), together with established principles of Hadith criticism.

HadithMisinfoBench instead studies an NLP-specific problem:

> **Given a claim presented as a Hadith, can a language model retrieve relevant canonical evidence and produce a calibrated evidence-grounded verification decision?**

The benchmark therefore separates:

1. **Benchmark labels** inherited from the source dataset.
2. **Retrieved canonical evidence** used by the verification system.
3. **System-level verification decisions** based on that evidence.
4. **Abstention** when the available evidence corpus is insufficient.

The system uses the following three-way verification schema:

| Verdict | Meaning |
| :--- | :--- |
| `SUPPORTED` | Retrieved canonical evidence directly corroborates the claim. |
| `NOT_SUPPORTED` | Retrieved evidence explicitly contradicts the claim or attribution. |
| `INSUFFICIENT_EVIDENCE` | The indexed corpus cannot establish support or contradiction. |

`INSUFFICIENT_EVIDENCE` is an abstention state. It does **not** mean that the claim is fabricated.

---

# Research Questions

### RQ1 — Retrieval Benefit

Does explicit canonical evidence retrieval improve LLM-based verification compared with parametric-only inference?

The primary comparison is:

```text
S3 - S1

where:

* `S1` = English parametric verification
* `S3` = English retrieval-augmented verification

The corresponding Bangla comparison is:

```text
S4 - S2
```

where:

* `S2` = Bangla parametric verification
* `S4` = Bangla retrieval-augmented verification

### RQ2 — Cross-Lingual Degradation

How does presenting the same underlying Hadith claim in Bangla affect verification and retrieval performance compared with English?

The benchmark uses paired English/Bangla claims derived from the same underlying source proposition, allowing language effects to be evaluated at the claim-pair level.

The primary comparisons are:

```text
S1 vs S2
S3 vs S4
```

### RQ2b — Cross-Lingual Mitigation

Can a lightweight Bangla-to-English query translation bridge recover the performance lost by direct Bangla lexical retrieval?

The translation experiment compares:

```text
S4^Trans vs S4^BM25
```

where Bangla claims are translated into English search queries before BM25 retrieval.

### RQ3 — Real-World Robustness

What operational failure modes occur when the controlled verification pipeline is applied to naturally occurring Bangla/Banglish social-media commentary?

This is evaluated qualitatively using Dataset C.

### RQ4 — Mitigation and Explanation Quality

Can evidence-grounded responses provide more verifiable and actionable misinformation interventions than unsupported model responses?

This is treated as a framework-level research direction and should only be reported quantitatively once a dedicated evaluation protocol has been executed.

---

# System Architecture

HadithMisinfoBench is organized as a four-stage verification and mitigation pipeline.

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
                    │  2. EVIDENCE RETRIEVAL   │
                    │                          │
                    │ BM25                     │
                    │ Dense retrieval*         │
                    │ Hybrid retrieval*        │
                    └────────────┬─────────────┘
                                 │
                         Top-k canonical
                            evidence
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  3. VERIFICATION         │
                    │                          │
                    │ SUPPORTED                │
                    │ NOT_SUPPORTED            │
                    │ INSUFFICIENT_EVIDENCE    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  4. MITIGATION            │
                    │                          │
                    │ Evidence-backed response │
                    │ Canonical reference      │
                    │ Confidence / abstention  │
                    └──────────────────────────┘
```

`*` Dense, hybrid, and reranking components are implemented/planned as retrieval extensions; the primary reported results in the current study are based on BM25 retrieval.

---

# Layer 1 — Claim Extraction

Social-media text is not necessarily a clean Hadith claim.

For example, a post may contain:

```text
ভাই, রাসূল (সাঃ) নাকি বলেছেন যে...
```

alongside commentary, opinions, emojis, URLs, or unrelated text.

The extraction layer identifies the underlying alleged Hadith proposition before retrieval.

The controlled benchmark bypasses this layer because Dataset A already contains isolated claims. Dataset C is used to investigate extraction and pipeline robustness on naturally occurring social-media text.

---

# Layer 2 — Evidence Retrieval

The retrieval layer searches an independently constructed canonical evidence corpus.

The primary corpus contains Hadith records from:

* *Sahih al-Bukhari*
* *Sahih Muslim*

The current BM25 index contains approximately:

```text
14,940 Hadith records
```

with the following document fields:

```text
evidence_id
collection
book
reference
arabic_matn
english_text
grade
```

The primary retrieval configuration uses BM25 over:

```text
english_text
arabic_matn
book
reference
```

For Bangla claims, direct lexical retrieval is evaluated separately from a translation-assisted retrieval bridge.

---

# Layer 3 — Evidence-Grounded Verification

The verifier receives the original claim and retrieved evidence.

It must return exactly one of:

```text
SUPPORTED
NOT_SUPPORTED
INSUFFICIENT_EVIDENCE
```

The verification policy is intentionally conservative.

A retrieved passage that merely looks similar to the claim is not automatically sufficient for `SUPPORTED`.

Likewise, failure to retrieve a matching Hadith is not automatically sufficient for `NOT_SUPPORTED`.

The operational distinction is:

```text
Evidence supports claim
        │
        ▼
    SUPPORTED


Evidence contradicts claim
        │
        ▼
 NOT_SUPPORTED


Evidence insufficient
        │
        ▼
INSUFFICIENT_EVIDENCE
```

---

# Layer 4 — Structured Mitigation

The final layer converts the verification decision into an evidence-grounded response.

Example:

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
  "explanation": "The claim is consistent with the retrieved canonical narration.",
  "recommended_action": "The claim is supported by the retrieved canonical source.",
  "abstained": false
}
```

For an unverified claim:

```json
{
  "claim": "...",
  "verdict": "INSUFFICIENT_EVIDENCE",
  "confidence": 0.40,
  "evidence": [],
  "explanation": "The available indexed evidence does not establish support or contradiction for this claim.",
  "recommended_action": "Avoid circulating the claim as an established Hadith until it can be verified against an appropriate source.",
  "abstained": true
}
```

The response layer must preserve the distinction between:

```text
not found
```

and

```text
proven false
```

---

# Experimental Matrix

The core benchmark evaluates four primary systems.

```text
                         Same underlying claim
                                  │
                   ┌──────────────┴──────────────┐
                   │                             │
                English                       Bangla
                   │                             │
             ┌─────┴─────┐                 ┌─────┴─────┐
             │           │                 │           │
            S1          S3                S2          S4
         LLM-only      RAG              LLM-only      RAG
             │           │                 │           │
             └─────┬─────┘                 └─────┬─────┘
                   │                             │
                   ▼                             ▼
               RAG gain                     RAG gain
                S3-S1                         S4-S2

                         S3 vs S4
                            │
                            ▼
                  Cross-lingual degradation
```

| System | Language | Retrieval                   | Purpose                                      |
| :----- | :------- | :-------------------------- | :------------------------------------------- |
| **S1** | English  | None                        | Parametric baseline                          |
| **S2** | Bangla   | None                        | Cross-lingual parametric baseline            |
| **S3** | English  | BM25                        | Evidence-grounded verification               |
| **S4** | Bangla   | BM25 / translation-assisted | Cross-lingual evidence-grounded verification |

Additional experimental controls:

| System                 | Purpose                                                               |
| :--------------------- | :-------------------------------------------------------------------- |
| **S<sub>Random</sub>** | Tests whether arbitrary context expansion alone improves verification |
| **S<sub>Oracle</sub>** | Provides an approximate upper bound using gold canonical evidence     |
| **S4<sup>Trans</sup>** | Tests Bangla-to-English query translation before BM25 retrieval       |

---

# Cross-Lingual Retrieval

The primary cross-lingual problem is a script and vocabulary mismatch.

The canonical evidence corpus contains Arabic and English text, whereas the benchmark provides Bangla claims.

Direct BM25 retrieval therefore follows:

```text
Bangla claim
     │
     ▼
BM25
     │
     ▼
Arabic / English evidence
```

which creates severe lexical mismatch.

The translation-assisted configuration instead uses:

```text
Bangla claim
     │
     ▼
Bangla → English translation
     │
     ▼
English query
     │
     ▼
BM25
     │
     ▼
Arabic / English evidence
```

The translation component receives only the Bangla claim and does not have access to:

* ground-truth labels,
* gold evidence IDs,
* retrieval results,
* expected verdicts.

This isolates query translation as a retrieval-side mitigation.

---

# Dataset Design

HadithMisinfoBench uses three logically separated datasets.

```text
Dataset A
MAHADDAT Benchmark
        │
        ├── English claims
        ├── Bangla claims
        └── Benchmark labels


Dataset B
Canonical Evidence Corpus
        │
        ├── Sahih al-Bukhari
        └── Sahih Muslim


Dataset C
Real-World Social Media
        │
        └── Bangla / Banglish Facebook comments
```

## Dataset A — Controlled Benchmark

The primary evaluation benchmark contains:

```text
490 paired claims
245 authentic
245 fabricated
```

The claims are sampled from the held-out MAHADDAT test partition using:

```text
seed = 42
```

Each benchmark instance contains an aligned English/Bangla pair derived from the same underlying Hadith proposition:

```text
Canonical Arabic Matn
        │
        ├──────────────► English paraphrase
        │
        └──────────────► Bangla paraphrase
```

This pairing is important because it reduces claim-difficulty variation when measuring cross-lingual degradation.

The benchmark inference interface exposes only:

```text
claim_text
language
```

Ground-truth labels and evidence identifiers remain inside the evaluation harness.

### Anti-Leakage Contract

The verification system does not receive:

```text
source_id
label
canonical_arabic
gold_evidence_ids
```

These fields are restricted to evaluation.

---

# Dataset B — Canonical Evidence Corpus

The primary evidence index is constructed independently of the benchmark evaluation records.

Current reported corpus:

| Collection       |    Records |
| :--------------- | ---------: |
| Sahih al-Bukhari |      7,563 |
| Sahih Muslim     |      7,377 |
| **Total**        | **14,940** |

Each Hadith is indexed as a single evidence document rather than being arbitrarily split into chunks.

Example schema:

```text
evidence_id
collection
book
reference
arabic_matn
english_text
grade
```

The corpus is intentionally restricted.

Therefore:

> Absence from Dataset B cannot establish that a Hadith claim is theologically fabricated.

This restriction is fundamental to the interpretation of all results.

---

# Dataset C — Real-World Social Media

Dataset C contains naturally occurring Bangla/Banglish social-media commentary from the Al-Zaman religious misinformation corpus.

A sample of approximately 50 comments is used for qualitative pipeline evaluation.

Dataset C is treated as an out-of-distribution robustness set.

It does not provide the same controlled benchmark labels as Dataset A. Consequently, Dataset C is not used to claim quantitative generalization accuracy.

The qualitative audit focuses on:

* noisy spelling,
* Banglish,
* implicit claims,
* fragmented claims,
* claim extraction failures,
* retrieval failures,
* evidence-grounding quality,
* unsupported attribution.

---

# Three-Way Verification and Benchmark Labels

The benchmark's source labels are binary:

```text
AUTHENTIC
FABRICATED
```

The system's operational outputs are three-way:

```text
SUPPORTED
NOT_SUPPORTED
INSUFFICIENT_EVIDENCE
```

These should not be treated as equivalent concepts.

The evaluation mapping is:

| System Verdict          | Evaluation Mapping |
| :---------------------- | :----------------- |
| `SUPPORTED`             | `AUTHENTIC`        |
| `NOT_SUPPORTED`         | `FABRICATED`       |
| `INSUFFICIENT_EVIDENCE` | Abstention         |

Under strict accuracy, abstentions are counted as incorrect because a benchmark label exists.

Under selective metrics, abstentions are excluded from the denominator.

This allows the benchmark to distinguish:

```text
accuracy
coverage
selective precision
abstention behavior
```

rather than rewarding a system simply for abstaining on difficult claims.

---

# Metrics

## Verification Metrics

### Strict Accuracy

Abstentions count as incorrect:

```text
Strict Accuracy =
Number of correct predictions / Total benchmark instances
```

### Selective Accuracy

Only non-abstained predictions are evaluated:

```text
Selective Accuracy =
Correct decided predictions / Total decided predictions
```

### Coverage

```text
Coverage =
Number of decided predictions / Total benchmark instances
```

### Abstention Rate

```text
Abstention Rate =
Number of abstained predictions / Total benchmark instances
```

### Selective Macro-F1

Macro-F1 is calculated over:

```text
AUTHENTIC
FABRICATED
```

after removing `INSUFFICIENT_EVIDENCE` predictions.

Selective Macro-F1 must therefore always be interpreted together with coverage.

---

# Retrieval Metrics

Retrieval metrics are evaluated on the authentic subset because these instances have identifiable canonical evidence targets.

Current retrieval evaluation uses:

```text
n = 245 authentic claims
```

### Recall@k

Measures whether a gold evidence document occurs within the top `k` retrieved documents.

### Mean Reciprocal Rank

```text
MRR = mean(1 / rank)
```

for the first relevant evidence document.

Retrieval metrics are not interpreted as measures of fabrication detection because fabricated claims do not have corresponding canonical evidence documents in the restricted evidence corpus.

---

# Cross-Lingual Metrics

For any metric `M`:

```math
Δ_BN(M) = M_EN - M_BN
```

where the English and Bangla instances correspond to the same underlying claim pair.

The benchmark also measures Relative Gap Mitigation (RGM):

```math
RGM =
(M_{S4^{Trans}} - M_{S4^{BM25}})
/
(M_{S3^{BM25}} - M_{S4^{BM25}})
```

RGM measures the proportion of the English-vs-direct-Bangla retrieval gap recovered through query translation.

---

# Statistical Analysis

Reported confidence intervals use paired bootstrap resampling at the claim-pair level.

The current evaluation uses:

```text
95% confidence intervals
1,000 bootstrap iterations
claim-pair resampling
```

The paired design is important because each English/Bangla instance corresponds to the same underlying source proposition.

---

# Current Experimental Results

The current study reports the following primary verification results on the 490-claim benchmark:

| System                                        | Correct / 490 | Strict Accuracy |   Coverage | Abstention |
| :-------------------------------------------- | ------------: | --------------: | ---------: | ---------: |
| **S1 — English Parametric**                   |           109 |          0.2224 |     0.2551 |     74.49% |
| **S2 — Bangla Parametric**                    |            46 |          0.0939 |     0.0980 |     90.20% |
| **S<sub>Random</sub>**                        |           139 |          0.2837 |     0.4633 |     53.67% |
| **S3 — English BM25 RAG**                     |       **209** |      **0.4265** | **0.5143** | **48.57%** |
| **S4 — Direct Bangla BM25**                   |             4 |          0.0082 |     0.0082 |     99.18% |
| **S4<sup>Trans</sup> — Bangla Translate-RAG** |           184 |          0.3755 |     0.5020 |     49.80% |
| **S<sub>Oracle</sub>**                        |           398 |          0.8122 |     0.8776 |     12.24% |

The main English retrieval result is:

```text
S1: 0.2224
S3: 0.4265
```

representing:

```text
+20.41 percentage points
+91.74% relative improvement
```

in strict accuracy.

Direct Bangla BM25 retrieval exhibits near-total collapse:

```text
MRR = 0.004
Strict Accuracy = 0.0082
Abstention = 99.18%
```

The translation bridge improves Bangla strict accuracy:

```text
S4^BM25      = 0.0082
S4^Trans     = 0.3755
```

corresponding to approximately:

```text
87.80% Relative Gap Mitigation
```

relative to the English-RAG/direct-Bangla-RAG gap.

---

# Retrieval Results

On the 245 authentic claims:

| Retrieval Mode   | Recall@1 | Recall@5 |   MRR |
| :--------------- | -------: | -------: | ----: |
| English Direct   |    0.362 |    0.512 | 0.424 |
| Bangla Direct    |    0.004 |    0.004 | 0.004 |
| Bangla → English |    0.312 |    0.448 | 0.368 |

These results demonstrate that the primary failure of direct Bangla BM25 retrieval is retrieval-side lexical/script mismatch rather than simply a reduction in model reasoning capability.

---

# Verification Asymmetry

The benchmark reveals an important asymmetry between authentic and fabricated claims.

For English BM25 RAG:

```text
Authentic F1   = 0.88
Fabricated F1  = 0.69
```

The system is substantially better at confirming claims when relevant canonical evidence exists than rejecting fabricated claims when explicit refutation evidence is unavailable.

This is a structural property of the evidence corpus.

A repository containing authentic Hadith does not automatically contain explicit documents stating:

```text
"This fabricated claim is false."
```

Consequently, fabricated-claim rejection is intrinsically more difficult under the current evidence model.

---

# Random and Oracle Controls

Two additional configurations help interpret the RAG results.

## Random Evidence Control

`S_Random` supplies randomly selected Hadith passages rather than retrieved relevant evidence.

Its purpose is to test whether additional context alone explains the RAG improvement.

The current results show:

```text
Random context:
Coverage = 0.4633
Selective Accuracy = 0.6120

Relevant BM25 context:
Coverage = 0.5143
Selective Accuracy = 0.8294
```

This indicates that simply increasing context does not explain the observed performance gain.

Evidence relevance matters.

## Oracle Evidence Control

`S_Oracle` supplies the gold canonical evidence document.

The resulting strict accuracy is:

```text
0.8122
```

This provides an approximate upper bound for the current verification architecture and indicates that retrieval quality remains a substantial bottleneck.

---

# Repository Structure

```text
hadith-misinfo-bench/
│
├── README.md
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
│       ├── extraction/
│       │   ├── extractor.py
│       │   └── prompts.py
│       │
│       ├── verification/
│       │   ├── verifier.py
│       │   ├── prompts.py
│       │   └── policies.py
│       │
│       ├── mitigation/
│       │   ├── responder.py
│       │   └── prompts.py
│       │
│       ├── llm/
│       │   ├── base.py
│       │   ├── openai.py
│       │   ├── anthropic.py
│       │   └── ollama.py
│       │
│       └── evaluation/
│           ├── retrieval.py
│           ├── verification.py
│           ├── crosslingual.py
│           ├── grounding.py
│           └── report.py
│
├── scripts/
│   ├── inspect_mahaddat.py
│   ├── inspect_hadith_json.py
│   ├── inspect_al_zaman.py
│   ├── build_evidence_index.py
│   ├── build_benchmark.py
│   ├── run_experiment.py
│   ├── evaluate.py
│   ├── run_social_media.py
│   └── demo.py
│
├── experiments/
│   ├── configs/
│   ├── pilot/
│   └── full/
│
├── results/
│   ├── retrieval/
│   ├── verification/
│   ├── grounding/
│   └── tables/
│
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

# Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/your-org/hadith-misinfo-bench.git
cd hadith-misinfo-bench

pip install -e .
```

For the complete development/retrieval stack:

```bash
pip install -e ".[all,dev]"
```

---

# Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Configure the required LLM provider credentials.

For example:

```text
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
OLLAMA_BASE_URL=...
```

Only the provider required by the selected experiment needs to be configured.

---

# Quickstart

## 1. Inspect source datasets

```bash
python scripts/inspect_mahaddat.py
python scripts/inspect_hadith_json.py
python scripts/inspect_al_zaman.py
```

## 2. Build the evidence index

BM25:

```bash
python scripts/build_evidence_index.py --retriever bm25
```

Dense retrieval:

```bash
python scripts/build_evidence_index.py --retriever dense
```

Both:

```bash
python scripts/build_evidence_index.py --retriever both
```

## 3. Build the paired benchmark

```bash
python scripts/build_benchmark.py \
    --provider openai \
    --model gpt-4o-mini
```

The resulting benchmark should preserve the anti-leakage contract and expose only inference-time fields to the verification system.

## 4. Run experiments

English parametric:

```bash
python scripts/run_experiment.py --system S1
```

Bangla parametric:

```bash
python scripts/run_experiment.py --system S2
```

English BM25 RAG:

```bash
python scripts/run_experiment.py \
    --system S3 \
    --retriever bm25
```

Bangla RAG:

```bash
python scripts/run_experiment.py \
    --system S4 \
    --retriever bm25
```

## 5. Evaluate results

```bash
python scripts/evaluate.py \
    --results-dir results/ \
    --export-dir results/tables/
```

## 6. Run the social-media pipeline

```bash
python scripts/run_social_media.py
```

## 7. Interactive demonstration

```bash
python scripts/demo.py \
    --system S4 \
    --retriever bm25
```

---

# Development and Testing

Run the test suite with:

```bash
pytest
```

The test suite covers:

```text
schemas
preprocessing
dataset ingestion
retrieval
verification
end-to-end pipeline behavior
```

The benchmark generation and evaluation pipeline should be deterministic wherever possible through explicit random seeds and stored experiment configurations.

---

# Experimental Reproducibility

Each experiment should record at minimum:

```text
system_id
model_id
retriever
retrieval_top_k
temperature
random_seed
benchmark_version
evidence_corpus_version
prompt_version
timestamp
```

The evaluation harness must retain a strict separation between:

```text
inference-time inputs
```

and

```text
evaluation-only metadata
```

This prevents labels or gold evidence IDs from being inadvertently exposed to retrieval or verification components.

---

# Limitations

HadithMisinfoBench has several important limitations.

### Restricted Evidence Corpus

The primary evidence corpus consists of *Sahih al-Bukhari* and *Sahih Muslim*. It is therefore not an exhaustive representation of the Hadith literature.

### Evidence Asymmetry

Authentic claims may have identifiable canonical evidence, while fabricated claims generally lack explicit refutation documents.

Consequently:

```text
No retrieved evidence
```

does not imply:

```text
Fabricated
```

### Benchmark-to-Real-World Gap

Dataset A consists of controlled claims, whereas real social-media posts contain:

* spelling errors,
* code-switching,
* incomplete claims,
* commentary,
* quotations,
* sarcasm,
* multiple claims,
* ambiguous attribution.

Dataset C provides qualitative investigation of this gap but does not constitute a fully labeled real-world benchmark.

### Translation Dependence

The translation-assisted Bangla retrieval system introduces an additional model-dependent transformation:

```text
Bangla claim
→ English query
→ retrieval
```

Translation errors can therefore affect retrieval quality.

### Retrieval Bottleneck

The difference between oracle and retrieved-evidence performance indicates that evidence retrieval remains a major source of system error.

### Benchmark Label Semantics

The binary MAHADDAT labels are used as benchmark ground truth. They should not be interpreted as a complete formal scholarly authentication judgment.

---

# Research Status

The project currently distinguishes between evaluated components and planned extensions.

## Evaluated

* English parametric verification
* Bangla parametric verification
* English BM25 RAG
* Direct Bangla BM25 retrieval
* Bangla-to-English translation-assisted BM25
* Random-context control
* Oracle-evidence control
* Retrieval Recall@1
* Retrieval Recall@5
* MRR
* Strict Accuracy
* Selective Accuracy
* Coverage
* Selective Macro-F1
* Abstention rate
* Paired bootstrap confidence intervals
* Qualitative Dataset C analysis

## Extension / Planned Evaluation

The repository also contains infrastructure for:

* BGE-M3 dense retrieval
* Hybrid BM25 + dense retrieval
* Reciprocal Rank Fusion
* Cross-encoder reranking
* expanded evidence collections
* larger social-media evaluation
* structured mitigation evaluation

These components should not be interpreted as experimentally validated results until the corresponding experiments are executed and reported.

---

# License

This project is released under the MIT License.

See:

```text
LICENSE
```

for the complete license text.

---

# Ethical and Interpretive Statement

HadithMisinfoBench is an NLP benchmark, not an automated religious authority.

System outputs are constrained by the evidence corpus, retrieval mechanism, language representation, and verification model. A `SUPPORTED` result means that the claim is corroborated by the indexed evidence under the benchmark's operational definition. An `INSUFFICIENT_EVIDENCE` result means that the system could not establish support or contradiction from the available corpus.

The benchmark should therefore be used to study evidence-grounded language-model behavior rather than to replace qualified Hadith scholarship.

```
