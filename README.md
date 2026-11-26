# HadithMisinfoBench

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**HadithMisinfoBench** is a controlled benchmark for **cross-lingual canonical Hadith evidence retrieval**: given a Bangla (or English) claim, can a system recover source-associated evidence from an Arabic–English Hadith corpus?

This repository accompanies the ICCIT 2026 paper *"Cross-Lingual Canonical Evidence Retrieval for Bangla Hadith Claims"* (see [Citation](#citation)). It contains the benchmark, the evidence corpus, the retrieval systems, the evaluation code, and a re-analysis notebook that regenerates every number reported in the paper.

> **Scope.** This is a benchmark for **evidence retrieval**, not Hadith authentication. Retrieval success means a *source-associated* record was recovered. Retrieval failure does **not** mean a claim is fabricated.

---

## Contents

- [Overview](#overview)
- [Key results](#key-results)
- [Benchmark construction](#benchmark-construction)
- [Evidence corpus](#evidence-corpus)
- [Retrieval systems](#retrieval-systems)
- [Evaluation protocol](#evaluation-protocol)
- [Detailed findings](#detailed-findings)
- [Re-analysis notebook and outputs](#re-analysis-notebook-and-outputs)
- [Repository structure](#repository-structure)
- [Installation and quickstart](#installation-and-quickstart)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Interpretation boundary](#interpretation-boundary)
- [Citation](#citation)
- [License](#license)

---

## Overview

The core difficulty is a representation mismatch: queries are in Bangla, but the evidence corpus contains Arabic *Matn* and English translations and **no Bangla text**.

```text
                 Arabic Matn (source)
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
   English claim (C_EN)        Bangla claim (C_BN)   ← primary query
          │                           │
          ▼                           ▼
   English BM25        Direct BM25 / Qwen→BM25 / Dense / RRF
                                      │
                                      ▼
                        Arabic–English evidence corpus
                                      │
                                      ▼
                          Top-10 candidate records
```

**Research questions**

| RQ | Question |
| :-- | :-- |
| RQ1 | How does retrieval change when queries shift from English to Bangla, for lexical and for dense retrievers? |
| RQ2 | To what extent can LLM translation or multilingual dense encoders mitigate the mismatch? |
| RQ3 | Does lexical–semantic rank fusion (RRF) improve evidence recovery over standalone retrievers? |

---

## Key results

All results are on the **986 authentic claims** (each has five derived gold records). Recall@k is a hit-rate: a claim counts as solved if *at least one* gold record is in the top *k*. MRR is MRR@10.

### Overall retrieval performance (Table II of the paper)

| System | Query | R@1 | R@5 | R@10 | MRR |
| :-- | :-- | --: | --: | --: | --: |
| **RRF: English BM25 + LaBSE** (S10) | EN + BN | **0.420** | **0.579** | **0.614** | **0.487** |
| RRF: English BM25 + BGE-M3 (S8) | EN + BN | 0.390 | 0.574 | 0.600 | 0.467 |
| RRF: English BM25 + mE5-Large (S9) | EN + BN | 0.369 | 0.556 | 0.590 | 0.449 |
| RRF: Qwen→BM25 + LaBSE (S14) | BN only | 0.336 | 0.521 | 0.576 | 0.418 |
| RRF: Qwen→BM25 + BGE-M3 (S12) | BN only | 0.294 | 0.518 | 0.574 | 0.390 |
| RRF: Qwen→BM25 + mE5-Large (S13) | BN only | 0.286 | 0.507 | 0.557 | 0.380 |
| English BM25 (S1) | EN | 0.348 | 0.516 | 0.553 | 0.417 |
| LaBSE, direct Bangla (S6) | BN | 0.346 | 0.501 | 0.550 | 0.412 |
| RRF: English BM25 + MiniLM (S11) | EN + BN | 0.347 | 0.476 | 0.516 | 0.388 |
| BGE-M3, direct Bangla (S4) | BN | 0.308 | 0.446 | 0.479 | 0.367 |
| mE5-Large, direct Bangla (S5) | BN | 0.269 | 0.398 | 0.450 | 0.326 |
| Qwen 2.5-3B → English BM25 (S3) | BN→EN | 0.264 | 0.399 | 0.442 | 0.321 |
| RRF: Qwen→BM25 + MiniLM (S15) | BN only | 0.264 | 0.362 | 0.399 | 0.297 |
| MiniLM, direct Bangla (S7) | BN | 0.000 | 0.000 | 0.004 | 0.000 |
| Direct Bangla BM25 (S2) | BN | 0.000 | 0.002 | 0.008 | 0.001 |

S12–S15 values come from `bangla_only_rrf_table.csv` in the re-analysis outputs.

### Headline findings

- **Direct Bangla BM25 collapses** (0.8% Recall@10 vs. 55.3% for English BM25). 968 of 986 Bangla queries share no token with the index, so its top-10 is pure tie order (0.3–0.8% depending on tie-breaking).
- **Translation recovers much of the gap:** Qwen 2.5-3B → BM25 reaches 44.2%, but costs **11.1 pp** relative to English BM25 (95% CI [+8.6, +13.6]).
- **LaBSE is the strongest standalone encoder** (55.0% R@10), followed by BGE-M3 (47.9%) and mE5-Large (45.0%). MiniLM fails (0.4%).
- **The best overall result needs a parallel English claim.** English BM25 + LaBSE reaches 61.4% R@10 (0.487 MRR), but a real Bangla user would not have the English claim. The **best Bangla-only system is Qwen→BM25 + LaBSE at 57.6%**, only +2.6 pp over LaBSE alone (95% CI [+0.7, +4.5]).
- **Within RRF, LaBSE vs. BGE-M3 is not significant** at Recall@10 (+1.3 pp, 95% CI [−0.1, +2.8]).
- **Query language is not the main limit for working dense encoders** (see below): the corpus structure is.

### English vs. Bangla queries for dense retrievers (Recall@10, %)

| Model | English query | Bangla query | Δ (pp) |
| :-- | --: | --: | --: |
| LaBSE | 54.3 | 55.0 | +0.7 |
| BGE-M3 | 48.6 | 47.9 | −0.7 |
| mE5-Large | 48.3 | 45.0 | −3.3 |
| MiniLM | 11.7 | 0.4 | −11.3 |

For LaBSE, BGE-M3 and mE5-Large, switching from English to Bangla changes R@10 by at most 3.3 pp. MiniLM is the exception.

### Paired bootstrap (Recall@10, 95% CI, pp)

| Comparison | Diff. | 95% CI |
| :-- | --: | :-- |
| Translation − Direct Bangla BM25 | +43.4 | [+40.3, +46.6] |
| BGE-M3 − Direct Bangla BM25 | +47.1 | [+43.8, +50.2] |
| BGE-M3 − Translation | +3.7 | [+0.2, +6.8] |
| LaBSE − BGE-M3 | +7.1 | [+4.7, +9.6] |
| English BM25 + LaBSE − English BM25 | +6.1 | [+4.3, +7.9] |
| English BM25 − Translation | +11.1 | [+8.6, +13.6] |

Intervals use 2,000 paired resamples (seed 42), without multiple-comparison correction. Because 480 claims share a Matn with another claim (240 Matns appear twice), we also resample by Matn (746 clusters). The BGE-M3 advantage over translation stays significant at R@1 and R@5 but **not** at R@10 under Matn-level resampling (CI [−0.1, +7.4]), so treat it as suggestive. The full table (all *k*, claim- and Matn-level) is in `bootstrap_claim_vs_cluster.csv`.

---

## Benchmark construction

The benchmark is built from the **MAHADDAT train partition** (Gaanoun & Alsuhaibani, IEEE Access 2022).

```text
2,000 sampled records (seed 42)
├── 986 authentic   → five derived gold identifiers each (2,823 unique)
└── 1,014 fabricated → no gold identifiers
```

For each record, the source Arabic *Matn* is used to generate paired English and Bangla claims:

```text
C_AR → { C_EN, C_BN }
```

| Setting | Value |
| :-- | :-- |
| Generator | GPT-4o-mini |
| Temperature | 0.3 |
| Workers | 10 concurrent |
| Prompt | Short social-media-style restatement; must not mention Hadith, collection names or authenticity |
| Query roles | Bangla = primary query; English = reference condition |

The claims are controlled, source-derived paraphrases, **not** naturally occurring misinformation.

### Gold evidence is derived, not annotated

MAHADDAT provides no Bukhari/Muslim reference numbers. Gold is therefore **derived**: for each authentic record, BM25 is run with the **Arabic Matn as the query** over an Arabic + English index (with Arabic diacritic, alef-variant and teh-marbuta normalization), and the **top five** records are kept. The re-analysis notebook reproduces this exactly (100% of stored gold lists match, in order).

Consequences:

- Gold favours records lexically close to the source Matn.
- Gold is skewed toward Muslim records (70% vs. 49% of the corpus); about 84% of Muslim records are Arabic-only, which also lowers the ceiling for English-text systems.
- For 62.8% of claims some gold record contains at least 90% of the source tokens; for 8.1% none reaches 50% (sources absent from the two collections receive nearest neighbours).
- Recall@k therefore measures recovery of a **source-associated neighbourhood**, not a verified original.
- The 986 claims derive from **746 unique Matns** (240 appear twice, i.e., 480 claims, with identical gold).

### Anti-leakage design

Gold information is confined to the evaluation harness. Retrieval components never receive `source_id`, the benchmark label, the canonical Arabic source, gold evidence IDs, or expected ranks. At inference time a system sees only the claim representation required by the condition.

---

## Evidence corpus

Built from [`AhmedBaset/hadith-json`](https://github.com/AhmedBaset/hadith-json), restricted to Sahih al-Bukhari and Sahih Muslim.

| Collection | Records | With English text |
| :-- | --: | --: |
| Sahih al-Bukhari | 7,580 | 7,580 |
| Sahih Muslim | 7,360 | 1,191 |
| **Total** | **14,940** | **8,771** |

6,169 records (41.3% of the corpus) are **Arabic-only**. The corpus has **no Bangla fields**, which creates the controlled cross-lingual setting. Retained fields include `evidence_id`, `collection`, `book`, `reference`, `arabic_matn`, `english_text` and `grade`.

**Corpus structure bounds English-text retrieval.** 235 claims (23.8%) have no gold record with English text. English BM25 gets R@10 = 0.000 on them and 72.6% on the rest, so any English lexical route (including translation-assisted BM25) is capped at **76.2%**. Dense retrievers could in principle match these records through their Arabic text but rarely do: R@10 on these 235 claims is only 1.7% (BGE-M3), 5.5% (mE5-Large) and 6.4% (LaBSE).

---

## Retrieval systems

Fifteen systems are evaluated (Table I of the paper). All return their top ten candidates.

| ID | Query | Retrieval | Condition |
| :-- | :-- | :-- | :-- |
| S1 | EN | BM25 | English BM25 |
| S2 | BN | BM25 | Direct Bangla BM25 |
| S3 | BN→EN | BM25 | Qwen 2.5-3B + BM25 |
| S4 | BN | BGE-M3 | Direct Bangla BGE-M3 |
| S5 | BN | mE5-Large | Direct Bangla mE5-Large |
| S6 | BN | LaBSE | Direct Bangla LaBSE |
| S7 | BN | MiniLM | Direct Bangla MiniLM |
| S8 | EN + BN | RRF | English BM25 + BGE-M3 |
| S9 | EN + BN | RRF | English BM25 + mE5-Large |
| S10 | EN + BN | RRF | English BM25 + LaBSE |
| S11 | EN + BN | RRF | English BM25 + MiniLM |
| S12 | BN, BN→EN | RRF | Qwen+BM25 + BGE-M3 (Bangla-only) |
| S13 | BN, BN→EN | RRF | Qwen+BM25 + mE5-Large (Bangla-only) |
| S14 | BN, BN→EN | RRF | Qwen+BM25 + LaBSE (Bangla-only) |
| S15 | BN, BN→EN | RRF | Qwen+BM25 + MiniLM (Bangla-only) |

**Deployability.** S1 and S8–S11 use a parallel English claim, which a Bangla-only user would not have. S2–S7 and S12–S15 use the Bangla claim alone.

### Implementation details

- **BM25:** `k1 = 1.5`, `b = 0.75`; regex tokenizer with lowercasing; diacritics kept. The index covers Arabic Matn + English text.
- **Translation:** `Qwen/Qwen2.5-3B-Instruct`, deterministic greedy decoding, outputs cached. Conclusions about translation are specific to this 3B translator.
- **Dense models:** `BAAI/bge-m3`, `intfloat/multilingual-e5-large` (with its query/passage prefixes), `sentence-transformers/LaBSE`, `paraphrase-multilingual-MiniLM-L12-v2`. Each document is the Arabic Matn, a newline, then the English text, truncated to 512 tokens (long records may lose their English part). Queries are truncated to 256 tokens; embeddings are fp16 and normalized. BGE-M3's sparse and multi-vector modes are not used.
- **RRF:** `RRF(d) = Σ_m 1 / (k_RRF + r_m(d))`, `k_RRF = 60`, fusing **two top-10 lists** (at most 20 candidates). By construction RRF cannot retrieve outside the union of its inputs. Fusing deeper lists (20–100) changes R@10 by −3.5 to +0.4 pp (`rrf_depth_ablation.csv`).

---

## Evaluation protocol

```text
Recall@k = (1/N) Σ_i  1[ G_i ∩ TopK_i ≠ ∅ ]        k ∈ {1, 5, 10}
MRR@10   = (1/N) Σ_i  1 / r_i                       r_i = rank of first gold record in the top 10; 0 if absent
```

- `N = 986` authentic claims. Fabricated claims have no gold and are excluded from Recall@k and MRR.
- Recall@k is a hit-rate and is lenient with five approximate gold records per claim.
- Paired bootstrap: 2,000 resamples, seed 42, 95% CIs, at claim level and at Matn level (746 clusters).

---

## Detailed findings

### Complementarity of English BM25 and dense retrievers (Table IV)

Claims split by whether each system retrieves gold in the top 10:

| Dense partner | BM25 only | Dense only | Both | Neither | Oracle | RRF |
| :-- | --: | --: | --: | --: | --: | --: |
| LaBSE | 86 | 83 | 459 | 358 | 0.637 | 0.614 |
| BGE-M3 | 139 | 66 | 406 | 375 | 0.620 | 0.600 |
| mE5-Large | 173 | 72 | 372 | 369 | 0.626 | 0.590 |

An oracle choosing the better system per claim would reach 62.0–63.7% R@10 (65.4% for English BM25 + LaBSE + BGE-M3 combined). RRF attains 94–97% of the pairwise oracle. 36–38% of claims are missed by both systems, and about 60% of those (220–231 claims) have no English text in any gold record.

### Translation vs. dense disagreements

| Pair | Outcome | Cases |
| :-- | :-- | --: |
| Translation vs. BGE-M3 (272 disagreements) | BGE-M3 succeeds where translation fails | 154 (56.6%) |
| | Translation succeeds where BGE-M3 fails | 118 (43.4%) |
| BGE-M3 vs. LaBSE (158 disagreements) | LaBSE succeeds where BGE-M3 fails | 114 (72.2%) |
| | BGE-M3 succeeds where LaBSE fails | 44 (27.8%) |

Case studies: **C0004** (rescinding a transaction before the parties separate) is recovered by BGE-M3 (`bukhari_2112` rank 2, `bukhari_2111` rank 3) but missed by translation-assisted BM25's top five. **C0007** (a companion's foot touched before prostration) is the reverse: translation-assisted BM25 ranks `bukhari_519` first while BGE-M3's top five contain no gold. The two cases illustrate complementary failure modes.

### Why LaBSE leads (hypothesis, not a finding)

A plausible explanation is that LaBSE is trained on translation pairs and the claims are near-translations of the corpus text. This is untested here.

### MiniLM failure analysis

MiniLM's rank-1 predictions on its 982 failed queries collapse onto 162 distinct records: `muslim_495` alone accounts for 17.4% (171 queries) and the top five for 40.3%. MiniLM reaches 11.7% R@10 with English claims but 0.4% with Bangla, and retrieves the paired English claim from a Bangla claim at rank 1 only 1.0% of the time (BGE-M3: 69.4%). The evidence points to a lack of usable Bangla representations, with hubness a symptom rather than an established cause.

### Robustness of the ranking

System rankings are stable across unique-Matn, English-gold and reliable-gold subsets (Spearman ρ ≥ 0.97 over S1–S15). The reliable-gold subset keeps the 619 claims with a gold record containing ≥ 90% of the source tokens and scores only against such records; there LaBSE overtakes English BM25 (R@10 0.493 vs. 0.467).

---

## Re-analysis notebook and outputs

`HMB_audit_reanalysis_no_plots.ipynb` rebuilds all fifteen systems from the repository data and cached Qwen translations, and regenerates the paper's numbers. It:

1. audits the data (label counts, gold provenance, English-text coverage, duplicates, gold quality);
2. rebuilds S1–S15 and saves per-claim top-10 rankings;
3. checks results against `results/tables/retrieval_summary_final.csv`;
4. computes claim-level and Matn-clustered bootstrap CIs;
5. evaluates sensitivity subsets (unique Matns, reliable gold, English/no-English gold, twin-aware gold);
6. runs the RRF depth ablation (10/20/50/100);
7. runs the MiniLM failure diagnostics and the English–Bangla alignment check;
8. exports the figure data.

Runtime is about 30–50 minutes on a Colab T4 the first time (document embeddings are cached). Set `RUN_DENSE = False` for a CPU-only run of everything except the dense, RRF and diagnostic parts (about one minute).

**Deliberate differences.** BM25 ties are broken by corpus order (stable sort), so results are deterministic. Direct Bangla BM25 therefore differs slightly from earlier runs. Dense scores can differ in the third decimal from fp16/hardware effects: individual cells move by at most ±0.2 pp.

### Output files

| File | Contents |
| :-- | :-- |
| `audit_summary.json` | Label counts, gold provenance, English coverage, Matn duplicates, reliable-gold share, subset Spearman values |
| `audit_containment.csv` | Gold quality by token-containment threshold |
| `reproduction_check.csv` | Per-system comparison against the stored results table |
| `bootstrap_claim_vs_cluster.csv` | Paired bootstrap, claim-level and Matn-clustered |
| `sensitivity_subsets.csv`, `sensitivity_mrr_pivot.csv` | Subset robustness |
| `rrf_depth_ablation.csv` | RRF fusion depth 10/20/50/100 |
| `bangla_only_rrf_table.csv` | S12–S15 metrics and comparisons |
| `diag_en_bn_alignment.csv` | English vs. Bangla query results and embedding alignment |
| `diag_hubness.csv`, `diag_modal_own_failures.csv`, `diag_modal_shared_failures.csv` | MiniLM failure diagnostics |
| `rankings_top10.json`, `rankings_deep.npz` | Per-claim rankings for all systems |
| `system_order.json` | Canonical system ordering |
| `figure_data/` | Underlying data for the recall and RRF-gain figures |

---

## Repository structure

Abbreviated; see the repository for the full tree.

```text
hadith-misinfo-bench/
├── README.md
├── LICENSE
├── pyproject.toml
├── .env.example
├── data/
│   ├── raw/                  # MAHADDAT, hadith-json, Al-Zaman
│   ├── processed/
│   │   ├── evidence.jsonl                      # 14,940-record corpus
│   │   └── benchmark/benchmark_dataset_a.jsonl # 2,000-record benchmark
│   └── indices/              # BM25 and dense indices
├── src/hadith_misinfo/       # ingestion, preprocessing, benchmark, retrieval, evaluation, llm
├── scripts/                  # build, run_experiment, evaluate, inspection scripts
├── experiments/              # configs, pilot, full
├── results/
│   ├── cache/                # e.g. bangla_to_english_qwen25_3b_seed42.json
│   ├── tables/               # e.g. retrieval_summary_final.csv
│   └── reanalysis/           # outputs of the re-analysis notebook
├── notebooks/
│   └── HMB_audit_reanalysis_no_plots.ipynb
├── figures/
└── tests/
```

---

## Installation and quickstart

Python 3.10 or newer is required.

```bash
git clone https://github.com/Rahatut/hadith-misinfo-bench.git
cd hadith-misinfo-bench

python -m venv .venv
source .venv/bin/activate

pip install -e .                 # core
pip install -e ".[all,dev]"      # full retrieval and development stack
```

Copy the example environment file. An API key is only needed to regenerate benchmark claims; evaluation uses the cached benchmark and translation artifacts.

```bash
cp .env.example .env             # then set OPENAI_API_KEY=... if regenerating claims
```

```bash
# 1. Inspect source datasets
python scripts/inspect_mahaddat.py
python scripts/inspect_hadith_json.py
python scripts/inspect_al_zaman.py

# 2. Build the evidence indices (bm25 | dense | both); 14,940 records each
python scripts/build_evidence_index.py --retriever both

# 3. Build the benchmark (only to regenerate claims)
python scripts/build_benchmark.py --provider openai --model gpt-4o-mini

# 4. Run a system (S1–S11)
python scripts/run_experiment.py --system S10

# 5. Evaluate
python scripts/evaluate.py --results-dir results/ --export-dir results/tables/
```

S12–S15 (Bangla-only RRF) are produced by the re-analysis notebook.

**Re-analysis (Colab or local):**

```bash
jupyter notebook notebooks/HMB_audit_reanalysis_no_plots.ipynb
# or set HMB_REPO=/path/to/local/checkout to skip the clone
```

---

## Reproducibility

| Setting | Value |
| :-- | :-- |
| Benchmark source | MAHADDAT train partition |
| Benchmark size | 2,000 (986 authentic, 1,014 fabricated) |
| Evaluation set | 986 authentic claims, five derived gold IDs each (2,823 unique) |
| Unique Matns | 746 |
| Evidence corpus | 14,940 records |
| Sampling seed | 42 |
| Bootstrap | 2,000 resamples, seed 42, claim- and Matn-level |
| Top-k | 10 |
| RRF k | 60 |
| Claim generation | GPT-4o-mini, temperature 0.3 |
| Translation | `Qwen/Qwen2.5-3B-Instruct`, greedy decoding |
| Hardware | Google Colab NVIDIA T4 |

Re-running changes individual Recall values by at most ±0.2 pp owing to fp16 nondeterminism, except direct Bangla BM25, whose top-10 is tie-order dependent (0.3–0.8% Recall@10).

---

## Limitations

- **Corpus scope.** Only Sahih al-Bukhari and Sahih Muslim are indexed. Retrieval failure does not mean no supporting source exists elsewhere.
- **Derived gold.** Gold comes from Arabic-query BM25, so it is approximate and lexically biased. Recall@k is a lenient hit-rate over five approximate records.
- **Synthetic claims.** Claims are LLM-generated, not naturally occurring Bangla misinformation, and were not human-verified at scale. They may contain terminology substitution, entity confusion or semantic drift. Real claims will include spelling variation, Banglish, code-switching and incomplete quotations. The benchmark supports controlled comparison, not deployment estimates.
- **Translation fidelity is unmeasured.** The Qwen translations were not scored against the paired English claims, and the findings on translation are specific to a 3B translator.
- **Asymmetric labels.** Authentic claims have evidence IDs; fabricated claims have no contradiction annotations, so metrics cover the 986 authentic claims only.
- **Scope of the dense study.** Four encoders, one indexing configuration, no cross-encoder reranking, domain adaptation, larger models, alternative fusion, or downstream verification.
- **No multiplicity correction** across the many paired comparisons; borderline results should be read as suggestive.
- **MiniLM.** The failure is characterised, but its cause is not established.

Planned future work includes expert-annotated natural Bangla and Banglish claims, robustness tests, translation-fidelity measurement with stronger translators, an Arabic-bridge baseline (translating Bangla claims into Arabic), reranking, and domain adaptation.

### Al-Zaman social-media data

The repository may contain Bangla/Banglish material from the Al-Zaman religious-misinformation corpus. It is **not** part of the controlled retrieval benchmark, has no controlled gold evidence, and must not be used to report benchmark Recall@k or MRR. It is intended for qualitative study of noisy spelling, transliteration, code-switching and implicit claims.

---

## Interpretation boundary

HadithMisinfoBench is a computational retrieval benchmark, not an automated religious authority. A high retrieval score does not establish Hadith authenticity, theological correctness, scholarly consensus, or interpretive correctness. Likewise, *no retrieved evidence* must not be read as *the Hadith is fabricated*. Use it to study cross-lingual evidence retrieval, not to replace qualified Hadith scholarship.

---

## Citation

```bibtex
@inproceedings{hadithmisinfobench2026,
  title     = {Cross-Lingual Canonical Evidence Retrieval for Bangla Hadith Claims},
  author    = {Mounota, Rahatut Tahrim and Kabir, Md. Mahiul and Prince, Nayeemul Hasan},
  booktitle = {2026 29th International Conference on Computer and Information Technology (ICCIT)},
  address   = {Cox's Bazar, Bangladesh},
  month     = dec,
  year      = {2026}
}
```

If you use the benchmark claims, please also cite MAHADDAT:
K. Gaanoun and M. Alsuhaibani, "Fabricated Hadith detection: A novel Matn-based approach with transformer language models," *IEEE Access*, vol. 10, pp. 113330–113342, 2022.

---

## License

Released under the [MIT License](LICENSE). Third-party data (MAHADDAT, `hadith-json`, Al-Zaman) remains subject to its own licence and terms.
