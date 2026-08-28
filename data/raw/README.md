# Data

This directory contains the source data, processed benchmark data, and evidence corpus used in the experiments on cross-lingual Hadith claim retrieval and evidence assessment.

## MAHADDAT Dataset

The benchmark is constructed from **MAHADDAT**, a dataset developed for Hadith misinformation detection. MAHADDAT contains Hadith-related claims annotated according to their authenticity status and provides Arabic Hadith text that can be used as the canonical textual source.

In this work, MAHADDAT serves as the source dataset from which the benchmark instances are selected. The original Arabic `Matn` is retained as the canonical representation of a Hadith, while corresponding English and Bangla claim representations are used to evaluate cross-lingual retrieval.

The benchmark uses a controlled subset of the source data. With **random seed 42**, we sample **2,000 records** with a balanced distribution:

- **1,000 authentic claims**
- **1,000 fabricated claims**

The original MAHADDAT records are retained under `raw/mahaddat` for provenance. Derived benchmark files are stored under `processed/benchmark`.

## Evidence Corpus

The **evidence corpus** is the collection of canonical Hadith texts used as the retrieval target in this study.

Rather than retrieving from translated or generated evidence, the corpus retains the original **Arabic Hadith `Matn`** as the canonical evidence representation. Each evidence document is assigned a stable identifier so that retrieved documents can be matched against the corresponding ground-truth evidence during evaluation.

The evidence corpus is constructed from the canonical Hadith records associated with the source data and is treated as a **fixed retrieval collection** throughout the experiments.

This creates a cross-lingual retrieval setting in which the query and evidence may be expressed in different languages:

```text
                 Query
        ┌──────────┼──────────┐
        │          │          │
      Arabic     English     Bangla
        │          │          │
        └──────────┼──────────┘
                   ▼
          Cross-Lingual Retriever
                   │
                   ▼
          Arabic Evidence Corpus
                   │
                   ▼
          Ranked Evidence Documents

```

The original files are retained under `raw/hadith-json` while derived evidence corpus file is stored under `processed/`.