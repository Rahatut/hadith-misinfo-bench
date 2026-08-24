"""Evidence corpus definitions and configurations (RAG-2 vs RAG-17)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CollectionConfig:
    name: str
    stem_variants: tuple[str, ...]
    is_core_rag2: bool = True


# RAG-2 (Core Benchmark Corpus)
RAG2_COLLECTIONS: dict[str, CollectionConfig] = {
    "bukhari": CollectionConfig(
        name="Sahih al-Bukhari",
        stem_variants=("bukhari", "eng-bukhari", "ara-bukharibukhari"),
        is_core_rag2=True,
    ),
    "muslim": CollectionConfig(
        name="Sahih Muslim",
        stem_variants=("muslim", "eng-muslim", "ara-muslimmuslim"),
        is_core_rag2=True,
    ),
}

# Extended collections for RAG-17 ablation
RAG17_EXTENSIONS: dict[str, CollectionConfig] = {
    "abudawud": CollectionConfig(
        name="Sunan Abi Dawud",
        stem_variants=("abudawud", "eng-abudawud", "ara-abudawud"),
        is_core_rag2=False,
    ),
    "tirmidhi": CollectionConfig(
        name="Jami` at-Tirmidhi",
        stem_variants=("tirmidhi", "eng-tirmidhi", "ara-tirmidhi"),
        is_core_rag2=False,
    ),
    "nasai": CollectionConfig(
        name="Sunan an-Nasa'i",
        stem_variants=("nasai", "eng-nasai", "ara-nasai"),
        is_core_rag2=False,
    ),
    "ibnmajah": CollectionConfig(
        name="Sunan Ibn Majah",
        stem_variants=("ibnmajah", "eng-ibnmajah", "ara-ibnmajah"),
        is_core_rag2=False,
    ),
    "malik": CollectionConfig(
        name="Muwatta Malik",
        stem_variants=("malik", "eng-malik", "ara-malik"),
        is_core_rag2=False,
    ),
    "nawawi": CollectionConfig(
        name="40 Hadith Nawawi",
        stem_variants=("nawawi", "eng-nawawi", "ara-nawawi"),
        is_core_rag2=False,
    ),
}
