"""Tests for raw dataset ingestion adapters."""

import pandas as pd
import pytest

from hadith_misinfo.ingestion.mahaddat import (
    COLUMN_MAP,
    LABEL_MAP,
    RawRecord,
    iter_records,
)


def test_label_mapping():
    assert LABEL_MAP["authentic"] == "authentic"
    assert LABEL_MAP["fabricated"] == "fabricated"
    assert LABEL_MAP["صحيح"] == "authentic"
    assert LABEL_MAP["موضوع"] == "fabricated"


def test_iter_records_csv_fixture(tmp_path):
    csv_path = tmp_path / "mahaddat_test.csv"
    df = pd.DataFrame({
        "arabic": ["حديث صحيح", "حديث موضوع"],
        "label": ["authentic", "fabricated"],
        "split": ["test", "test"],
    })
    df.to_csv(csv_path, index=False)

    records = list(iter_records(tmp_path, split="test"))
    assert len(records) == 2
    assert records[0].label == "authentic"
    assert records[1].label == "fabricated"
