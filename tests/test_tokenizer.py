"""SPEC §4 — the band is measured with the BGE-M3 tokenizer, not an approximation."""

import json
from pathlib import Path

from rag.ingest.tokenizer import count_tokens


def test_counts_special_tokens() -> None:
    """`<s>` + `</s>` are always added — the convention SPEC §4's own numbers were taken under."""
    assert count_tokens("") == 2


def test_longer_text_counts_more_tokens() -> None:
    assert count_tokens("Bonjour le monde") > count_tokens("Bonjour")


def test_matches_spec_measured_median() -> None:
    """Calibration check: SPEC §4 reports article `texte` median 167 tokens on this corpus."""
    articles_path = Path(__file__).resolve().parents[1] / "data" / "corpus" / "articles.jsonl"
    counts = sorted(count_tokens(json.loads(line)["texte"]) for line in articles_path.read_text(encoding="utf-8").splitlines())
    median = counts[len(counts) // 2]
    assert median == 167
