"""Corpus-reading helpers shared by the golden-set validator and the annotation helper
(#33). Both need the same three things read from the same committed corpus — which fiche
ids and article `cid`s exist, and a fiche's chunk texts — so this is the one place that
reads it, rather than two copies that could drift apart.

**Read-only.** Nothing here writes to `data/corpus/` (SPEC §12's "the helper is read-only
against the corpus" constraint applies equally to the validator).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from rag.ingest.articles import ArticleRow
from rag.ingest.fiches import chunk_fiche, parse_fiche
from rag.ingest.refresh_diff import load_jsonl

__all__ = [
    "ArticleRow",
    "FicheSummary",
    "Section",
    "article_cids",
    "articles_by_section",
    "fiche_chunk_texts",
    "fiche_ids",
    "fiche_sections",
    "fiche_summaries",
    "fiche_title",
    "load_articles",
]

_DC_TITLE_TAG = "{http://purl.org/dc/elements/1.1/}title"


@dataclass(frozen=True)
class FicheSummary:
    fiche_id: str
    title: str


@dataclass(frozen=True)
class Section:
    """One `<dc:source>` LEGISCTA id, resolved to its title and in-force articles — empty
    `articles` is a legitimate outcome (SPEC §12.3: a section can sit above any article's
    immediate parent), not an error."""

    section_id: str
    title: str
    articles: tuple[ArticleRow, ...]


def load_articles(articles_path: Path) -> list[dict[str, Any]]:
    return load_jsonl(articles_path)


def article_cids(articles: Iterable[ArticleRow]) -> set[str]:
    return {row["cid"] for row in articles}


def fiche_ids(fiches_dir: Path) -> set[str]:
    """Every fiche id on disk — the filename stem, which is also the `Publication ID`
    attribute every committed fiche was fetched and named by (#21)."""
    return {path.stem for path in fiches_dir.glob("*.xml")}


def fiche_title(xml_bytes: bytes) -> str:
    root = ET.fromstring(xml_bytes)
    element = root.find(_DC_TITLE_TAG)
    return (element.text or "") if element is not None else ""


def fiche_summaries(fiches_dir: Path) -> list[FicheSummary]:
    """`(fiche_id, title)` for every committed fiche — the annotation helper's picker
    list, searchable client-side without a server round trip."""
    return [
        FicheSummary(fiche_id=path.stem, title=fiche_title(path.read_bytes()))
        for path in sorted(fiches_dir.glob("*.xml"))
    ]


def fiche_chunk_texts(fiche_id: str, fiches_dir: Path) -> list[str]:
    """A fiche's ordered chunk texts — the same `chunk_fiche` output ingest will embed, so
    a `gold_span` verbatim in one of these is guaranteed verbatim in the payload `text` a
    real retrieval run would return (SPEC §12.2)."""
    xml_bytes = (fiches_dir / f"{fiche_id}.xml").read_bytes()
    return [chunk.text for chunk in chunk_fiche(xml_bytes)]


def articles_by_section(articles: Iterable[ArticleRow]) -> dict[str, list[ArticleRow]]:
    grouped: dict[str, list[ArticleRow]] = defaultdict(list)
    for row in articles:
        section_id = row.get("sectionParentId")
        if section_id:
            grouped[section_id].append(row)
    return dict(grouped)


def fiche_sections(fiche_id: str, fiches_dir: Path, articles: Sequence[ArticleRow]) -> list[Section]:
    """The `<dc:source>` reading list (SPEC §12.3, ADR-0010) — a suggestion, never a label
    source. `annotate_app.py` renders these sections' articles as one of two equally easy
    entry points into `gold_articles`, the other being full-corpus search, so that picking
    an article the fiche never cited is never the harder path."""
    xml_bytes = (fiches_dir / f"{fiche_id}.xml").read_bytes()
    parsed = parse_fiche(xml_bytes)
    grouped = articles_by_section(articles)
    sections = []
    for section_id in parsed.section_ids:
        rows = grouped.get(section_id, [])
        title = rows[0]["sectionParentTitre"] if rows else section_id
        sections.append(Section(section_id=section_id, title=title, articles=tuple(rows)))
    return sections
