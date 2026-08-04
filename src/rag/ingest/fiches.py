"""Fiche scope-rule parsing (SPEC §1.2, §3.1, #21).

Extracts the two fields the scope rule and ingest assertion 5 need out of a fiche's raw
DILA XML — the `Publication` id and the `LEGISCTA` ids in `<dc:source>` — and nothing else.
Full-body parsing (`<Chapitre>`/`<Cas>`/…) is §4.1's job, the chunking ticket, not this one:
parsing here would answer that question by accident (SPEC §3.2).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import NamedTuple
from xml.etree import ElementTree as ET

__all__ = ["ParsedFiche", "parse_fiche", "in_scope"]

_DC_SOURCE_TAG = "{http://purl.org/dc/elements/1.1/}source"
_LEGISCTA_RE = re.compile(r"LEGISCTA\d+")


class ParsedFiche(NamedTuple):
    fiche_id: str
    section_ids: list[str]


def parse_fiche(xml_bytes: bytes) -> ParsedFiche:
    """`fiche_id` (the `Publication` id) and the deduped `LEGISCTA` ids in `<dc:source>`.

    A fiche without a `<dc:source>` element, or with one carrying no `LEGISCTA` id, gets an
    empty `section_ids` — it is trivially out of scope (SPEC §1.2), not an error.
    """
    root = ET.fromstring(xml_bytes)
    fiche_id = root.attrib["ID"]
    source_el = root.find(_DC_SOURCE_TAG)
    source_text = source_el.text or "" if source_el is not None else ""
    deduped: dict[str, None] = dict.fromkeys(_LEGISCTA_RE.findall(source_text))
    return ParsedFiche(fiche_id=fiche_id, section_ids=list(deduped))


def in_scope(section_ids: Iterable[str], article_section_ids: set[str]) -> bool:
    """SPEC §1.2's scope rule: a nonempty intersection with the in-force articles' sections."""
    return any(section_id in article_section_ids for section_id in section_ids)
