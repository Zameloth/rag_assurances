"""The footer's DILA / Licence Ouverte 2.0 attribution and the corpus snapshot stamp, read
from `corpus_manifest.json` (SPEC §13.5, §16.4, #50).

A licence *condition*, not decoration: rendered from the script-emitted manifest rather
than hand-written into a template, so it cannot drift from what was actually downloaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

__all__ = ["CORPUS_MANIFEST_PATH", "CorpusAttribution", "SourceAttribution", "load_attribution"]

CORPUS_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "data" / "corpus" / "corpus_manifest.json"

# The manifest's top-level keys (SPEC §16.4), named for a reader.
_SOURCE_NAMES = {"articles": "Code des assurances", "fiches": "Fiches service-public.fr"}


@dataclass(frozen=True)
class SourceAttribution:
    """One manifest entry: the producer, the licence, and SPEC §16.4's LO 2.0 three —
    download URL, filename, file date. `mirror_of` is set only when the download URL is
    not DILA's own."""

    name: str
    producer: str
    licence: str
    licence_url: str
    download_url: str
    filename: str
    file_date: date
    mirror_of: str | None


@dataclass(frozen=True)
class CorpusAttribution:
    sources: tuple[SourceAttribution, ...]
    # SPEC §10.5/§13.5: one corpus-level stamp, rendered once — the corpus is in-force-only,
    # so per-document dates would repeat one fact thousands of times. The *oldest* source
    # file date, not the latest fetch: re-downloading an old mirror today does not make the
    # article text it carries any fresher.
    snapshot_date: date


def load_attribution(path: Path = CORPUS_MANIFEST_PATH) -> CorpusAttribution:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    sources = tuple(
        SourceAttribution(
            name=_SOURCE_NAMES[key],
            producer=entry["producer"],
            licence=entry["licence"],
            licence_url=entry["licence_url"],
            download_url=entry["download_url"],
            filename=entry["filename"],
            file_date=date.fromisoformat(entry["file_date"]),
            mirror_of=entry.get("mirror_of"),
        )
        for key, entry in sorted(manifest.items())
    )
    return CorpusAttribution(
        sources=sources, snapshot_date=min(source.file_date for source in sources)
    )
