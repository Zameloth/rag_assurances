#!/usr/bin/env python3
"""Fetch the in-force Code des assurances into `data/corpus/articles.jsonl` (SPEC §3, #20).

**A dev tool, run by hand — never imported from `rag` and never called by the pipeline**
(SPEC §3.3). Ingest reads `data/corpus/articles.jsonl` from git; nothing in the build path
makes a network call. Run it with the one-off dependency group it needs:

    uv run --group fetch python scripts/fetch_articles.py

**Route (SPEC §17.1)**: PISTE OAuth needs a manually-reviewed production application with
an open-ended wait; the DILA bulk tarball is 1.1 GB for a 57 MB/8,692-file slice that still
needs local in-force filtering. This pulls the HF parquet mirror
`louisbrulenaudet/code-assurances` instead — a 1:1 field match with the Légifrance API
payload and the measured source for every number in SPEC.md. The commit is resolved at run
time so the manifest records exactly what was fetched; DILA bulk stays the documented,
no-third-party fallback if this mirror ever disappears. The manifest still names DILA as
producer and Licence Ouverte 2.0 as the licence regardless of the mirror's own
`apache-2.0` card label (SPEC §16.4), and records `mirror_of` so the hop is never hidden.

Before writing anything, this checks that `texteHtml` is present and non-empty on every
row (SPEC §3.2's hard requirement — the sole basis for splitting over-band articles once
chunking lands) and that ingest assertions 1–4 (SPEC §7.4) pass over the filtered rows. A
corpus that fails its own assertions is not written.

**Refresh (SPEC §3.3, #22)**: also before writing, this diffs the freshly-fetched rows
against whatever `articles.jsonl` is already committed and reports three counts — added,
removed, and changed-text-under-the-same-`cid`, the dangerous one since 52% of Code
articles have been amended at least once. And it re-runs ingest assertion 5 against
whatever fiches are already committed: an articles-only refresh can drop or move a
`sectionParentId` a committed fiche's `<dc:source>` depends on, and that must fail loudly
here rather than surface later as a silent expansion gap. This script is never scheduled —
refresh is a manual, reviewed commit, run by hand and reviewed as a diff.

Writes `articles.jsonl` and `corpus_manifest.json` only. `data/corpus/LICENSE.md` is a
static file carrying no per-fetch data — it points at the manifest rather than duplicating
it (SPEC §16.2) — so it is authored once, not regenerated here.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq

from rag.ingest.assertions import CorpusAssertionError, run_article_assertions, run_fiche_assertions
from rag.ingest.fiches import parse_fiche
from rag.ingest.refresh_diff import diff_corpus, load_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_CORPUS = REPO_ROOT / "data" / "corpus"

HF_REPO = "louisbrulenaudet/code-assurances"
HF_FILE_PATH = "data/train-00000-of-00001.parquet"
HF_API_URL = f"https://huggingface.co/api/datasets/{HF_REPO}"

# The canonical DILA landing page for the Code des assurances — what `mirror_of` points
# a reader at. The bulk LEGI tarball behind it is documented in SPEC §17.1 / research note
# docs/research/corpus-sources.md as the reproducible, no-third-party alternative route.
DILA_CANONICAL_URL = "https://www.legifrance.gouv.fr/codes/texte_lc/LEGITEXT000006073984/"
LICENCE_OUVERTE_URL = "https://www.etalab.gouv.fr/wp-content/uploads/2017/04/ETALAB-Licence-Ouverte-v2.0.pdf"

# The subset of the source's own fields kept in articles.jsonl, mapped `source column ->
# output key`. §3.2: "no structure to protect; the field set is fixed" — this is the
# source's field set, reshaped into sorted JSONL and renamed only where SPEC's own
# vocabulary (CONTEXT.md, §7.3) already fixes a different name for the field; everything
# else keeps its source name rather than pre-empting the §7.2 payload taxonomy, which is
# an ingest-time transform, not a corpus-commit one.
FIELD_MAP = {
    "cid": "cid",  # chronicle id — identity everywhere (point ids, gold labels, joins)
    "id": "id",  # this version's id — provenance + Légifrance click-through URL
    "num": "citation_id",  # SPEC §7.3's name: verbatim, never null, never normalized
    "texte": "texte",  # plain text — zero newlines corpus-wide
    "texteHtml": "texteHtml",  # the hard requirement (§3.2): sole basis for over-band splitting
    "etat": "etat",  # always VIGUEUR after the filter below
    "dateDebut": "dateDebut",  # ISO date — converted from epoch-ms so a refresh diffs legibly
    "dateFin": "dateFin",  # ISO date; DILA's ~2999-01-01 sentinel means "no scheduled end"
    "nature": "nature",  # "Article" for every VIGUEUR row; kept in case that ever isn't true
    "type": "type",  # "AUTONOME" for every VIGUEUR row; kept in case that ever isn't true
    "sectionParentId": "sectionParentId",  # LEGISCTA id — the expansion join target (§9.2)
    "sectionParentTitre": "sectionParentTitre",
    "fullSectionsTitre": "fullSectionsTitre",  # prompt display (§7.2), first segment dropped there
    "nota": "nota",
    "notaHtml": "notaHtml",
    "ref": "ref",  # human-readable citation, e.g. "Code des assurances, art. L100-1"
}


def main() -> None:
    commit_sha, file_date = _resolve_commit()
    download_url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{commit_sha}/{HF_FILE_PATH}"
    filename = HF_FILE_PATH.rsplit("/", 1)[-1]
    retrieved_at = datetime.now(UTC)

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_RAW / filename
    print(f"Downloading {download_url}")
    sha256 = _download(download_url, raw_path)
    print(f"  -> {raw_path} ({raw_path.stat().st_size:,} bytes, sha256={sha256})")

    rows = pq.read_table(raw_path).to_pylist()
    print(f"Read {len(rows)} rows from {filename}")

    missing_html = [r["cid"] for r in rows if not r.get("texteHtml")]
    if missing_html:
        raise SystemExit(
            f"{len(missing_html)} row(s) missing texteHtml — the fetch route no longer "
            f"satisfies SPEC §3.2's hard requirement: {missing_html[:10]}"
        )

    etat_counts = Counter(r["etat"] for r in rows)
    print(f"etat distribution: {dict(etat_counts)}")
    # Assertion 2 is the literal string "VIGUEUR", so a row in a legally-in-force-today
    # edge state such as ABROGE_DIFF (repealed with deferred effect) is dropped here even
    # though it has not actually lapsed yet. This is why the committed count can read a few
    # articles under the source's own "in force" snapshot total — see README.md's Corpus
    # section for the count this run actually produced.
    in_force = [r for r in rows if r["etat"] == "VIGUEUR"]
    dropped = len(rows) - len(in_force)
    if dropped:
        print(f"Dropping {dropped} row(s) with etat != VIGUEUR (assertion 2): {dict(etat_counts)}")

    articles = [_project(row) for row in in_force]
    articles.sort(key=lambda row: row["cid"])

    try:
        run_article_assertions(articles)
    except CorpusAssertionError as exc:
        raise SystemExit(f"Refusing to write a corpus that fails its own assertions:\n{exc}") from exc
    print(f"Ingest assertions 1-4 passed over {len(articles)} articles")

    _gate_committed_fiches_still_resolve(articles)
    _report_refresh_diff(articles)

    DATA_CORPUS.mkdir(parents=True, exist_ok=True)
    _write_jsonl(DATA_CORPUS / "articles.jsonl", articles)
    print(f"Wrote {DATA_CORPUS / 'articles.jsonl'}")

    manifest = {
        "articles": {
            "producer": "DILA",
            "licence": "Licence Ouverte 2.0",
            "licence_url": LICENCE_OUVERTE_URL,
            "download_url": download_url,
            "filename": filename,
            "file_date": file_date,
            "retrieved_at": retrieved_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sha256": sha256,
            "document_count": len(articles),
            "mirror_of": DILA_CANONICAL_URL,
        }
    }
    manifest_path = DATA_CORPUS / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


def _resolve_commit() -> tuple[str, str]:
    """The dataset repo's current commit sha and its date, resolved at run time.

    Not hardcoded: a refresh re-resolves whatever is current on the `main` branch and
    records exactly that in the manifest, the same "fetch current, pin what you got"
    shape as the DILA bulk route's daily tarballs.
    """
    response = httpx.get(HF_API_URL, timeout=30, follow_redirects=True)
    response.raise_for_status()
    meta = response.json()
    commit_sha: str = meta["sha"]
    last_modified = datetime.fromisoformat(meta["lastModified"].replace("Z", "+00:00"))
    return commit_sha, last_modified.date().isoformat()


def _download(url: str, dest: Path) -> str:
    """Stream `url` to `dest`, returning the sha256 of what was written."""
    digest = hashlib.sha256()
    with httpx.stream("GET", url, timeout=120, follow_redirects=True) as response:
        response.raise_for_status()
        with dest.open("wb") as f:
            for chunk in response.iter_bytes():
                digest.update(chunk)
                f.write(chunk)
    return digest.hexdigest()


def _project(row: dict[str, Any]) -> dict[str, Any]:
    return {out_key: _convert(src_key, row[src_key]) for src_key, out_key in FIELD_MAP.items()}


def _convert(src_key: str, value: Any) -> Any:
    if src_key in ("dateDebut", "dateFin") and value is not None:
        return _epoch_ms_to_date(value).isoformat()
    return value


def _epoch_ms_to_date(epoch_ms: int) -> date:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).date()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _gate_committed_fiches_still_resolve(articles: list[dict[str, Any]]) -> None:
    """SPEC §3.3 — assertion 5 gates the refresh, not only the first ingest.

    An articles-only refresh can drop or move a `sectionParentId` that an already-committed
    fiche's `<dc:source>` depends on. Re-running assertion 5 here, against whatever fiches
    are already on disk, catches a broken fiche->section join at refresh time instead of
    leaving it for `scripts/fetch_fiches.py` or the test suite to discover later. Empty or
    missing fiches (the pre-#21 first ingest) is not a violation — there is nothing to check.
    """
    fiches_dir = DATA_CORPUS / "fiches"
    fiche_paths = sorted(fiches_dir.glob("F*.xml")) if fiches_dir.exists() else []
    if not fiche_paths:
        return
    parsed = [parse_fiche(path.read_bytes()) for path in fiche_paths]
    try:
        run_fiche_assertions(
            ({"fiche_id": fiche.fiche_id, "section_ids": fiche.section_ids} for fiche in parsed),
            articles,
        )
    except CorpusAssertionError as exc:
        raise SystemExit(
            "Refusing to write: this refresh would break the fiche->section join for "
            f"already-committed fiches (run scripts/fetch_fiches.py after fixing this):\n{exc}"
        ) from exc
    print(f"Ingest assertion 5 passed over {len(parsed)} already-committed fiche(s) against the refreshed articles")


def _report_refresh_diff(articles: list[dict[str, Any]]) -> None:
    """SPEC §3.3 — added / removed / changed-text-under-the-same-`cid`, before writing anything.

    Compares against whatever `articles.jsonl` is already committed; empty on a first ingest,
    which correctly reports every row as added and nothing as changed.
    """
    existing_path = DATA_CORPUS / "articles.jsonl"
    old_rows = load_jsonl(existing_path) if existing_path.exists() else []
    old_by_cid = {row["cid"]: row["texte"] for row in old_rows}
    new_by_cid = {row["cid"]: row["texte"] for row in articles}
    diff = diff_corpus(old_by_cid, new_by_cid)
    print(f"Refresh diff: {diff.summary()}")
    if diff.changed:
        citation_by_cid = {row["cid"]: row["citation_id"] for row in articles}
        print(
            f"{len(diff.changed)} article(s) changed text under the same cid — "
            "re-review gold labels touching these:"
        )
        for cid in diff.changed:
            print(f"  {cid}  {citation_by_cid.get(cid, '?')}")


if __name__ == "__main__":
    main()
