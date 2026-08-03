# Data licence — Licence Ouverte 2.0

The contents of `data/corpus/` — `articles.jsonl` and `fiches/*.xml` — are published by the
**DILA** (Direction de l'information légale et administrative) under the
**[Licence Ouverte / Open Licence v2.0](https://www.etalab.gouv.fr/wp-content/uploads/2017/04/ETALAB-Licence-Ouverte-v2.0.pdf)**
(Etalab), pursuant to the [Arrêté du 24 juin 2014](https://www.legifrance.gouv.fr/loda/id/JORFTEXT000029135221).

This applies regardless of any other licence label a third-party mirror may declare on its
own dataset card — see `mirror_of` in `corpus_manifest.json` below.

## Attribution

Licence Ouverte 2.0 requires three things for each source, all recorded per-source in
**[`corpus_manifest.json`](corpus_manifest.json) — the authoritative, machine-readable
version of this record**:

1. **Paternité** — the producer, always DILA.
2. **The long download URL** actually used to fetch the data (`download_url`), plus
   `mirror_of` when that URL is not DILA's own.
3. **The filename and file date** of the downloaded source (`filename`, `file_date`).

`corpus_manifest.json` additionally pins `retrieved_at`, `sha256` and `document_count` per
source — the provenance record that answers *"which corpus was the ladder run against?"*
from git alone (SPEC §16.4).

## Refresh

The corpus is refreshed by re-running `scripts/fetch_articles.py` by hand and committing
the result as a reviewed diff — never scheduled, never run as part of the build
(SPEC §3.3).
