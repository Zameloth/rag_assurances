"""Corpus → points: fetch · filter · chunk · embed · upsert (SPEC §4–§7).

`lookup_key` and `assertions` land with the corpus-commit ticket (#20) because the ingest
assertions gate that commit. `pipeline.run_ingest` (#26) is the orchestration entry point;
`python -m rag.ingest` (`__main__.py`) is what `make ingest` runs.
"""
