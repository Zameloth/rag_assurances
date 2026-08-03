"""Corpus → points: fetch · filter · chunk · embed · upsert (SPEC §4–§7).

`lookup_key` and `assertions` land with the corpus-commit ticket (#20) because the ingest
assertions gate that commit. Chunking, embedding and upsert are still empty — build order
step 2.
"""
