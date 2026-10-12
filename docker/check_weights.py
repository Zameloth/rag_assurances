"""The Dockerfile's offline load of the baked weights (ADR-0026): the model the lock names,
loaded the way `rag.ingest.embedder` loads it, with the Hub unreachable — so a file the
weights stage left out fails the build rather than a visitor's wake.

It runs ahead of `COPY src`, so a code change does not re-run a 2.3 GB load; that is why it
restates `embedder._model()`'s arguments instead of importing it.
"""

import json
from pathlib import Path

from FlagEmbedding import BGEM3FlagModel

model_id = json.loads(Path("index_lock.json").read_text())["embedder"]["dense"]["id"]
BGEM3FlagModel(model_id, use_fp16=False, cache_dir="data/raw/hf_cache")
