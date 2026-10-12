"""The weights stage of the Dockerfile: BGE-M3 at the revision `index_lock.json` pins, into
`/hf_cache`, laid out as the Hugging Face cache the app loads it from offline (ADR-0026).

A file rather than a `RUN <<` heredoc, which only BuildKit understands — the host may build
with the classic builder.
"""

import json
from pathlib import Path

from huggingface_hub import snapshot_download

cache = Path("/hf_cache")  # copied to rag.ingest.embedder.MODEL_CACHE_DIR
embedder = json.loads(Path("/tmp/index_lock.json").read_text())["embedder"]
pins = {(pin["id"], pin["revision"]) for pin in embedder.values()}
if len(pins) != 1:
    raise SystemExit(f"index_lock.json pins {sorted(pins)}; this image bakes one BGE-M3")
((model_id, revision),) = pins
snapshot_download(
    model_id,
    revision=revision,
    cache_dir=cache,
    ignore_patterns=["onnx/*", "imgs/*", "*.jpg"],  # never loaded
)
# FlagEmbedding (`rag.ingest.embedder`) names no revision and so asks the cache for `main`;
# offline, that is the `refs/main` file, which a download by commit hash does not write.
ref = cache / f"models--{model_id.replace('/', '--')}" / "refs" / "main"
ref.parent.mkdir(parents=True, exist_ok=True)
ref.write_text(revision)
