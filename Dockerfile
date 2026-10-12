# The app image (SPEC §14) — and, run as `python -m rag.restore`, the restore tool
# (SPEC §15.5): no repo checkout, no Python and no qdrant-client on the host.
#
# The topology, labels and mem_limit are `compose.yml`'s. What lives here is what travels
# with the image itself: the code, the `index_lock.json` it was committed against, the
# BGE-M3 weights that lock names, the corpus manifest the licence footer renders, and the
# healthcheck.

# --- weights -----------------------------------------------------------------
# BGE-M3, in the image rather than fetched on first start: the wake path makes no network
# call (SPEC §15.5), and a 2.3 GB download inside Sablier's waiting page would be exactly the
# cold start that rule exists to prevent. Pinned to the revision `index_lock.json` records,
# so the query encoder is the one that embedded the served index — fetching `main` would be
# whatever BAAI pushed last.
#
# A stage of its own, depending on the lock alone: an edit to the README or a dependency
# bump re-runs the layers below, and must not download 2.3 GB again to do it.
FROM python:3.13-slim AS weights

# The versions uv.lock resolves, so the download is the client the app itself runs.
RUN pip install --no-cache-dir huggingface-hub==0.36.2 hf-xet==1.6.0

COPY docker/fetch_weights.py index_lock.json /tmp/
RUN python /tmp/fetch_weights.py

# --- app ---------------------------------------------------------------------
FROM python:3.13-slim

# Pinned, not :latest — the same reason uv.lock is committed.
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Dependencies first, in their own layer: a code change should not re-download torch.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

# `rag.ingest.embedder.MODEL_CACHE_DIR`, relative to the checkout root. From here on nothing
# reaches the Hub: a load that needed a file left out fails now, not on a visitor's wake —
# which is what the load below checks, ahead of the code so a code change skips it.
COPY --from=weights /hf_cache ./data/raw/hf_cache
COPY index_lock.json ./
COPY docker/check_weights.py /tmp/
ENV HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1
RUN python /tmp/check_weights.py

COPY src ./src
COPY data/corpus/corpus_manifest.json data/corpus/LICENSE.md ./data/corpus/
# Installed in place, not as a wheel: the package finds `index_lock.json`, the corpus
# manifest and its `data/raw/hf_cache` model cache relative to the checkout root.
RUN uv sync --frozen --no-dev

EXPOSE 8000

# SPEC §14.3 — tolerating a long model load: startup blocks on BGE-M3, so for the first
# minutes the probe meets a closed port. urllib raises on a 503, so any /health clause
# failing fails the probe; the 4 s client timeout sits inside the 5 s one.
HEALTHCHECK --interval=10s --timeout=5s --retries=30 --start-period=30s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "rag.app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
