"""`make publish-index`: the scored index -> Parquet points dump -> GitHub Release (SPEC §15, ADR-0014, #52).

**It does not re-embed.** The ladder's scored index is not a file on disk — it is the dev
Qdrant's two arms, reached through the stable `articles`/`fiches` aliases every rung read
through. `dump_register` scrolls one of them back out, so what ships is the exact point set
the ladder measured rather than a re-derivation of it (SPEC §15.3): a rebuild, even from
the same committed corpus, runs on a different torch build and is *not the artifact that
was scored*.

**Parquet, float32, never text** (SPEC §15.2). The decisive property is exactness — the
deployed vectors must *be* the measured vectors — and fp32 stored natively round-trips by
construction, where 3.8M floats through JSON is the one step that could silently move one.
The payload is the exception that *is* JSON: one string column, the flat schema verbatim,
so the dump never grows a column per payload field that the restore would have to know.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models

from rag.config import load_settings
from rag.eval.retrieval_run import load_run, resolve_git_sha
from rag.ingest.ab_arms import ARTICLE_BREADCRUMB_ARM, FICHE_HEADER_ARM
from rag.ingest.arms import ARTICLES_ALIAS, FICHES_ALIAS
from rag.ingest.articles import BAND, STUB_FLOOR

# Private on purpose: making it public would edit a chunker file, which the publish guard
# below reads as "chunking changed since the scored run" — true of the bytes, false of the
# behaviour, and it would demand a re-ingest and a full ladder re-run to publish.
from rag.ingest.fiches import _MERGE_FLOOR as MERGE_FLOOR
from rag.ingest.pipeline import ARTICLES_ARM, FICHES_ARM, REPO_ROOT

__all__ = [
    "EmbedderPin",
    "IndexLock",
    "PublishError",
    "RegisterDump",
    "ReleaseFn",
    "ScoresPointer",
    "dump_register",
    "main",
    "publish",
]

# `(tag, target commit, assets, notes)` — raises if the release was not cut.
ReleaseFn = Callable[[str, str, Sequence[Path], str], None]

DEFAULT_CORPUS_MANIFEST = REPO_ROOT / "data" / "corpus" / "corpus_manifest.json"
DEFAULT_LOCK_PATH = REPO_ROOT / "index_lock.json"
# SPEC §16.1 — the dump is derived, so it lands under gitignored `data/raw/` until released.
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw" / "index"
# Where `rag.ingest.embedder`/`e5_embedder` point the HF cache — restated rather than
# imported, since importing either module loads torch to read one path.
_HF_CACHE_DIR = REPO_ROOT / "data" / "raw" / "hf_cache"

# The arms an enrichment A/B built (ADR-0021). Neither was adopted, but a run header naming
# one is still a run whose shipped index would carry enriched dense vectors.
_ENRICHED_ARMS = frozenset({ARTICLE_BREADCRUMB_ARM, FICHE_HEADER_ARM})

# Keyed by the plural collection name, as the aliases and the release asset names are.
_REGISTER_ALIASES = {"fiches": FICHES_ALIAS, "articles": ARTICLES_ALIAS}
_DEFAULT_ARMS = {"fiches": FICHES_ARM, "articles": ARTICLES_ARM}

# What decides the chunk population and the payload. `chunk_config` and
# `corpus_manifest_sha256` are read at publish time, so they are true of the scored index
# only if none of these moved since the run that scored it.
_CHUNKING_PATHS = (
    "data/corpus",
    "src/rag/ingest/articles.py",
    "src/rag/ingest/fiches.py",
    "src/rag/ingest/html_blocks.py",
    "src/rag/ingest/text_split.py",
    "src/rag/ingest/tokenizer.py",
    "src/rag/ingest/payload.py",
    "src/rag/ingest/lookup_key.py",
)

_SCROLL_PAGE = 256


class PublishError(Exception):
    """What would be published is not provably the index the named run scored."""


@dataclass(frozen=True)
class RegisterDump:
    """One register's line in `index_lock.json` (SPEC §15.6), minus the register name."""

    collection: str
    points: int
    asset_sha256: str
    vector_config_fingerprint: str


@dataclass(frozen=True)
class EmbedderPin:
    id: str
    revision: str


@dataclass(frozen=True)
class ScoresPointer:
    """The per-item scores that chose this index — `commit` is the one that added them, so
    the deployed artifact links to its evidence (SPEC §15.6) even after the file moves."""

    run_id: str
    path: str
    commit: str


@dataclass(frozen=True)
class IndexLock:
    """`index_lock.json` (SPEC §15.6): a committed pointer to a released index, not a binary.

    `embedder` is keyed by vector half (`dense`/`sparse`) rather than naming one model: the
    two agree on every M3 arm, and an e5-dense arm (rung 6, ADR-0022) is exactly the case
    where a single "embedder id" would record half the truth.
    """

    release_tag: str
    built_at: str
    git_commit: str
    corpus_manifest_sha256: str
    embedder: Mapping[str, EmbedderPin]
    chunk_config: Mapping[str, int]
    enrichment: bool
    registers: Mapping[str, RegisterDump]
    ladder_rung: str
    scores: ScoresPointer

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def dump_register(client: QdrantClient, alias: str, path: Path) -> RegisterDump:
    """Write every point behind `alias` to `path` as Parquet and describe what was written.

    Rows are sorted by point id, so the same collection always dumps to the same bytes —
    `asset_sha256` then identifies the point set rather than the order Qdrant happened to
    scroll it in.
    """
    collection = _alias_target(client, alias)
    params = client.get_collection(collection).config.params
    dense_params = _dense_params(params, collection)
    points = sorted(_scroll_all(client, alias), key=lambda point: str(point.id))

    ids: list[str] = []
    dense: list[list[float]] = []
    sparse_indices: list[list[int]] = []
    sparse_values: list[list[float]] = []
    payloads: list[str] = []
    for point in points:
        if not isinstance(point.vector, dict):
            raise ValueError(f"point {point.id} in {collection} carries no named vectors")
        dense_vector = point.vector["dense"]
        sparse_vector = point.vector["sparse"]
        if not isinstance(sparse_vector, models.SparseVector):
            raise ValueError(f"point {point.id} in {collection} has no sparse vector")
        ids.append(str(point.id))
        dense.append(list(dense_vector))  # type: ignore[arg-type]
        sparse_indices.append(sparse_vector.indices)
        sparse_values.append(sparse_vector.values)
        payloads.append(json.dumps(point.payload, ensure_ascii=False))

    table = pa.table(
        {
            "id": pa.array(ids, pa.string()),
            "dense": pa.array(dense, pa.list_(pa.float32(), dense_params.size)),
            "sparse_indices": pa.array(sparse_indices, pa.list_(pa.uint32())),
            "sparse_values": pa.array(sparse_values, pa.list_(pa.float32())),
            "payload": pa.array(payloads, pa.string()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)

    return RegisterDump(
        collection=collection,
        points=len(points),
        asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        vector_config_fingerprint=_vector_config_fingerprint(params, collection),
    )


def _alias_target(client: QdrantClient, alias: str) -> str:
    for entry in client.get_aliases().aliases:
        if entry.alias_name == alias:
            return entry.collection_name
    raise ValueError(f"no collection sits behind the {alias!r} alias — has `make ingest` run?")


def _dense_params(params: models.CollectionParams, collection: str) -> models.VectorParams:
    if not isinstance(params.vectors, dict) or "dense" not in params.vectors:
        raise ValueError(f"{collection} has no named `dense` vector")
    return params.vectors["dense"]


def _scroll_all(client: QdrantClient, alias: str) -> list[models.Record]:
    points: list[models.Record] = []
    offset: models.ExtendedPointId | None = None
    while True:
        page, offset = client.scroll(
            alias, limit=_SCROLL_PAGE, offset=offset, with_payload=True, with_vectors=True
        )
        points.extend(page)
        if offset is None:
            return points


def _vector_config_fingerprint(params: models.CollectionParams, collection: str) -> str:
    """A sha256 over the named-vector layout — dense width and distance, sparse names and
    modifiers. Restore creates its collection from code (SPEC §15.1), so this is what lets
    it notice that code no longer builds the layout these vectors were written into."""
    dense = _dense_params(params, collection)
    sparse = params.sparse_vectors or {}
    layout = {
        "dense": {"size": dense.size, "distance": str(dense.distance)},
        "sparse": {name: str(config.modifier) for name, config in sorted(sparse.items())},
    }
    canonical = json.dumps(layout, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def publish(
    client: QdrantClient,
    *,
    run_path: Path,
    repo_root: Path = REPO_ROOT,
    corpus_manifest_path: Path = DEFAULT_CORPUS_MANIFEST,
    out_dir: Path = DEFAULT_OUT_DIR,
    lock_path: Path = DEFAULT_LOCK_PATH,
    release: ReleaseFn | None = None,
    git_sha: Callable[[Path | None], str] | None = None,
    revision_of: Callable[[str], str] | None = None,
    changed_since: Callable[[str, Sequence[str]], bool] | None = None,
    now: datetime | None = None,
    tag: str | None = None,
) -> IndexLock:
    """Dump the index `run_path` scored, cut the release, then write `index_lock.json`.

    Refuses before dumping anything if an alias points anywhere but the arm that run read —
    a stable alias is only as honest as the last flip, and a rung-6 run interrupted before
    its `finally` would otherwise ship e5 vectors under rung-1 scores. It refuses too if the
    corpus or the chunkers moved since the run's `code_git_sha`, since `chunk_config` and
    `corpus_manifest_sha256` are read now, not recovered from then. The lock is written
    last, only once the release exists: a lock naming a tag with no assets behind it is the
    one state restore (#53) cannot recover from on its own.
    """
    release = release or _gh_release
    git_sha = git_sha or (lambda path: resolve_git_sha(repo_root, path=path))
    revision_of = revision_of or _cached_revision
    changed_since = changed_since or (lambda commit, paths: _changed_since(repo_root, commit, paths))
    now = now or datetime.now(UTC)

    header = load_run(run_path).header
    config = header.retrieval_config
    expected_arms: Mapping[str, str] = config.get("collection_arm") or _DEFAULT_ARMS
    targets = {register: _alias_target(client, alias) for register, alias in _REGISTER_ALIASES.items()}
    wrong = {r: t for r, t in targets.items() if t != expected_arms[r]}
    if wrong:
        found = ", ".join(f"{_REGISTER_ALIASES[r]} -> {t} (scored: {expected_arms[r]})" for r, t in wrong.items())
        raise PublishError(f"{header.run_id} did not score the index behind the aliases: {found}")
    if changed_since(header.code_git_sha, _CHUNKING_PATHS):
        raise PublishError(
            f"the corpus or the chunkers changed since {header.run_id} ran at {header.code_git_sha} — "
            "re-ingest and re-run the ladder before publishing"
        )

    dump_paths = {register: out_dir / f"points-{register}.parquet" for register in _REGISTER_ALIASES}
    registers = {
        register: dump_register(client, alias, dump_paths[register])
        for register, alias in _REGISTER_ALIASES.items()
    }
    release_tag = tag or f"index-{now:%Y-%m-%d}"
    git_commit = git_sha(None)
    dense_id = config.get("dense_embedder") or config["embedder"]
    sparse_id = config.get("sparse_embedder") or config["embedder"]
    lock = IndexLock(
        release_tag=release_tag,
        built_at=now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        git_commit=git_commit,
        corpus_manifest_sha256=hashlib.sha256(corpus_manifest_path.read_bytes()).hexdigest(),
        embedder={
            "dense": EmbedderPin(id=dense_id, revision=revision_of(dense_id)),
            "sparse": EmbedderPin(id=sparse_id, revision=revision_of(sparse_id)),
        },
        chunk_config={"band": BAND, "merge_floor": MERGE_FLOOR, "stub_floor": STUB_FLOOR},
        enrichment=any(dump.collection in _ENRICHED_ARMS for dump in registers.values()),
        registers=registers,
        ladder_rung=header.rung,
        scores=ScoresPointer(
            run_id=header.run_id,
            path=run_path.resolve().relative_to(repo_root.resolve()).as_posix(),
            commit=git_sha(run_path),
        ),
    )

    # The two dumps, then the attribution that must travel with them (SPEC §15.4).
    assets = [*dump_paths.values(), corpus_manifest_path]
    release(release_tag, git_commit, assets, _release_notes(lock))
    lock_path.write_text(json.dumps(lock.as_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return lock


def _release_notes(lock: IndexLock) -> str:
    counts = ", ".join(f"{register}: {dump.points} points" for register, dump in lock.registers.items())
    return (
        f"Points dump of the index scored by `{lock.scores.run_id}` ({lock.ladder_rung}), built from "
        f"{lock.git_commit}. {counts}.\n\n"
        "The payloads carry verbatim DILA text (Code des assurances, service-public.fr fiches), "
        "redistributed under Licence Ouverte 2.0 — `corpus_manifest.json` in this release is the "
        "attribution record. Verify each asset against `asset_sha256` in `index_lock.json`."
    )


def _gh_release(tag: str, target: str, assets: Sequence[Path], notes: str) -> None:
    subprocess.run(
        ["gh", "release", "create", tag, *map(str, assets), "--target", target, "--title", tag, "--notes", notes],
        check=True,
    )


def _changed_since(repo_root: Path, commit: str, paths: Sequence[str]) -> bool:
    diff = subprocess.run(["git", "-C", str(repo_root), "diff", "--quiet", commit, "HEAD", "--", *paths])
    if diff.returncode not in (0, 1):
        raise PublishError(f"git diff against {commit} failed — is that commit in this clone?")
    return diff.returncode == 1


def _cached_revision(model_id: str) -> str:
    """The snapshot the local HF cache resolves `model_id` to — the one the ladder loaded
    from on this machine, unless the cache was refreshed since. Not the Hub's current head,
    which may have moved."""
    ref = _HF_CACHE_DIR / f"models--{model_id.replace('/', '--')}" / "refs" / "main"
    if not ref.exists():
        raise PublishError(f"no cached snapshot of {model_id} under {_HF_CACHE_DIR}")
    return ref.read_text(encoding="utf-8").strip()


def _working_tree_is_clean(repo_root: Path) -> bool:
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"], capture_output=True, text=True, check=True
    )
    return not status.stdout.strip()


def main(argv: Sequence[str] | None = None) -> IndexLock:
    """The `python -m rag.publish` entry point behind `make publish-index`.

    Refuses a dirty working tree: `git_commit` would otherwise name a commit that is not the
    code the dump was taken with — the lock's whole job is to be believed.
    """
    parser = argparse.ArgumentParser(prog="python -m rag.publish", description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="the eval/runs/<run-id>.json that scored the index")
    parser.add_argument("--tag", help="release tag (default: index-<today>)")
    args = parser.parse_args(argv)
    if not _working_tree_is_clean(REPO_ROOT):
        raise PublishError("commit or stash your changes first — git_commit must name the code that ran")
    client = QdrantClient(load_settings().qdrant_url)
    try:
        lock = publish(client, run_path=args.run, tag=args.tag)
    finally:
        client.close()
    print(f"released {lock.release_tag}: " + ", ".join(f"{r} {d.points}" for r, d in lock.registers.items()))
    print(f"now commit it: git add index_lock.json && git commit -m 'chore: index lock {lock.release_tag}'")
    return lock


if __name__ == "__main__":
    main()
