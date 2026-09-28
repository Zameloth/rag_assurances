"""`python -m rag.restore`: the released points dump -> Qdrant, made live by an alias flip
(SPEC §15.5-§15.6, ADR-0014, #53).

    docker compose run --rm rag-assurances python -m rag.restore

**The app image is the tool** — no repo checkout, no Python and no qdrant-client on the
host. It reads the `index_lock.json` baked into the image, so the index installed is
always the one the code was committed against.

**A deploy step, never a boot step.** Nothing here runs on wake: Qdrant's volume outlives
sleep, and a boot-time fetch would put `github.com` in a visitor's cold start.

Each register goes into its own **generation**, `<register>__<release-tag>`, and the order
is what makes the process safe to kill at any point:

1. **sha256 of every downloaded asset against the lock, before a single point is written**
   — for both registers, so a bad articles asset leaves fiches untouched too;
2. the collection is created by the same code that creates the dev arm (only the vectors
   are derived — SPEC §15.1), and its layout fingerprint checked against the lock;
3. upsert, then the **exact count, before any alias moves**;
4. both aliases flip only once every register has verified — a partial write can never
   wear a valid name, so a mid-run kill leaves the old index serving and the fix is to
   re-run;
5. prune to the last two generations per register.

**Idempotent**: a generation whose tag and count match the lock is complete (point ids are
unique and each upsert is atomic per point), so it is flipped to — or left live — without
downloading a byte. That is also **rollback**: deploy the previous image, whose lock names
the previous tag, and restore; the retained generation goes live with one alias flip.
"""

from __future__ import annotations

import argparse
import enum
import hashlib
import json
import shutil
import tempfile
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models

from rag.config import load_settings
from rag.ingest.arms import (
    ARTICLES_ALIAS,
    FICHES_ALIAS,
    ensure_articles_collection,
    ensure_fiches_collection,
    flip_alias,
)
from rag.ingest.upsert import UPSERT_BATCH_SIZE
from rag.publish import DEFAULT_LOCK_PATH, IndexLock, load_index_lock, vector_config_fingerprint

__all__ = [
    "DownloadFn",
    "RegisterOutcome",
    "RegisterRestore",
    "RestoreError",
    "RestoreReport",
    "generation_name",
    "main",
    "release_tag_of",
    "restore",
]

# `(release tag, asset name, destination)` — raises if the asset could not be fetched.
DownloadFn = Callable[[str, str, Path], None]

# Keyed by the plural collection name, as the lock's registers and the asset names are.
_REGISTER_ALIASES = {"fiches": FICHES_ALIAS, "articles": ARTICLES_ALIAS}
_ENSURE_COLLECTION = {"fiches": ensure_fiches_collection, "articles": ensure_articles_collection}

# `make publish-index` tags `index-<date>`, so generation names sort by time. Pruning only
# ever touches names of this shape: a dev ladder arm (`fiches__m3__c512__v1`) shares the
# register prefix and must never be read as an old release.
_GENERATION_TAG_PREFIX = "index-"
_GENERATIONS_KEPT = 2


class RestoreError(Exception):
    """The index on offer is not provably the one the lock names — nothing was flipped."""


class RegisterOutcome(enum.StrEnum):
    """Which path a register took — recorded, since "restore did nothing" and "restore
    rolled back" look the same from the alias alone."""

    ALREADY_LIVE = "already_live"  # tag and count matched the live alias: a no-op
    RETAINED = "retained"  # a complete generation was on disk: flipped, zero bytes downloaded
    DOWNLOADED = "downloaded"  # fetched, verified, written, verified


@dataclass(frozen=True)
class RegisterRestore:
    collection: str
    points: int
    outcome: RegisterOutcome


@dataclass(frozen=True)
class RestoreReport:
    release_tag: str
    registers: Mapping[str, RegisterRestore]
    pruned: tuple[str, ...]


def generation_name(register: str, release_tag: str) -> str:
    """SPEC §15.5 — short, time-ordered, traceable to its release. The config lives in the
    lock, so the name is a key, not a description."""
    return f"{register}__{release_tag}"


def release_tag_of(register: str, collection: str) -> str | None:
    """The release tag a generation name carries, or `None` if `collection` is not a
    generation of `register` — how `/health` checks the alias against the lock for free."""
    prefix = f"{register}__"
    tag = collection.removeprefix(prefix)
    if tag == collection or not tag.startswith(_GENERATION_TAG_PREFIX):
        return None
    return tag


def restore(client: QdrantClient, lock: IndexLock, *, download: DownloadFn, work_dir: Path) -> RestoreReport:
    """Make the index `lock` names the one behind the `fiches`/`articles` aliases. See the
    module docstring for the order and why each step sits where it does."""
    aliases = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}
    existing = {c.name for c in client.get_collections().collections}

    outcomes: dict[str, RegisterOutcome] = {}
    for register, alias in _REGISTER_ALIASES.items():
        target = generation_name(register, lock.release_tag)
        expected = lock.registers[register].points
        complete = target in existing and _count(client, target) == expected
        if aliases.get(alias) == target and not complete:
            raise RestoreError(
                f"{target} is live but does not hold the {expected} points the lock names — "
                "restore never flips onto a short collection, so it was changed by hand; "
                "inspect it before re-running"
            )
        if aliases.get(alias) == target:
            outcomes[register] = RegisterOutcome.ALREADY_LIVE
        elif complete:
            outcomes[register] = RegisterOutcome.RETAINED
        else:
            outcomes[register] = RegisterOutcome.DOWNLOADED

    to_write = [r for r, outcome in outcomes.items() if outcome is RegisterOutcome.DOWNLOADED]
    assets = {register: _fetch_verified(lock, register, download, work_dir) for register in to_write}
    for register, asset in assets.items():
        _write_generation(client, lock, register, asset, leftover=generation_name(register, lock.release_tag) in existing)

    for register, alias in _REGISTER_ALIASES.items():
        flip_alias(client, alias, generation_name(register, lock.release_tag))

    return RestoreReport(
        release_tag=lock.release_tag,
        registers={
            register: RegisterRestore(
                collection=generation_name(register, lock.release_tag),
                points=lock.registers[register].points,
                outcome=outcome,
            )
            for register, outcome in outcomes.items()
        },
        pruned=tuple(name for register in _REGISTER_ALIASES for name in _prune(client, register)),
    )


def _count(client: QdrantClient, collection: str) -> int:
    return client.count(collection, exact=True).count


def _fetch_verified(lock: IndexLock, register: str, download: DownloadFn, work_dir: Path) -> Path:
    asset = f"points-{register}.parquet"
    path = work_dir / asset
    download(lock.release_tag, asset, path)
    actual = _sha256(path)
    expected = lock.registers[register].asset_sha256
    if actual != expected:
        raise RestoreError(
            f"{asset} from {lock.release_tag} has sha256 {actual}, the lock says {expected} — "
            "nothing was written"
        )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_generation(client: QdrantClient, lock: IndexLock, register: str, asset: Path, *, leftover: bool) -> None:
    """Create the generation from code, check its layout, upsert, check the count.

    A generation that already exists here is incomplete (a complete one is `RETAINED`) —
    the leftover of a killed run. It is dropped rather than topped up: its points are a
    subset of the same asset's, but "whatever a killed run left" is not a state worth
    reasoning about when rewriting it costs seconds."""
    collection = generation_name(register, lock.release_tag)
    table = pq.read_table(asset)
    if leftover:
        client.delete_collection(collection)
    _ENSURE_COLLECTION[register](client, collection, dense_dim=table.schema.field("dense").type.list_size)

    expected = lock.registers[register]
    fingerprint = vector_config_fingerprint(client.get_collection(collection).config.params, collection)
    if fingerprint != expected.vector_config_fingerprint:
        client.delete_collection(collection)
        raise RestoreError(
            f"{collection}: the code builds a vector layout whose fingerprint is {fingerprint}, "
            f"the dump was written into {expected.vector_config_fingerprint} — this image cannot "
            f"serve {lock.release_tag}"
        )

    for batch in _points(table):
        client.upsert(collection, points=batch)

    written = _count(client, collection)
    if written != expected.points:
        raise RestoreError(
            f"{collection} holds {written} points after the upsert, the lock says {expected.points} — "
            "no alias was moved"
        )


def _points(table: pa.Table) -> Iterator[list[models.PointStruct]]:
    """The dump's rows as points, `UPSERT_BATCH_SIZE` at a time — the same ~12.5 MB request
    ceiling ingest sizes its batches against (`rag.ingest.upsert`)."""
    for batch in table.to_batches(max_chunksize=UPSERT_BATCH_SIZE):
        yield [
            models.PointStruct(
                id=row["id"],
                vector={
                    "dense": row["dense"],
                    "sparse": models.SparseVector(indices=row["sparse_indices"], values=row["sparse_values"]),
                },
                payload=json.loads(row["payload"]),
            )
            for row in batch.to_pylist()
        ]


def _prune(client: QdrantClient, register: str) -> list[str]:
    """Keep the live generation and the newest other one; drop the rest.

    "The newest other", not "the one before": after a rollback the live generation is the
    older one, and the newer is what rolling forward again needs."""
    live = next(
        (a.collection_name for a in client.get_aliases().aliases if a.alias_name == _REGISTER_ALIASES[register]),
        None,
    )
    generations = sorted(
        (c.name for c in client.get_collections().collections if release_tag_of(register, c.name)),
        reverse=True,
    )
    others = [name for name in generations if name != live]
    doomed = others[_GENERATIONS_KEPT - 1 :]
    for name in doomed:
        client.delete_collection(name)
    return doomed


def _https_download(base_url: str) -> DownloadFn:
    def download(tag: str, asset: str, dest: Path) -> None:
        with urllib.request.urlopen(f"{base_url}/{tag}/{asset}", timeout=60) as response, dest.open("wb") as out:
            shutil.copyfileobj(response, out)

    return download


def main(argv: Sequence[str] | None = None) -> RestoreReport:
    """The `python -m rag.restore` entry point `make deploy` runs inside the app image."""
    parser = argparse.ArgumentParser(prog="python -m rag.restore", description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH, help="index_lock.json to install")
    args = parser.parse_args(argv)
    settings = load_settings()
    lock = load_index_lock(args.lock)
    client = QdrantClient(settings.qdrant_url)
    try:
        with tempfile.TemporaryDirectory(prefix="rag-restore-") as work_dir:
            report = restore(
                client, lock, download=_https_download(settings.index_release_url), work_dir=Path(work_dir)
            )
    finally:
        client.close()
    for register, done in report.registers.items():
        print(f"{register}: {done.collection} ({done.points} points) — {done.outcome.value}")
    if report.pruned:
        print("pruned: " + ", ".join(report.pruned))
    return report


if __name__ == "__main__":
    main()
