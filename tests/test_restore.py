"""`python -m rag.restore` (SPEC §15.5-§15.6, ADR-0014, #53) — `rag.restore`.

The points dump is built the way `make publish-index` builds it — `dump_register` over a
"dev" in-memory store — then restored into a second, empty one standing in for the VPS.
The in-memory client is plumbing-only here (SPEC §6.3): collection names, aliases and
counts, never a ranking.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

import rag.restore as restore_module
from rag.ingest.arms import ensure_articles_collection, ensure_fiches_collection, flip_alias
from rag.publish import EmbedderPin, IndexLock, RegisterDump, ScoresPointer, dump_register
from rag.restore import (
    RegisterOutcome,
    RestoreError,
    RestoreReport,
    generation_name,
    main,
    release_tag_of,
    restore,
)

DIM = 4
TAG = "index-2026-09-28"


def _point(point_id: str, dense: list[float], payload: dict[str, object]) -> models.PointStruct:
    sparse = models.SparseVector(indices=[3, 17, 250002], values=[0.125, 0.3, 0.0421])
    return models.PointStruct(id=point_id, vector={"dense": dense, "sparse": sparse}, payload=payload)


FICHE_POINTS = [
    _point("6f1c2b0e-8a53-5c1e-9f7e-2d4f8a1b3c5d", [0.1, 0.2, 0.3, 0.9273618495495703], {"fiche_id": "F2594"}),
    _point("0a1b2c3d-4e5f-5a6b-8c7d-9e0f1a2b3c4d", [0.7, 0.1, 0.1, 0.6928203230275509], {"fiche_id": "F2595"}),
]
ARTICLE_POINTS = [
    _point(
        "1b2c3d4e-5f6a-5b7c-8d9e-0f1a2b3c4d5e",
        [0.5, 0.5, 0.5, 0.5],
        {"legiarti_cid": "LEGIARTI000006792938", "lookup_key": "L113-12", "section_id": "LEGISCTA1"},
    ),
    _point(
        "2c3d4e5f-6a7b-5c8d-9e0f-1a2b3c4d5e6f",
        [0.9, 0.1, 0.3, 0.2],
        {"legiarti_cid": "LEGIARTI000006792940", "lookup_key": "L113-14", "section_id": "LEGISCTA1"},
    ),
    _point(
        "3d4e5f6a-7b8c-5d9e-8f1a-2b3c4d5e6f7a",
        [0.2, 0.2, 0.9, 0.1],
        {"legiarti_cid": "LEGIARTI000006792950", "lookup_key": None, "section_id": "LEGISCTA2"},
    ),
]


class Release:
    """A GitHub Release standing in for the real one: `download` copies an asset out of
    `dir` and records the call, so a test can say "zero bytes downloaded" literally."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.downloads: list[tuple[str, str]] = []

    def download(self, tag: str, asset: str, dest: Path) -> None:
        self.downloads.append((tag, asset))
        shutil.copyfile(self.dir / asset, dest)


@pytest.fixture
def published(tmp_path: Path) -> tuple[IndexLock, Release]:
    """What `make publish-index` leaves behind: the two dumps on a release, and the lock."""
    dev = QdrantClient(":memory:")
    ensure_fiches_collection(dev, "fiches__m3__c512__v1", dense_dim=DIM)
    ensure_articles_collection(dev, "articles__m3__c512__v1", dense_dim=DIM)
    dev.upsert("fiches__m3__c512__v1", points=FICHE_POINTS)
    dev.upsert("articles__m3__c512__v1", points=ARTICLE_POINTS)
    flip_alias(dev, "fiches", "fiches__m3__c512__v1")
    flip_alias(dev, "articles", "articles__m3__c512__v1")
    release_dir = tmp_path / "release"
    registers = {
        register: dump_register(dev, register, release_dir / f"points-{register}.parquet")
        for register in ("fiches", "articles")
    }
    dev.close()
    return _lock(registers), Release(release_dir)


def _lock(registers: dict[str, RegisterDump], tag: str = TAG) -> IndexLock:
    return IndexLock(
        release_tag=tag,
        built_at="2026-09-28T10:00:00Z",
        git_commit="c0ffee",
        corpus_manifest_sha256="abc123",
        embedder={
            "dense": EmbedderPin(id="BAAI/bge-m3", revision="5617a9f"),
            "sparse": EmbedderPin(id="BAAI/bge-m3", revision="5617a9f"),
        },
        chunk_config={"band": 512, "merge_floor": 100, "stub_floor": 32},
        enrichment=False,
        registers=registers,
        ladder_rung="rung1",
        scores=ScoresPointer(run_id="rung1-x", path="eval/runs/rung1-x.json", commit="48021c9"),
    )


def _with_register(lock: IndexLock, register: str, **changes: object) -> IndexLock:
    registers = dict(lock.registers)
    registers[register] = replace(registers[register], **changes)  # type: ignore[arg-type]
    return replace(lock, registers=registers)


def _aliases(client: QdrantClient) -> dict[str, str]:
    return {a.alias_name: a.collection_name for a in client.get_aliases().aliases}


def _collections(client: QdrantClient) -> set[str]:
    return {c.name for c in client.get_collections().collections}


def _count(client: QdrantClient, collection: str) -> int:
    return client.count(collection, exact=True).count


def _restore(client: QdrantClient, lock: IndexLock, release: Release, tmp_path: Path) -> RestoreReport:
    work_dir = tmp_path / "work"
    work_dir.mkdir(exist_ok=True)
    return restore(client, lock, download=release.download, work_dir=work_dir)


# --- naming -------------------------------------------------------------------------------


def test_a_generation_is_named_after_its_register_and_release_tag() -> None:
    assert generation_name("fiches", "index-2026-08-03") == "fiches__index-2026-08-03"


def test_the_release_tag_reads_back_out_of_a_generation_name() -> None:
    assert release_tag_of("fiches", "fiches__index-2026-08-03") == "index-2026-08-03"


@pytest.mark.parametrize(
    "collection",
    [
        "fiches__m3__c512__v1",  # a dev ladder arm: a collection, never a release
        "articles__index-2026-08-03",  # another register's generation
        "fiches",
    ],
)
def test_a_collection_that_is_not_a_generation_of_the_register_has_no_release_tag(collection: str) -> None:
    assert release_tag_of("fiches", collection) is None


# --- a fresh store ------------------------------------------------------------------------


def test_restore_writes_each_register_into_its_tag_named_generation_and_flips_the_alias(
    published: tuple[IndexLock, Release], qdrant: QdrantClient, tmp_path: Path
) -> None:
    lock, release = published

    report = _restore(qdrant, lock, release, tmp_path)

    assert _aliases(qdrant) == {"fiches": f"fiches__{TAG}", "articles": f"articles__{TAG}"}
    assert _count(qdrant, "fiches") == 2
    assert _count(qdrant, "articles") == 3
    assert {r: o.outcome for r, o in report.registers.items()} == {
        "fiches": RegisterOutcome.DOWNLOADED,
        "articles": RegisterOutcome.DOWNLOADED,
    }


def test_restored_points_carry_the_dumped_vectors_and_payload(
    published: tuple[IndexLock, Release], qdrant: QdrantClient, tmp_path: Path
) -> None:
    lock, release = published

    _restore(qdrant, lock, release, tmp_path)

    [record] = qdrant.retrieve("articles", ids=[ARTICLE_POINTS[0].id], with_vectors=True)
    assert record.payload == ARTICLE_POINTS[0].payload
    assert isinstance(record.vector, dict)
    source = ARTICLE_POINTS[0].vector
    assert isinstance(source, dict)
    assert record.vector["dense"] == pytest.approx(source["dense"])
    sparse = record.vector["sparse"]
    assert isinstance(sparse, models.SparseVector)
    assert sparse.indices == [3, 17, 250002]


def test_restore_creates_each_generation_with_the_code_that_creates_the_dev_arm(
    published: tuple[IndexLock, Release], qdrant: QdrantClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC §15.1: only the vectors are derived — the collection config, payload indexes
    included, is the same code dev runs. Pinned by who is called rather than by reading the
    payload schema back: local mode ignores payload indexes."""
    lock, release = published
    created: list[tuple[str, str]] = []

    def spy(label: str, create: Callable[..., None]) -> Callable[..., None]:
        def wrapped(client: QdrantClient, name: str, **kwargs: int) -> None:
            created.append((label, name))
            create(client, name, **kwargs)

        return wrapped

    monkeypatch.setitem(restore_module._ENSURE_COLLECTION, "fiches", spy("fiches", ensure_fiches_collection))
    monkeypatch.setitem(restore_module._ENSURE_COLLECTION, "articles", spy("articles", ensure_articles_collection))

    _restore(qdrant, lock, release, tmp_path)

    assert sorted(created) == [("articles", f"articles__{TAG}"), ("fiches", f"fiches__{TAG}")]


# --- verification before writing ----------------------------------------------------------


def test_a_checksum_mismatch_fails_before_a_single_point_is_written(
    published: tuple[IndexLock, Release], qdrant: QdrantClient, tmp_path: Path
) -> None:
    """The articles asset is the one that is wrong — fiches, checked first and fine, must
    not have been written either."""
    lock, release = published
    lock = _with_register(lock, "articles", asset_sha256="0" * 64)

    with pytest.raises(RestoreError, match="articles"):
        _restore(qdrant, lock, release, tmp_path)

    assert _collections(qdrant) == set()
    assert _aliases(qdrant) == {}


def test_a_layout_the_code_no_longer_builds_fails_before_a_point_is_written(
    published: tuple[IndexLock, Release], qdrant: QdrantClient, tmp_path: Path
) -> None:
    lock, release = published
    lock = _with_register(lock, "fiches", vector_config_fingerprint="f" * 64)

    with pytest.raises(RestoreError, match="fingerprint"):
        _restore(qdrant, lock, release, tmp_path)

    assert _aliases(qdrant) == {}
    assert all(_count(qdrant, name) == 0 for name in _collections(qdrant))


def test_a_post_upsert_count_short_of_the_lock_never_moves_the_alias(
    published: tuple[IndexLock, Release], qdrant: QdrantClient, tmp_path: Path
) -> None:
    lock, release = published
    old = "articles__index-2026-08-03"
    ensure_articles_collection(qdrant, old, dense_dim=DIM)
    flip_alias(qdrant, "articles", old)
    lock = _with_register(lock, "articles", points=4)

    with pytest.raises(RestoreError, match="articles"):
        _restore(qdrant, lock, release, tmp_path)

    assert _aliases(qdrant)["articles"] == old
    assert "fiches" not in _aliases(qdrant), "no alias moves unless every register verified"


# --- idempotence, retention, rollback -----------------------------------------------------


def test_restore_is_a_no_op_when_the_tag_and_count_already_match(
    published: tuple[IndexLock, Release], qdrant: QdrantClient, tmp_path: Path
) -> None:
    lock, release = published
    _restore(qdrant, lock, release, tmp_path)
    release.downloads.clear()

    report = _restore(qdrant, lock, release, tmp_path)

    assert release.downloads == []
    assert {o.outcome for o in report.registers.values()} == {RegisterOutcome.ALREADY_LIVE}


def test_a_retained_generation_goes_live_with_zero_bytes_downloaded(
    published: tuple[IndexLock, Release], qdrant: QdrantClient, tmp_path: Path
) -> None:
    """Rollback: the previous generation is still on disk, so going back to its lock is
    one alias flip per register."""
    lock, release = published
    _restore(qdrant, lock, release, tmp_path)
    for register in ("fiches", "articles"):
        ensure_fiches_collection(qdrant, f"{register}__index-2026-10-05", dense_dim=DIM)
        flip_alias(qdrant, register, f"{register}__index-2026-10-05")
    release.downloads.clear()

    report = _restore(qdrant, lock, release, tmp_path)

    assert release.downloads == []
    assert _aliases(qdrant) == {"fiches": f"fiches__{TAG}", "articles": f"articles__{TAG}"}
    assert {o.outcome for o in report.registers.values()} == {RegisterOutcome.RETAINED}


def test_a_partial_generation_left_by_a_killed_run_is_rewritten_not_trusted(
    published: tuple[IndexLock, Release], qdrant: QdrantClient, tmp_path: Path
) -> None:
    lock, release = published
    ensure_articles_collection(qdrant, f"articles__{TAG}", dense_dim=DIM)
    qdrant.upsert(f"articles__{TAG}", points=ARTICLE_POINTS[:1])

    report = _restore(qdrant, lock, release, tmp_path)

    assert _count(qdrant, "articles") == 3
    assert report.registers["articles"].outcome is RegisterOutcome.DOWNLOADED


def test_a_live_generation_whose_count_disagrees_with_the_lock_fails_loud(
    published: tuple[IndexLock, Release], qdrant: QdrantClient, tmp_path: Path
) -> None:
    """Restore never flips onto a short collection, so this is a store someone edited by
    hand — rewriting it under a live alias is not restore's call to make."""
    lock, release = published
    _restore(qdrant, lock, release, tmp_path)
    qdrant.delete(f"fiches__{TAG}", points_selector=models.PointIdsList(points=[FICHE_POINTS[0].id]))

    with pytest.raises(RestoreError, match="fiches"):
        _restore(qdrant, lock, release, tmp_path)


def test_pruning_keeps_the_live_and_the_previous_generation_only(
    published: tuple[IndexLock, Release], qdrant: QdrantClient, tmp_path: Path
) -> None:
    lock, release = published
    for tag in ("index-2026-06-01", "index-2026-07-01", "index-2026-08-03"):
        ensure_fiches_collection(qdrant, f"fiches__{tag}", dense_dim=DIM)
        ensure_articles_collection(qdrant, f"articles__{tag}", dense_dim=DIM)
    flip_alias(qdrant, "fiches", "fiches__index-2026-08-03")
    flip_alias(qdrant, "articles", "articles__index-2026-08-03")
    # A dev ladder arm shares the register prefix and is not a generation.
    ensure_fiches_collection(qdrant, "fiches__m3__c512__v1", dense_dim=DIM)

    report = _restore(qdrant, lock, release, tmp_path)

    assert _collections(qdrant) == {
        f"fiches__{TAG}",
        "fiches__index-2026-08-03",
        f"articles__{TAG}",
        "articles__index-2026-08-03",
        "fiches__m3__c512__v1",
    }
    assert set(report.pruned) == {
        "fiches__index-2026-06-01",
        "fiches__index-2026-07-01",
        "articles__index-2026-06-01",
        "articles__index-2026-07-01",
    }


def test_after_a_rollback_pruning_keeps_the_newer_generation_to_roll_forward_to(
    published: tuple[IndexLock, Release], qdrant: QdrantClient, tmp_path: Path
) -> None:
    lock, release = published
    _restore(qdrant, lock, release, tmp_path)
    for tag in ("index-2026-10-05", "index-2026-06-01"):
        ensure_fiches_collection(qdrant, f"fiches__{tag}", dense_dim=DIM)

    _restore(qdrant, lock, release, tmp_path)

    assert {c for c in _collections(qdrant) if c.startswith("fiches__")} == {
        f"fiches__{TAG}",
        "fiches__index-2026-10-05",
    }


# --- the entry point ----------------------------------------------------------------------


def test_main_installs_the_lock_it_is_given_from_the_configured_release_url(
    published: tuple[IndexLock, Release], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over the real download path: `file://` stands in for the GitHub Release base URL,
    laid out `<base>/<tag>/<asset>` the way release downloads are."""
    lock, release = published
    tag_dir = tmp_path / "releases" / TAG
    shutil.copytree(release.dir, tag_dir)
    lock_path = tmp_path / "index_lock.json"
    lock_path.write_text(json.dumps(lock.as_dict()), encoding="utf-8")
    monkeypatch.setenv("INDEX_RELEASE_URL", (tmp_path / "releases").as_uri())
    monkeypatch.setenv("QDRANT_URL", ":memory:")

    report = main(["--lock", str(lock_path)])

    assert {r: o.outcome for r, o in report.registers.items()} == {
        "fiches": RegisterOutcome.DOWNLOADED,
        "articles": RegisterOutcome.DOWNLOADED,
    }
