"""`make publish-index` (SPEC §15.2-§15.6, ADR-0014, #52) — `rag.publish`.

Three seams: `dump_register` (an aliased collection -> one Parquet file), `build_index_lock`
(the §15.6 record, pure) and `publish` (the orchestration, against a fake release runner).
The in-memory client is plumbing-only here (SPEC §6.3) — nothing below asserts a ranking.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from qdrant_client import QdrantClient, models

from rag.ingest.arms import ensure_articles_collection, ensure_fiches_collection, flip_alias
from rag.publish import (
    EmbedderPin,
    IndexLock,
    PublishError,
    RegisterDump,
    ScoresPointer,
    dump_register,
    publish,
)

DIM = 4

# Chosen so none is exact in binary: a value that survived a text or float64 detour with
# one ulp of drift would fail the equality below, which is the whole §15.2 argument.
DENSE_A = [0.1, 0.2, 0.3, 0.9273618495495703]
DENSE_B = [0.7, 0.1, 0.1, 0.6928203230275509]


def _point(point_id: str, dense: list[float], payload: dict[str, object]) -> models.PointStruct:
    sparse = models.SparseVector(indices=[3, 17, 250002], values=[0.125, 0.3, 0.0421])
    return models.PointStruct(id=point_id, vector={"dense": dense, "sparse": sparse}, payload=payload)


@pytest.fixture
def fiches_arm(qdrant: QdrantClient) -> QdrantClient:
    ensure_fiches_collection(qdrant, "fiches__m3__c512__v1", dense_dim=DIM)
    qdrant.upsert(
        "fiches__m3__c512__v1",
        points=[
            _point(
                "6f1c2b0e-8a53-5c1e-9f7e-2d4f8a1b3c5d",
                DENSE_A,
                {"fiche_id": "F2594", "text": "Si vous êtes locataire…", "section_ids": ["LEGISCTA1"]},
            ),
            _point(
                "0a1b2c3d-4e5f-5a6b-8c7d-9e0f1a2b3c4d",
                DENSE_B,
                {"fiche_id": "F2595", "text": "Assurance « habitation »", "cas_label": None},
            ),
        ],
    )
    flip_alias(qdrant, "fiches", "fiches__m3__c512__v1")
    return qdrant


def test_dump_register_writes_one_row_per_point_with_the_spec_columns(
    fiches_arm: QdrantClient, tmp_path: Path
) -> None:
    path = tmp_path / "points-fiches.parquet"

    dump = dump_register(fiches_arm, "fiches", path)

    table = pq.read_table(path)
    assert dump.points == 2 == table.num_rows
    assert table.schema.field("id").type == pa.string()
    assert table.schema.field("dense").type == pa.list_(pa.float32(), DIM)
    assert table.schema.field("sparse_indices").type == pa.list_(pa.uint32())
    assert table.schema.field("sparse_values").type == pa.list_(pa.float32())
    assert table.schema.field("payload").type == pa.string()


def test_dump_register_names_the_physical_collection_behind_the_alias(
    fiches_arm: QdrantClient, tmp_path: Path
) -> None:
    dump = dump_register(fiches_arm, "fiches", tmp_path / "points-fiches.parquet")

    assert dump.collection == "fiches__m3__c512__v1"


def test_dump_register_vectors_are_bit_identical_to_what_qdrant_serves(
    fiches_arm: QdrantClient, tmp_path: Path
) -> None:
    path = tmp_path / "points-fiches.parquet"
    dump_register(fiches_arm, "fiches", path)

    rows = {row["id"]: row for row in pq.read_table(path).to_pylist()}
    served = fiches_arm.retrieve("fiches", ids=list(rows), with_vectors=True)
    for record in served:
        assert isinstance(record.vector, dict)
        dense, sparse = record.vector["dense"], record.vector["sparse"]
        assert isinstance(sparse, models.SparseVector)
        row = rows[str(record.id)]
        # Both sides as float32 bytes: an equality on Python floats would hide a drift the
        # f32 cast then rounds back.
        assert pa.array(row["dense"], pa.float32()).buffers()[1] == pa.array(dense, pa.float32()).buffers()[1]
        assert row["sparse_indices"] == sparse.indices
        assert pa.array(row["sparse_values"], pa.float32()).buffers()[1] == pa.array(
            sparse.values, pa.float32()
        ).buffers()[1]


def test_dump_register_keeps_the_payload_verbatim_as_json(fiches_arm: QdrantClient, tmp_path: Path) -> None:
    path = tmp_path / "points-fiches.parquet"
    dump_register(fiches_arm, "fiches", path)

    rows = {row["id"]: row for row in pq.read_table(path).to_pylist()}
    payload = json.loads(rows["0a1b2c3d-4e5f-5a6b-8c7d-9e0f1a2b3c4d"]["payload"])
    assert payload == {"fiche_id": "F2595", "text": "Assurance « habitation »", "cas_label": None}
    # Not escaped to «: the payload is French text and the file is never reviewed, but
    # a restore that re-parses it must get the same characters back, not an equivalent.
    assert "« habitation »" in rows["0a1b2c3d-4e5f-5a6b-8c7d-9e0f1a2b3c4d"]["payload"]


def test_dump_register_reports_the_sha256_of_the_file_it_wrote(
    fiches_arm: QdrantClient, tmp_path: Path
) -> None:
    path = tmp_path / "points-fiches.parquet"

    dump = dump_register(fiches_arm, "fiches", path)

    assert dump.asset_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_dump_register_is_byte_deterministic(fiches_arm: QdrantClient, tmp_path: Path) -> None:
    first = dump_register(fiches_arm, "fiches", tmp_path / "a.parquet")
    second = dump_register(fiches_arm, "fiches", tmp_path / "b.parquet")

    assert first.asset_sha256 == second.asset_sha256


def test_vector_config_fingerprint_moves_with_the_dense_width(qdrant: QdrantClient, tmp_path: Path) -> None:
    ensure_articles_collection(qdrant, "narrow", dense_dim=DIM)
    ensure_articles_collection(qdrant, "wide", dense_dim=DIM * 2)
    flip_alias(qdrant, "narrow_alias", "narrow")
    flip_alias(qdrant, "wide_alias", "wide")

    narrow = dump_register(qdrant, "narrow_alias", tmp_path / "n.parquet")
    wide = dump_register(qdrant, "wide_alias", tmp_path / "w.parquet")

    assert narrow.vector_config_fingerprint != wide.vector_config_fingerprint
    assert narrow.points == 0


def _lock() -> IndexLock:
    return IndexLock(
        release_tag="index-2026-09-28",
        built_at="2026-09-28T10:00:00Z",
        git_commit="c0ffee",
        corpus_manifest_sha256="abc123",
        embedder={
            "dense": EmbedderPin(id="BAAI/bge-m3", revision="5617a9f"),
            "sparse": EmbedderPin(id="BAAI/bge-m3", revision="5617a9f"),
        },
        chunk_config={"band": 512, "merge_floor": 100, "stub_floor": 32},
        enrichment=False,
        registers={
            "fiches": RegisterDump(
                collection="fiches__m3__c512__v1", points=849, asset_sha256="f1", vector_config_fingerprint="v1"
            ),
            "articles": RegisterDump(
                collection="articles__m3__c512__v1", points=2801, asset_sha256="a1", vector_config_fingerprint="v2"
            ),
        },
        ladder_rung="rung1",
        scores=ScoresPointer(
            run_id="rung1-20260916T181814Z", path="eval/runs/rung1-20260916T181814Z.json", commit="48021c9"
        ),
    )


def test_index_lock_serializes_every_spec_15_6_field() -> None:
    assert _lock().as_dict() == {
        "release_tag": "index-2026-09-28",
        "built_at": "2026-09-28T10:00:00Z",
        "git_commit": "c0ffee",
        "corpus_manifest_sha256": "abc123",
        "embedder": {
            "dense": {"id": "BAAI/bge-m3", "revision": "5617a9f"},
            "sparse": {"id": "BAAI/bge-m3", "revision": "5617a9f"},
        },
        "chunk_config": {"band": 512, "merge_floor": 100, "stub_floor": 32},
        "enrichment": False,
        "registers": {
            "fiches": {
                "collection": "fiches__m3__c512__v1",
                "points": 849,
                "asset_sha256": "f1",
                "vector_config_fingerprint": "v1",
            },
            "articles": {
                "collection": "articles__m3__c512__v1",
                "points": 2801,
                "asset_sha256": "a1",
                "vector_config_fingerprint": "v2",
            },
        },
        "ladder_rung": "rung1",
        "scores": {
            "run_id": "rung1-20260916T181814Z",
            "path": "eval/runs/rung1-20260916T181814Z.json",
            "commit": "48021c9",
        },
    }


NOW = datetime(2026, 9, 28, 10, 0, tzinfo=UTC)
HEAD = "1111111111111111111111111111111111111111"
SCORES_COMMIT = "2222222222222222222222222222222222222222"


class FakeRelease:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[str, str, list[Path]]] = []
        self.fail = fail

    def __call__(self, tag: str, target: str, assets: Sequence[Path], notes: str) -> None:
        self.calls.append((tag, target, list(assets)))
        if self.fail:
            raise RuntimeError("gh release create exited 1")


def _git_sha(path: Path | None) -> str:
    return HEAD if path is None else SCORES_COMMIT


def _write_run(tmp_path: Path, retrieval_config: dict[str, object]) -> Path:
    header = {
        "run_id": "rung1-20260916T181814Z",
        "rung": "rung1",
        "arm": "rung1",
        "golden_set_git_sha": "g",
        "langfuse_dataset_version": "v",
        "retrieval_config": retrieval_config,
        "code_git_sha": "c",
        "timestamp": "2026-09-16T18:18:25+00:00",
        "langfuse_run_name": "rung1-20260916T181814Z",
    }
    path = tmp_path / "rung1-20260916T181814Z.json"
    path.write_text(json.dumps({"header": header, "items": []}), encoding="utf-8")
    return path


@pytest.fixture
def scored_index(qdrant: QdrantClient) -> QdrantClient:
    """Both default M3 arms, behind their aliases — the index every rung but 6 read."""
    for name, ensure in (
        ("articles__m3__c512__v1", ensure_articles_collection),
        ("fiches__m3__c512__v1", ensure_fiches_collection),
    ):
        ensure(qdrant, name, dense_dim=DIM)
        qdrant.upsert(name, points=[_point("6f1c2b0e-8a53-5c1e-9f7e-2d4f8a1b3c5d", DENSE_A, {"k": name})])
        flip_alias(qdrant, name.split("__")[0], name)
    return qdrant


def _publish(client: QdrantClient, tmp_path: Path, run: Path, release: FakeRelease) -> IndexLock:
    manifest = tmp_path / "corpus_manifest.json"
    manifest.write_text('{"articles": {}}\n', encoding="utf-8")
    return publish(
        client,
        run_path=run,
        repo_root=tmp_path,
        corpus_manifest_path=manifest,
        out_dir=tmp_path / "out",
        lock_path=tmp_path / "index_lock.json",
        release=release,
        git_sha=_git_sha,
        revision_of=lambda model_id: f"rev-of-{model_id}",
        now=NOW,
    )


def test_publish_cuts_the_release_with_both_dumps_and_the_corpus_manifest(
    scored_index: QdrantClient, tmp_path: Path
) -> None:
    release = FakeRelease()

    _publish(scored_index, tmp_path, _write_run(tmp_path, {"embedder": "BAAI/bge-m3"}), release)

    [(tag, target, assets)] = release.calls
    assert tag == "index-2026-09-28"
    assert target == HEAD
    assert [asset.name for asset in assets] == [
        "points-fiches.parquet",
        "points-articles.parquet",
        "corpus_manifest.json",
    ]
    assert all(asset.exists() for asset in assets)


def test_publish_writes_the_lock_linking_corpus_scores_and_artifact(
    scored_index: QdrantClient, tmp_path: Path
) -> None:
    _publish(scored_index, tmp_path, _write_run(tmp_path, {"embedder": "BAAI/bge-m3"}), FakeRelease())

    lock = json.loads((tmp_path / "index_lock.json").read_text(encoding="utf-8"))
    assert lock["release_tag"] == "index-2026-09-28"
    assert lock["built_at"] == "2026-09-28T10:00:00Z"
    assert lock["git_commit"] == HEAD
    assert lock["corpus_manifest_sha256"] == hashlib.sha256(b'{"articles": {}}\n').hexdigest()
    assert lock["embedder"]["dense"] == {"id": "BAAI/bge-m3", "revision": "rev-of-BAAI/bge-m3"}
    assert lock["chunk_config"] == {"band": 512, "merge_floor": 100, "stub_floor": 32}
    assert lock["enrichment"] is False
    assert lock["registers"]["articles"]["collection"] == "articles__m3__c512__v1"
    assert lock["registers"]["fiches"]["points"] == 1
    assert lock["registers"]["fiches"]["asset_sha256"] == hashlib.sha256(
        (tmp_path / "out" / "points-fiches.parquet").read_bytes()
    ).hexdigest()
    assert lock["ladder_rung"] == "rung1"
    assert lock["scores"] == {
        "run_id": "rung1-20260916T181814Z",
        "path": "rung1-20260916T181814Z.json",
        "commit": SCORES_COMMIT,
    }


def test_publish_refuses_an_alias_that_is_not_the_arm_the_run_scored(
    scored_index: QdrantClient, tmp_path: Path
) -> None:
    # Left flipped to the e5 arm after an interrupted rung-6 run: publishing now would ship
    # an index the rung-1 scores never measured.
    ensure_fiches_collection(scored_index, "fiches__e5-m3__c512__v1", dense_dim=DIM)
    flip_alias(scored_index, "fiches", "fiches__e5-m3__c512__v1")
    release = FakeRelease()

    with pytest.raises(PublishError, match="fiches__e5-m3__c512__v1"):
        _publish(scored_index, tmp_path, _write_run(tmp_path, {"embedder": "BAAI/bge-m3"}), release)

    assert release.calls == []
    assert not (tmp_path / "index_lock.json").exists()


def test_publish_follows_the_arm_a_run_header_names(scored_index: QdrantClient, tmp_path: Path) -> None:
    for register in ("articles", "fiches"):
        name = f"{register}__e5-m3__c512__v1"
        ensure_fiches_collection(scored_index, name, dense_dim=DIM)
        flip_alias(scored_index, register, name)
    run = _write_run(
        tmp_path,
        {
            "embedder": "BAAI/bge-m3",
            "dense_embedder": "intfloat/multilingual-e5-large-instruct",
            "sparse_embedder": "BAAI/bge-m3",
            "collection_arm": {"articles": "articles__e5-m3__c512__v1", "fiches": "fiches__e5-m3__c512__v1"},
        },
    )

    lock = _publish(scored_index, tmp_path, run, FakeRelease())

    assert lock.registers["fiches"].collection == "fiches__e5-m3__c512__v1"
    assert lock.embedder["dense"].id == "intfloat/multilingual-e5-large-instruct"
    assert lock.embedder["sparse"].id == "BAAI/bge-m3"


def test_publish_writes_no_lock_when_the_release_fails(scored_index: QdrantClient, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        _publish(
            scored_index, tmp_path, _write_run(tmp_path, {"embedder": "BAAI/bge-m3"}), FakeRelease(fail=True)
        )

    # A lock naming a release that does not exist would send restore after a 404.
    assert not (tmp_path / "index_lock.json").exists()
