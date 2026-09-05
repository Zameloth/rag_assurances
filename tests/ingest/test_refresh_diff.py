"""SPEC §3.3 / #22 — the corpus refresh diff shared by both fetch scripts."""

from rag.ingest.refresh_diff import RefreshDiff, diff_corpus


def test_first_ingest_reports_everything_added_and_nothing_changed() -> None:
    """`old` empty is the first-ingest shape — every id is added, none changed."""
    diff = diff_corpus({}, {"cid-1": "texte v1", "cid-2": "texte v2"})
    assert diff == RefreshDiff(added=["cid-1", "cid-2"], removed=[], changed=[])


def test_identical_snapshots_report_nothing() -> None:
    same = {"cid-1": "texte v1", "cid-2": "texte v2"}
    diff = diff_corpus(same, dict(same))
    assert diff == RefreshDiff(added=[], removed=[], changed=[])


def test_new_id_is_added() -> None:
    diff = diff_corpus({"cid-1": "texte v1"}, {"cid-1": "texte v1", "cid-2": "texte v2"})
    assert diff.added == ["cid-2"]
    assert diff.removed == []
    assert diff.changed == []


def test_missing_id_is_removed() -> None:
    diff = diff_corpus({"cid-1": "texte v1", "cid-2": "texte v2"}, {"cid-1": "texte v1"})
    assert diff.added == []
    assert diff.removed == ["cid-2"]
    assert diff.changed == []


def test_same_id_different_text_is_changed_not_added_or_removed() -> None:
    """The dangerous case: an amendment under the same `cid` (SPEC §3.3)."""
    diff = diff_corpus({"cid-1": "texte v1"}, {"cid-1": "texte v2 amended"})
    assert diff.added == []
    assert diff.removed == []
    assert diff.changed == ["cid-1"]


def test_added_removed_and_changed_all_report_together() -> None:
    old = {"cid-1": "unchanged", "cid-2": "will be removed", "cid-3": "old text"}
    new = {"cid-1": "unchanged", "cid-3": "new text", "cid-4": "brand new"}
    diff = diff_corpus(old, new)
    assert diff.added == ["cid-4"]
    assert diff.removed == ["cid-2"]
    assert diff.changed == ["cid-3"]


def test_summary_names_all_three_counts() -> None:
    diff = diff_corpus({"cid-1": "a"}, {"cid-1": "b", "cid-2": "c"})
    assert diff.summary() == "added=1 removed=0 changed-text-under-the-same-id=1"


def test_comparison_works_on_bytes_too() -> None:
    """Fiches diff on verbatim XML bytes, not `texte` strings — same function, either type."""
    diff = diff_corpus({"F1": b"<Publication/>"}, {"F1": b"<Publication>changed</Publication>"})
    assert diff.changed == ["F1"]
