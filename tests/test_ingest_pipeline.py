"""`make ingest` end to end (SPEC §4-§7, #26) — `rag.ingest.pipeline`, stub embedder,
against the real committed corpus.

The point count is a property of chunking and upsert, never of what the embedder returns
(the same `Embedding` tuple shape lands on a point whether it came from BGE-M3 or a stub),
so running the real corpus through a stub embedder is exactly as informative as the real
~2.3 GB model here and several minutes cheaper. `test_embedder.py` covers the BGE-M3
wrapper itself; this file covers the orchestration around it.

The real-corpus run is expensive (chunking 2,377 articles + 87 fiches, twice — once for
the assertion gate, once for upsert) and shared via a module-scoped fixture so the suite
pays it once, not once per assertion; `main`'s own wiring is instead checked against a
two-document fixture corpus, since that behaviour has nothing to do with the real corpus's
size.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from rag.ingest.arms import DENSE_DIM
from rag.ingest.pipeline import IngestReport, main, run_ingest
from rag.ingest.upsert import Embedding

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTICLES_PATH = REPO_ROOT / "data" / "corpus" / "articles.jsonl"
FICHES_DIR = REPO_ROOT / "data" / "corpus" / "fiches"

# See rag.ingest.pipeline's module docstring — short of SPEC §4.4's documented 3,687
# (2,805 + 882), a known and already-accepted #23/#24 discrepancy this ticket reports
# rather than re-opens.
MEASURED_ARTICLE_POINTS = 2801
MEASURED_FICHE_POINTS = 849
MEASURED_TOTAL_POINTS = MEASURED_ARTICLE_POINTS + MEASURED_FICHE_POINTS

_FICHE_XML = (
    b'<?xml version="1.0" encoding="UTF-8"?>'
    b'<Publication xmlns:dc="http://purl.org/dc/elements/1.1/" ID="F1" '
    b'type="Fiche d\'information conditionn\xc3\xa9e" '
    b'spUrl="https://www.service-public.gouv.fr/particuliers/vosdroits/F1">'
    b"<dc:title>Titre de test</dc:title>"
    b"<dc:date>modified 2025-04-28</dc:date>"
    b"<dc:source>https://www.legifrance.gouv.fr/codes/id/LEGISCTA000000000099</dc:source>"
    b'<FilDAriane><Niveau ID="Particuliers">Accueil particuliers</Niveau></FilDAriane>'
    b"<Texte><Paragraphe>La modification du contrat peut \xc3\xaatre demand\xc3\xa9e "
    b"par l'assur\xc3\xa9 \xc3\xa0 tout moment de la vie du contrat d'assurance "
    b"habitation, notamment en cas de changement de situation personnelle ou "
    b"familiale.</Paragraphe></Texte>"
    b"</Publication>"
)

_ARTICLE_ROW = {
    "cid": "LEGIARTI000000000001",
    "id": "LEGIARTI000000000002",
    "citation_id": "L113-1",
    "texte": "placeholder",
    "texteHtml": (
        "<p>Le contrat d'assurance est régi par les dispositions du présent titre "
        "et les stipulations particulières.</p>"
    ),
    "etat": "VIGUEUR",
    "dateDebut": "2018-04-01",
    "sectionParentId": "LEGISCTA000000000099",
    "fullSectionsTitre": "Partie réglementaire > Livre Ier",
}


def stub_embed(texts: Sequence[str]) -> list[Embedding]:
    """Real-shaped (`DENSE_DIM`), fake-valued — the point counts this file asserts on
    depend only on shape matching the real collections `run_ingest` creates, never on the
    vectors' content."""
    return [
        ([1.0] * DENSE_DIM, models.SparseVector(indices=[i], values=[1.0]))
        for i in range(len(texts))
    ]


@pytest.fixture(scope="module")
def client() -> Iterator[QdrantClient]:
    c = QdrantClient(":memory:")
    try:
        yield c
    finally:
        c.close()


@pytest.fixture(scope="module")
def report(client: QdrantClient) -> IngestReport:
    return run_ingest(client, stub_embed, articles_path=ARTICLES_PATH, fiches_dir=FICHES_DIR)


def test_reports_the_measured_point_counts_for_the_real_corpus(report: IngestReport) -> None:
    assert report.articles_points == MEASURED_ARTICLE_POINTS
    assert report.fiches_points == MEASURED_FICHE_POINTS
    assert report.total_points == MEASURED_TOTAL_POINTS


def test_points_are_reachable_through_the_stable_aliases(
    client: QdrantClient, report: IngestReport
) -> None:
    assert client.count("articles").count == MEASURED_ARTICLE_POINTS
    assert client.count("fiches").count == MEASURED_FICHE_POINTS


def test_rerunning_the_whole_corpus_is_a_no_op_upsert(
    client: QdrantClient, report: IngestReport
) -> None:
    """SPEC §6.4 — a second `make ingest` overwrites in place, not a second copy."""
    second = run_ingest(client, stub_embed, articles_path=ARTICLES_PATH, fiches_dir=FICHES_DIR)

    assert second.total_points == MEASURED_TOTAL_POINTS
    assert client.count("articles").count == MEASURED_ARTICLE_POINTS
    assert client.count("fiches").count == MEASURED_FICHE_POINTS


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> tuple[Path, Path]:
    """One article, one fiche — `main`'s wiring is under test here, not chunking at scale."""
    articles_path = tmp_path / "articles.jsonl"
    articles_path.write_text(json.dumps(_ARTICLE_ROW) + "\n", encoding="utf-8")
    fiches_dir = tmp_path / "fiches"
    fiches_dir.mkdir()
    (fiches_dir / "F1.xml").write_bytes(_FICHE_XML)
    return articles_path, fiches_dir


def test_main_wires_the_injected_client_and_embedder_and_prints_the_report(
    tiny_corpus: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    articles_path, fiches_dir = tiny_corpus
    tiny_client = QdrantClient(":memory:")
    try:
        report = main(
            client=tiny_client, embed=stub_embed, articles_path=articles_path, fiches_dir=fiches_dir
        )

        assert isinstance(report, IngestReport)
        assert report.articles_points == 1
        assert report.fiches_points == 1
        out = capsys.readouterr().out
        assert "articles: 1 point(s)" in out
        assert "fiches:   1 point(s)" in out
        assert "total:    2 point(s)" in out
    finally:
        tiny_client.close()


def test_main_does_not_close_an_injected_client(tiny_corpus: tuple[Path, Path]) -> None:
    """Only a client `main` constructed itself is `main`'s to close."""
    articles_path, fiches_dir = tiny_corpus
    tiny_client = QdrantClient(":memory:")
    try:
        main(
            client=tiny_client, embed=stub_embed, articles_path=articles_path, fiches_dir=fiches_dir
        )

        tiny_client.get_collections()
    finally:
        tiny_client.close()
