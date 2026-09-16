"""SPEC §12.8 / #38 — embedding-time enrichment, the two pre-ladder A/Bs' dense-text builders."""

from rag.ingest.articles import ArticleChunk
from rag.ingest.enrichment import enrich_article_dense_text, enrich_fiche_dense_text
from rag.ingest.fiches import FicheChunk, FicheMetadata

_CHUNK_TEXT = "La résiliation prend effet un mois après réception de la lettre."


def _article_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "cid": "LEGIARTI000000000001",
        "citation_id": "L113-3",
        "fullSectionsTitre": (
            "Partie législative > Livre Ier : Le contrat > Titre Ier : Règles communes"
        ),
    }
    base.update(overrides)
    return base


def _article_chunk(**overrides: object) -> ArticleChunk:
    base: dict[str, object] = dict(text=_CHUNK_TEXT, chunk_index=0, tokens=12, is_stub=False)
    base.update(overrides)
    return ArticleChunk(**base)  # type: ignore[arg-type]


def _fiche_meta(**overrides: object) -> FicheMetadata:
    base: dict[str, object] = dict(
        fiche_id="F2594",
        section_ids=["LEGISCTA000006157200"],
        title="Modification du contrat d'assurance habitation",
        sp_url="https://www.service-public.gouv.fr/particuliers/vosdroits/F2594",
        date_modified="2025-04-28",
        fil_ariane="Accueil particuliers > Assurance habitation",
        fiche_type="Fiche d'information conditionnée",
    )
    base.update(overrides)
    return FicheMetadata(**base)  # type: ignore[arg-type]


def _fiche_chunk(**overrides: object) -> FicheChunk:
    base: dict[str, object] = dict(
        text=_CHUNK_TEXT, chunk_index=0, tokens=12, chapitre_titre=None, cas_label=None
    )
    base.update(overrides)
    return FicheChunk(**base)  # type: ignore[arg-type]


class TestEnrichArticleDenseText:
    def test_prepends_the_breadcrumb_ahead_of_the_raw_chunk(self) -> None:
        row = _article_row()
        chunk = _article_chunk()

        enriched = enrich_article_dense_text(row, chunk)

        assert enriched == f"{row['fullSectionsTitre']}\n{_CHUNK_TEXT}"

    def test_falls_back_to_raw_text_when_full_sections_titre_is_absent(self) -> None:
        row = _article_row(fullSectionsTitre=None)
        chunk = _article_chunk()

        assert enrich_article_dense_text(row, chunk) == _CHUNK_TEXT

    def test_falls_back_to_raw_text_when_full_sections_titre_is_missing_entirely(self) -> None:
        row = {"cid": "X", "citation_id": "L1"}
        chunk = _article_chunk()

        assert enrich_article_dense_text(row, chunk) == _CHUNK_TEXT

    def test_does_not_mutate_the_raw_chunk_text(self) -> None:
        chunk = _article_chunk()

        enrich_article_dense_text(_article_row(), chunk)

        assert chunk.text == _CHUNK_TEXT


class TestEnrichFicheDenseText:
    def test_prepends_title_only_when_chapitre_and_cas_are_absent(self) -> None:
        meta = _fiche_meta()
        chunk = _fiche_chunk()

        enriched = enrich_fiche_dense_text(meta, chunk)

        assert enriched == f"{meta.title}\n{_CHUNK_TEXT}"

    def test_prepends_title_chapitre_and_cas_when_all_present(self) -> None:
        meta = _fiche_meta()
        chunk = _fiche_chunk(chapitre_titre="Résiliation", cas_label="Si vous êtes locataire")

        enriched = enrich_fiche_dense_text(meta, chunk)

        assert enriched == f"{meta.title}\nRésiliation\nSi vous êtes locataire\n{_CHUNK_TEXT}"

    def test_skips_a_null_chapitre_titre_but_keeps_cas_label(self) -> None:
        meta = _fiche_meta()
        chunk = _fiche_chunk(chapitre_titre=None, cas_label="Si vous êtes locataire")

        enriched = enrich_fiche_dense_text(meta, chunk)

        assert enriched == f"{meta.title}\nSi vous êtes locataire\n{_CHUNK_TEXT}"

    def test_does_not_mutate_the_raw_chunk_text(self) -> None:
        chunk = _fiche_chunk(chapitre_titre="Résiliation")

        enrich_fiche_dense_text(_fiche_meta(), chunk)

        assert chunk.text == _CHUNK_TEXT
