"""SPEC §4.2 / ADR-0003 / #23 — article chunking."""

from rag.ingest.articles import BAND, STUB_FLOOR, chunk_article
from rag.ingest.tokenizer import count_tokens


def _row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "cid": "LEGIARTI000000000001",
        "id": "LEGIARTI000000000002",
        "citation_id": "L113-1",
        "texte": "placeholder",
        "texteHtml": "<p>placeholder</p>",
        "sectionParentTitre": "Chapitre I : Placeholder",
    }
    base.update(overrides)
    return base


def test_short_article_is_one_whole_point() -> None:
    text = (
        "Le contrat d'assurance est régi par les dispositions du présent titre, ainsi que "
        "par les stipulations particulières convenues entre les parties, dans le respect "
        "des règles d'ordre public applicables aux assurances de dommages."
    )
    row = _row(texteHtml=f"<p>{text}</p>")
    chunks = chunk_article(row)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert not chunks[0].is_stub
    assert chunks[0].chunk_index == 0


def test_every_article_yields_at_least_one_chunk() -> None:
    """Assertion 6 (SPEC §4.5) — an all-table body must still produce a point."""
    row = _row(citation_id="Annexe à l'article A1", texteHtml="<table><tr><td>1</td></tr></table>")
    chunks = chunk_article(row)
    assert len(chunks) >= 1


def test_table_only_annexe_becomes_a_stub_with_table_marker() -> None:
    row = _row(
        citation_id="Annexe à l'article A132-9-4",
        texteHtml="<table><tbody><tr><td>1</td><td>2</td></tr></tbody></table>",
    )
    chunks = chunk_article(row)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.is_stub
    assert "Annexe à l'article A132-9-4" in chunk.text
    assert "table non indexée" in chunk.text
    assert "legifrance.gouv.fr/codes/article_lc/LEGIARTI000000000002" in chunk.text
    assert chunk.tokens >= STUB_FLOOR


def test_genuinely_short_article_becomes_a_stub_but_keeps_its_real_text() -> None:
    """No table involved — the real prose is padded with the citation suffix, not discarded."""
    text = "L'autorité administrative peut imposer l'usage de clauses types de contrats."
    row = _row(citation_id="L111-4", texteHtml=f"<p>{text}</p>")
    chunks = chunk_article(row)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.is_stub
    assert text in chunk.text
    assert "table non indexée" not in chunk.text
    assert "L111-4" in chunk.text
    assert chunk.tokens >= STUB_FLOOR


def test_table_content_is_stripped_but_surrounding_prose_survives() -> None:
    before = (
        "Les dispositions suivantes s'appliquent à tous les contrats souscrits après "
        "l'entrée en vigueur du présent article, sous réserve des clauses contraires."
    )
    after = (
        "Elles ne font en aucun cas obstacle à l'application des règles générales du "
        "droit des assurances prévues par ailleurs dans le présent code."
    )
    html = f"<p>{before}</p><table><tbody><tr><td>1</td></tr></tbody></table><p>{after}</p>"
    row = _row(texteHtml=html)
    chunks = chunk_article(row)
    assert len(chunks) == 1
    assert chunks[0].text == f"{before} {after}"
    assert not chunks[0].is_stub


def _twenty_paragraph_html() -> str:
    # Twenty ~40-token paragraphs — well over the 512-token band as a whole.
    paragraph = "Les assureurs et les intermédiaires doivent respecter les obligations légales relatives à l'information des assurés avant la conclusion du contrat."
    return "".join(f"<p>{paragraph} Paragraphe numéro {i}.</p>" for i in range(20))


def test_over_band_article_splits_on_blocks_with_no_chunk_over_band() -> None:
    row = _row(texteHtml=_twenty_paragraph_html())
    chunks = chunk_article(row)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.tokens <= BAND
        assert not chunk.is_stub


def test_split_chunks_are_ordered_and_reconstruct_without_loss() -> None:
    row = _row(texteHtml=_twenty_paragraph_html())
    chunks = chunk_article(row)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for i in range(20):
        assert any(f"Paragraphe numéro {i}." in c.text for c in chunks)


def test_oversized_single_block_falls_back_to_sentence_splitting() -> None:
    sentence = "Cette phrase juridique très longue décrit une obligation contractuelle précise applicable aux assureurs et aux intermédiaires dans le cadre de la souscription d'un contrat d'assurance vie ou dommages."
    block = " ".join(f"{sentence} Référence numéro {i}." for i in range(40))
    row = _row(texteHtml=f"<p>{block}</p>")
    chunks = chunk_article(row)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.tokens <= BAND


def test_stub_text_token_count_matches_count_tokens() -> None:
    row = _row(citation_id="Annexe à l'article A1", texteHtml="")
    chunk = chunk_article(row)[0]
    assert chunk.is_stub
    assert chunk.tokens == count_tokens(chunk.text)
