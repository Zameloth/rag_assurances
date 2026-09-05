"""SPEC §4.1 / ADR-0003 / #24 — fiche chunking."""

from rag.ingest.articles import BAND
from rag.ingest.fiches import chunk_fiche


def _fiche_xml(body: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Publication xmlns:dc="http://purl.org/dc/elements/1.1/" ID="F1">'
        "<dc:title>Titre de test</dc:title>"
        '<dc:source>https://www.legifrance.gouv.fr/codes/id/LEGISCTA000000000001</dc:source>'
        '<FilDAriane><Niveau ID="Particuliers">Accueil particuliers</Niveau></FilDAriane>'
        f"{body}"
        "</Publication>"
    ).encode()


def test_short_fiche_is_one_whole_point() -> None:
    text = "Le contrat d'assurance habitation couvre les dommages causés au logement loué ou occupé par l'assuré."
    xml = _fiche_xml(f"<Texte><Paragraphe>{text}</Paragraphe></Texte>")
    chunks = chunk_fiche(xml)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].chapitre_titre is None
    assert chunks[0].cas_label is None
    assert chunks[0].chunk_index == 0


def test_navigation_and_definition_elements_produce_no_chunks() -> None:
    """Only Introduction/Texte/Conclusion/ListeSituations are indexed (SPEC §4.1) —
    Reference and Definition sit outside the four and must never be descended into."""
    xml = _fiche_xml(
        "<Texte><Paragraphe>Contenu utile de la fiche à indexer pour les besoins du test.</Paragraphe></Texte>"
        '<Reference type="Texte de référence" URL="https://x" ID="R1">'
        "<Titre>Un intitulé de référence à ignorer</Titre></Reference>"
        '<Definition ID="R2"><Titre>Terme du glossaire</Titre>'
        "<Texte><Paragraphe>Définition à ignorer complètement.</Paragraphe></Texte></Definition>"
    )
    chunks = chunk_fiche(xml)
    combined = " ".join(c.text for c in chunks)
    assert "Définition à ignorer" not in combined
    assert "intitulé de référence" not in combined
    assert "Contenu utile" in combined


def test_listesituations_is_indexed_even_without_texte() -> None:
    """Assertion 6's motivating case (SPEC §4.1) — the 8 fiches lacking `<Texte>` still
    yield chunks because `<ListeSituations>` is treated as a body element."""
    xml = _fiche_xml(
        "<ListeSituations><Situation><Titre>Cas général</Titre><Texte><Paragraphe>"
        "Contenu spécifique à cette situation particulière de la fiche à indexer."
        "</Paragraphe></Texte></Situation></ListeSituations>"
    )
    chunks = chunk_fiche(xml)
    assert len(chunks) >= 1
    assert any("Contenu spécifique" in c.text for c in chunks)


def test_fiche_with_no_body_elements_yields_no_chunks() -> None:
    """Documents the pathological-input boundary, not a corpus condition — every committed
    fiche carries real content in one of the four body elements (assertion 6 at the corpus
    test level, not a guarantee `chunk_fiche` itself can make for an empty document)."""
    xml = _fiche_xml('<Reference type="Texte de référence" URL="https://x" ID="R1"><Titre>Seule référence</Titre></Reference>')
    assert chunk_fiche(xml) == []


def _bulk_paragraph(label: str, count: int = 25) -> str:
    """Enough repeated content to push a `<Chapitre>`/`<Cas>` comfortably over band on its
    own, forcing the parent to descend into it as its own context frame instead of emitting
    the whole subtree — and its own chapitre_titre/cas_label — as one combined chunk. Each
    paragraph is kept comfortably over the absolute 32-token floor on its own so a leftover
    single-paragraph tail never trips the cross-boundary floor fallback this test isn't
    about (that fallback gets its own coverage below)."""
    sentence = (
        f"Contenu de remplissage assez long pour le test {label} qui décrit une situation "
        "d'assurance précise et détaillée applicable à ce cas particulier."
    )
    return "".join(f"<Paragraphe>{sentence} Occurrence numéro {i}.</Paragraphe>" for i in range(count))


def test_chapitre_titre_is_excluded_from_text_but_carried_as_metadata() -> None:
    xml = _fiche_xml(
        "<Texte>"
        f"<Chapitre><Titre><Paragraphe>Quelles garanties choisir ?</Paragraphe></Titre>{_bulk_paragraph('un')}</Chapitre>"
        f"<Chapitre><Titre><Paragraphe>Comment résilier le contrat ?</Paragraphe></Titre>{_bulk_paragraph('deux')}</Chapitre>"
        "</Texte>"
    )
    chunks = chunk_fiche(xml)
    assert len(chunks) >= 2
    first_chapitre_chunks = [c for c in chunks if c.chapitre_titre == "Quelles garanties choisir ?"]
    assert first_chapitre_chunks
    for chunk in first_chapitre_chunks:
        assert "Quelles garanties choisir" not in chunk.text
        assert chunk.cas_label is None


def test_cas_label_is_carried_and_never_merges_across_a_cas_boundary() -> None:
    xml = _fiche_xml(
        "<Texte><Chapitre><Titre><Paragraphe>Résiliation du contrat</Paragraphe></Titre>"
        '<BlocCas affichage="radio">'
        f"<Cas><Titre>Vous êtes locataire</Titre>{_bulk_paragraph('locataire')}</Cas>"
        f"<Cas><Titre>Vous êtes propriétaire</Titre>{_bulk_paragraph('propriétaire')}</Cas>"
        "</BlocCas></Chapitre></Texte>"
    )
    chunks = chunk_fiche(xml)
    assert {c.cas_label for c in chunks} == {"Vous êtes locataire", "Vous êtes propriétaire"}
    for chunk in chunks:
        assert chunk.chapitre_titre == "Résiliation du contrat"
        assert chunk.cas_label is not None
        assert chunk.cas_label not in chunk.text
    locataire_chunks = [c for c in chunks if c.cas_label == "Vous êtes locataire"]
    proprietaire_chunks = [c for c in chunks if c.cas_label == "Vous êtes propriétaire"]
    assert not any("propriétaire" in c.text for c in locataire_chunks)
    assert not any("locataire" in c.text for c in proprietaire_chunks)


def test_short_paragraphs_merge_under_the_floor_without_exceeding_the_band() -> None:
    paragraph = "Les assureurs proposent plusieurs garanties complémentaires selon le profil de chaque client assuré."
    paragraphs = "".join(f"<Paragraphe>{paragraph} Paragraphe numéro {i}.</Paragraphe>" for i in range(40))
    xml = _fiche_xml(f"<Texte><Chapitre><Titre><Paragraphe>Chapitre long</Paragraphe></Titre>{paragraphs}</Chapitre></Texte>")
    chunks = chunk_fiche(xml)
    assert 1 < len(chunks) < 40
    for chunk in chunks:
        assert chunk.tokens <= BAND
        assert chunk.chapitre_titre == "Chapitre long"


def test_split_chunks_are_ordered_and_reconstruct_without_loss() -> None:
    paragraph = "Les assureurs proposent plusieurs garanties complémentaires selon le profil de chaque client assuré."
    paragraphs = "".join(f"<Paragraphe>{paragraph} Paragraphe numéro {i}.</Paragraphe>" for i in range(40))
    xml = _fiche_xml(f"<Texte><Chapitre><Titre><Paragraphe>Chapitre long</Paragraphe></Titre>{paragraphs}</Chapitre></Texte>")
    chunks = chunk_fiche(xml)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for i in range(40):
        assert any(f"Paragraphe numéro {i}." in c.text for c in chunks)


def test_oversized_atomic_node_falls_back_to_sentence_split() -> None:
    sentence = (
        "Cette phrase juridique très longue décrit une situation contractuelle précise "
        "applicable aux assurés et aux assureurs dans le cadre d'un contrat d'assurance "
        "habitation ou automobile souscrit auprès d'une compagnie agréée."
    )
    block = " ".join(f"{sentence} Référence numéro {i}." for i in range(40))
    xml = _fiche_xml(f"<Texte><Paragraphe>{block}</Paragraphe></Texte>")
    chunks = chunk_fiche(xml)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.tokens <= BAND


def test_no_chunk_text_repeats_a_sibling_chunk_s_text() -> None:
    """SPEC §4.3 — no overlap."""
    paragraph = "Les assureurs proposent plusieurs garanties complémentaires selon le profil de chaque client assuré."
    paragraphs = "".join(f"<Paragraphe>{paragraph} Paragraphe numéro {i}.</Paragraphe>" for i in range(40))
    xml = _fiche_xml(f"<Texte><Chapitre><Titre><Paragraphe>Chapitre long</Paragraphe></Titre>{paragraphs}</Chapitre></Texte>")
    chunks = chunk_fiche(xml)
    texts = [c.text for c in chunks]
    for i, text in enumerate(texts):
        for j, other in enumerate(texts):
            if i != j:
                assert text not in other
