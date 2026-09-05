"""SPEC §10.7, ADR-0009, #42 — `generate()` wires the prompt, an injected `generate_fn`
seam, and the citation guardrail into the fat `GenerationResult` object, never repairing."""

from __future__ import annotations

from rag.generation.pipeline import GenerateFn, generate
from rag.generation.prompt import HistoryTurn, Message
from rag.generation.schema import Envelope, FondementJuridique, Reponse
from rag.retrieval.candidates import Candidate, Provenance, Register
from rag.retrieval.pipeline import RetrievalResult
from rag.retrieval.short_circuit import ShortCircuitPath


def _article(citation_id: str) -> Candidate:
    return Candidate(
        id=f"point-{citation_id}",
        score=0.9,
        register=Register.ARTICLE,
        payload={"citation_id": citation_id, "text": "..."},
        provenance=frozenset({Provenance.SEARCH}),
    )


def _fake_generate_fn(envelope: Envelope) -> GenerateFn:
    def fn(messages: list[Message]) -> Envelope:
        return envelope

    return fn


def _retrieval(contexts: list[Candidate], candidate_pools: dict[str, list[Candidate]]) -> RetrievalResult:
    return RetrievalResult(
        short_circuit_path=ShortCircuitPath.NO_REFERENCE,
        contexts=contexts,
        candidate_pools=candidate_pools,
    )


def test_generate_returns_the_envelope_the_seam_produced() -> None:
    envelope = Reponse(explanation="Bonjour.", fondement_juridique=[])
    retrieval = _retrieval([], {})

    result = generate("bonjour", retrieval, _fake_generate_fn(envelope))

    assert result.envelope == envelope


def test_generate_carries_the_final_contexts_and_candidate_pools_forward() -> None:
    fiche_pool = [Candidate(id="f1", score=0.5, register=Register.FICHE, payload={}, provenance=frozenset())]
    article = _article("L113-15-2")
    retrieval = _retrieval([article], {"fiche_leg": fiche_pool})
    envelope = Reponse(
        explanation="...",
        fondement_juridique=[FondementJuridique(article_id="L113-15-2", gloss="résiliation")],
    )

    result = generate("une question", retrieval, _fake_generate_fn(envelope))

    assert result.contexts == [article]
    assert result.candidate_pools == {"fiche_leg": fiche_pool}


def test_generate_checks_citations_against_the_same_contexts_used_for_the_prompt() -> None:
    retrieval = _retrieval([_article("L113-15-2")], {})
    envelope = Reponse(
        explanation="...",
        fondement_juridique=[FondementJuridique(article_id="L121-1", gloss="fabriquée")],
    )

    result = generate("une question", retrieval, _fake_generate_fn(envelope))

    assert not result.citation_outcome.valid
    assert result.citation_outcome.fabricated_ids == frozenset({"L121-1"})


def test_generate_never_repairs_the_returned_envelope() -> None:
    """SPEC §10.5: the guardrail is never auto-repaired — `generate()`'s own return must
    carry the model's fabricated citation untouched, for eval to record."""
    retrieval = _retrieval([_article("L113-15-2")], {})
    envelope = Reponse(
        explanation="...",
        fondement_juridique=[FondementJuridique(article_id="L121-1", gloss="fabriquée")],
    )

    result = generate("une question", retrieval, _fake_generate_fn(envelope))

    assert result.envelope == envelope
    assert [f.article_id for f in result.envelope.fondement_juridique] == ["L121-1"]


def test_generate_passes_the_raw_turn_and_contexts_into_the_prompt_seen_by_generate_fn() -> None:
    captured: list[list[Message]] = []

    def capturing_fn(messages: list[Message]) -> Envelope:
        captured.append(messages)
        return Reponse(explanation="...", fondement_juridique=[])

    retrieval = _retrieval([_article("L113-15-2")], {})
    history = [HistoryTurn(role="user", content="précédent")]

    generate("une question", retrieval, capturing_fn, history=history)

    assert len(captured) == 1
    messages = captured[0]
    assert messages[1] == ("user", "précédent")
    assert "une question" in messages[-1][1]
    assert "L113-15-2" in messages[-1][1]
