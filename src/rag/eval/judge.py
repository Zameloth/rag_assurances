"""The two judged generation metrics — faithfulness and point coverage (SPEC §12.9, §12.10,
ADR-0025, #47).

Pure scoring logic behind one seam, `JudgeCall`: a prompt and an output schema in, the
parsed output and the **OpenRouter-resolved provider** out. `rag.eval.judge_chain` builds
the real one (`ChatOpenAI` on OpenRouter, pinned routing); tests inject a fake. Same split
`rag.generation.pipeline`/`rag.generation.chain` already draw around `GenerateFn`.

**Faithfulness is the managed Faithfulness v2 prompt, run client-side** (ADR-0025).
`FAITHFULNESS_V2_PROMPT_EN` is Langfuse's own template (`managed-evaluators.json`, v2 dated
2026-04-17 — the only managed RAG judge that implements the procedure it is named after,
`docs/research/langfuse-rag-eval.md` §5.3) and `FaithfulnessOutput` carries its
`outputDefinition` field for field. It runs as an experiment evaluator rather than a
server-side managed evaluator because a Langfuse LLM connection has nowhere to put
OpenRouter's `provider` request block and never reports the resolved provider back — the
two things SPEC §12.10 requires of every judge call.

**Point coverage has no managed equivalent** (SPEC §12.10), so it is ours: *did the answer
assert this point?* — coverage, not similarity (SPEC §12.9). The judge returns one verdict
per numbered point and **the score is computed here**, never asked of the model — a typed
field beats an inference (CONTEXT.md), and a verdict set that does not match the points
one-for-one raises instead of reading a skipped point as "not asserted".

**Both prompt languages are selectable per metric** (`JUDGE_FAITHFULNESS_LANGUAGE`,
`JUDGE_POINT_COVERAGE_LANGUAGE`): whether an English judge prompt over French content hurts
is settled on the calibration set, not from first principles (SPEC §12.10). The output
schemas' field descriptions stay the managed English ones in both languages — they are the
tool definition, shared by both prompts, not part of the prompt being A/B'd.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel, Field

from rag.generation.schema import Envelope, Refus

__all__ = [
    "FAITHFULNESS_V2_PROMPT_EN",
    "FaithfulnessOutput",
    "Judge",
    "JudgeCall",
    "JudgeOutputError",
    "JudgeScore",
    "JudgedMetric",
    "PointCoverageOutput",
    "PointVerdict",
    "PromptLanguage",
    "render_answer",
]

_M = TypeVar("_M", bound=BaseModel)


class PromptLanguage(enum.StrEnum):
    EN = "en"
    FR = "fr"


class JudgedMetric(enum.StrEnum):
    """The two judged rows of SPEC §12.9's table; the value is the score name used in
    Langfuse and in `eval/runs/<run-id>.json`."""

    FAITHFULNESS = "faithfulness"
    POINT_COVERAGE = "point_coverage"


class JudgeCall(Protocol):
    """One judge call: `prompt` as the single user message, `schema` as the function the
    judge must call. Returns the parsed output and the provider OpenRouter actually
    resolved (SPEC §12.10: recorded in every persisted run)."""

    def __call__(self, prompt: str, schema: type[_M]) -> tuple[_M, str]: ...


class JudgeOutputError(Exception):
    """The judge's structured output doesn't fit the question it was asked."""


class FaithfulnessOutput(BaseModel):
    """Managed Faithfulness v2's `outputDefinition`, verbatim."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Based on the claim analysis provided, give a single score from 0 to 1 (where 1 is "
            "perfectly faithful and 0 is entirely unsupported) representing the overall "
            "proportion of the answer that is grounded in the context. Output only the number"
        ),
    )
    reasoning: str = Field(description="One sentence reasoning for the score")


class PointVerdict(BaseModel):
    index: int = Field(description="The number of the expected point this verdict is about")
    asserted: bool = Field(description="True if the answer asserts this point, false otherwise")
    reasoning: str = Field(description="One sentence reasoning for the verdict")


class PointCoverageOutput(BaseModel):
    verdicts: list[PointVerdict] = Field(description="Exactly one verdict per expected point")


# Langfuse `managed-evaluators.json`, Faithfulness v2 (id `cmal6wart010lynrdtpv6olfv2`,
# 2026-04-17) — verbatim, `{{context}}`/`{{answer}}` included. Do not edit: this string *is*
# "managed Faithfulness v2"; an edited copy would be a third configuration nobody calibrated.
FAITHFULNESS_V2_PROMPT_EN = (
    "You are an expert evaluator. Your task is to determine the Faithfulness of a generated "
    "answer based on a provided context.\n\n"
    "Follow these steps exactly:\n"
    '1. Deconstruction: Break the "Answer" down into a list of atomic, self-contained '
    "statements. Do not use pronouns; replace them with the actual subjects.\n"
    '2. Verification: For each statement, check if it is supported by the "Context."\n'
    "3. Verdict: Assign a 1 if the statement is directly supported by the context, or a 0 if "
    "it is not supported or contradicted. Provide a brief reason for each.\n"
    "4. Calculation: Calculate the final faithfulness score as: Total Verdicts of 1 divided by "
    "Total Number of Statements.\n\n"
    "Input Data:\n"
    "Context: {{context}}\n"
    "Answer: {{answer}}"
)

# A faithful translation of the four steps — nothing added, nothing dropped. The FR/EN
# calibration A/B (SPEC §12.10) must vary the prompt's language and nothing else, or a
# difference between the two runs can't be attributed to language.
_FAITHFULNESS_PROMPT_FR = (
    "Tu es un évaluateur expert. Ta tâche est de mesurer la fidélité (faithfulness) d'une "
    "réponse générée par rapport au contexte fourni.\n\n"
    "Suis exactement ces étapes :\n"
    "1. Décomposition : découpe la « Réponse » en une liste d'affirmations atomiques et "
    "autonomes. N'utilise pas de pronoms ; remplace-les par les sujets réels.\n"
    "2. Vérification : pour chaque affirmation, vérifie si elle est étayée par le « Contexte ».\n"
    "3. Verdict : attribue 1 si l'affirmation est directement étayée par le contexte, 0 si elle "
    "ne l'est pas ou si elle le contredit. Justifie brièvement chaque verdict.\n"
    "4. Calcul : le score de fidélité final est le nombre de verdicts à 1 divisé par le nombre "
    "total d'affirmations.\n\n"
    "Données :\n"
    "Contexte : {{context}}\n"
    "Réponse : {{answer}}"
)

_POINT_COVERAGE_PROMPT_FR = (
    "Tu es un évaluateur expert. On te donne une question posée par un particulier sur le droit "
    "français des assurances, la réponse d'un assistant, et une liste numérotée de points "
    "attendus.\n\n"
    "Pour chaque point attendu, indique si la réponse l'affirme. Un point est affirmé si la "
    "réponse en exprime le contenu, même avec d'autres mots ; il ne l'est pas si la réponse "
    "l'omet, le contredit, ou ne l'évoque que de façon trop vague pour qu'un lecteur l'en "
    "retienne. Ne juge pas le style, la longueur ni la ressemblance avec une formulation "
    "particulière : seulement la présence du point.\n\n"
    "Rends exactement un verdict par point, en reprenant son numéro.\n\n"
    "Question : {{question}}\n\n"
    "Réponse : {{answer}}\n\n"
    "Points attendus :\n{{expected_points}}"
)

_POINT_COVERAGE_PROMPT_EN = (
    "You are an expert evaluator. You are given a question asked by a consumer about French "
    "insurance law, an assistant's answer, and a numbered list of expected points.\n\n"
    "For each expected point, decide whether the answer asserts it. A point is asserted if the "
    "answer expresses its content, even in other words; it is not if the answer omits it, "
    "contradicts it, or mentions it too vaguely for a reader to take it away. Do not judge "
    "style, length or similarity to any particular phrasing: only whether the point is "
    "present.\n\n"
    "Return exactly one verdict per point, using its number.\n\n"
    "Question: {{question}}\n\n"
    "Answer: {{answer}}\n\n"
    "Expected points:\n{{expected_points}}"
)

_FAITHFULNESS_PROMPTS = {PromptLanguage.EN: FAITHFULNESS_V2_PROMPT_EN, PromptLanguage.FR: _FAITHFULNESS_PROMPT_FR}
_POINT_COVERAGE_PROMPTS = {PromptLanguage.EN: _POINT_COVERAGE_PROMPT_EN, PromptLanguage.FR: _POINT_COVERAGE_PROMPT_FR}


def _fill(template: str, **variables: str) -> str:
    """Langfuse-style `{{name}}` substitution. Plain `str.replace`, not `str.format`: the
    context and answer are corpus French that may itself contain braces. Each variable is
    substituted once, in order, so a value containing another `{{name}}` is left alone."""
    for name, value in variables.items():
        template = template.replace("{{" + name + "}}", value)
    return template


def render_answer(envelope: Envelope) -> str:
    """The envelope as the judge reads it — `explanation` **plus** the typed fields around
    it. A fabricated citation lives in `fondement_juridique` and a refusal's class in
    `motif`; an explanation-only rendering would hide both faults from the judge."""
    lines = [envelope.explanation]
    if envelope.fondement_juridique:
        lines.append("")
        lines.append("Fondement juridique :")
        lines.extend(f"- {citation.article_id} : {citation.gloss}" for citation in envelope.fondement_juridique)
    if isinstance(envelope, Refus):
        lines.append("")
        lines.append(f"Refus (motif : {envelope.motif.value})")
    elif envelope.aucun_fondement is not None:
        lines.append("")
        lines.append(f"Aucun fondement : {envelope.aucun_fondement}")
    return "\n".join(lines)


@dataclass(frozen=True)
class JudgeScore:
    metric: JudgedMetric
    value: float
    reasoning: str
    provider: str


@dataclass(frozen=True)
class Judge:
    """`JUDGE_MODEL` behind `call`, with the prompt language chosen per metric. `model` is
    carried only to be pinned in run headers — `call` already knows which model it hits."""

    call: JudgeCall
    model: str
    faithfulness_language: PromptLanguage
    point_coverage_language: PromptLanguage

    def languages(self) -> dict[str, str]:
        return {
            JudgedMetric.FAITHFULNESS.value: self.faithfulness_language.value,
            JudgedMetric.POINT_COVERAGE.value: self.point_coverage_language.value,
        }

    def faithfulness(self, *, context: str, answer: str) -> JudgeScore:
        """Statements in `answer` grounded in `context` (the rendered context section the
        generator itself was shown, `rag.generation.prompt.build_context_section`)."""
        prompt = _fill(_FAITHFULNESS_PROMPTS[self.faithfulness_language], context=context, answer=answer)
        output, provider = self.call(prompt, FaithfulnessOutput)
        return JudgeScore(JudgedMetric.FAITHFULNESS, output.score, output.reasoning, provider)

    def point_coverage(self, *, question: str, answer: str, expected_points: Sequence[str]) -> JudgeScore | None:
        """Share of `expected_points` the answer asserts. `None` — undefined, not zero —
        when there are no points (`hors_corpus` items, SPEC §12.9), without a call."""
        if not expected_points:
            return None
        numbered = "\n".join(f"{index}. {point}" for index, point in enumerate(expected_points, start=1))
        prompt = _fill(
            _POINT_COVERAGE_PROMPTS[self.point_coverage_language],
            question=question,
            answer=answer,
            expected_points=numbered,
        )
        output, provider = self.call(prompt, PointCoverageOutput)

        indices = [verdict.index for verdict in output.verdicts]
        expected_indices = list(range(1, len(expected_points) + 1))
        if sorted(indices) != expected_indices:
            raise JudgeOutputError(f"judge returned verdicts for points {indices}, expected exactly {expected_indices}")
        asserted = sum(1 for verdict in output.verdicts if verdict.asserted)
        reasoning = "; ".join(
            f"{verdict.index}: {verdict.reasoning}" for verdict in sorted(output.verdicts, key=lambda v: v.index)
        )
        return JudgeScore(JudgedMetric.POINT_COVERAGE, asserted / len(expected_points), reasoning, provider)
