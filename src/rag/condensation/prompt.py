"""Prompt assembly for the condenser: history turns, then the raw follow-up to condense
(SPEC §8.6, §8.7, ADR-0008, #43).

The three few-shot examples below are hand-written on topics deliberately outside the
golden set's ten `multi_turn`-tagged items (`eval/golden/golden-set.yaml`) — seeding the
condenser's prompt from the same items it is measured on would make the measurement partly
one of memorisation (ADR-0008, SPEC §8.6), the same reasoning the golden set itself follows
by refusing to generate questions from article text.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "MAX_HISTORY_TURNS",
    "MAX_HISTORY_TURN_CHARS",
    "SYSTEM_PROMPT",
    "HistoryTurn",
    "Message",
    "build_messages",
    "trim_history",
]

Message = tuple[Literal["system", "user", "assistant"], str]

# SPEC §8.7: "last 3 exchanges (up to 6 messages), both roles". Both roles are kept
# because the golden set scripts and freezes the prior assistant turn, so dropping
# assistant turns would make those fixtures carry text the system never sees (ADR-0008).
MAX_HISTORY_TURNS = 6

# SPEC §8.7: "trim on both turn count and raw size, since either alone is bypassable" — a
# turn-count-only trim still admits one 50k-character turn among the last 6; this bounds
# each turn independently of how many there are. Generous relative to a real French
# question/answer turn (low hundreds of characters) while still capping a hostile one.
MAX_HISTORY_TURN_CHARS = 2000


@dataclass(frozen=True)
class HistoryTurn:
    """One prior turn, already inside the last-3-exchanges window (SPEC §8.7). `content`
    is the full message for a `user` turn and stripped prose only for an `assistant` turn —
    the same stripped shape `rag.generation.prompt.HistoryTurn` carries, defined again here
    rather than imported: this package sits upstream of generation and must not depend on
    it (see the package docstring), even though the two types happen to agree structurally.

    Assistant-turn stripping (dropping any prior `fondement_juridique`) happens by
    construction — this type simply has no field to carry one in, the same reasoning
    `rag.generation.prompt.HistoryTurn`'s own docstring gives. Turn-count and raw-size
    trimming do not happen by construction — see `trim_history` below."""

    role: Literal["user", "assistant"]
    content: str


def trim_history(history: Sequence[HistoryTurn]) -> tuple[HistoryTurn, ...]:
    """SPEC §8.7's server-side enforcement: `history` is untrusted client input (the app is
    stateless — `CODING_STANDARDS.md`: "History is client-side, posted back each turn... a
    client can post 200 turns or one 50k-character turn"), so this is not optional and not
    the caller's judgement call.

    Keeps the last `MAX_HISTORY_TURNS`, then clamps each kept turn's `content` to
    `MAX_HISTORY_TURN_CHARS` — count and size are independent caps, applied together
    because either alone is bypassable (SPEC §8.7). Order is preserved; a `history` already
    inside both bounds passes through unchanged.
    """
    return tuple(
        HistoryTurn(role=turn.role, content=turn.content[:MAX_HISTORY_TURN_CHARS])
        for turn in history[-MAX_HISTORY_TURNS:]
    )


# SPEC §8.6's hard rules, plus three hand-written few-shot examples a 24B model wants. Kept
# as one system message rather than real few-shot message pairs — matches
# `rag.generation.prompt.SYSTEM_PROMPT`'s own shape, and keeps `build_messages` a plain
# history-then-turn assembly with no synthetic conversation spliced in front of the real one.
SYSTEM_PROMPT = """\
Tu es un module de reformulation de questions pour un assistant assurance francophone. \
Ta seule tâche : transformer le dernier message de l'utilisateur en une question \
autonome, compréhensible sans l'historique de la conversation. Tu ne réponds jamais à la \
question, tu la reformules.

Règles strictes :
1. Réponds uniquement par une question, en français, et rien d'autre — pas de préambule \
("Voici la question reformulée : "), pas d'explication, une seule ligne.
2. Conserve tels quels les mots du vocabulaire de l'utilisateur (par exemple « franchise \
», « vétusté », « délai de renonciation ») : ne les remplace jamais par la formulation \
d'un texte de loi.
3. Ne traduis jamais la question dans le registre juridique du Code des assurances.
4. Si la dernière question de l'utilisateur est déjà autonome — elle ne dépend pas de \
l'historique — renvoie-la exactement telle quelle, mot pour mot.
5. N'introduis jamais de référence à un article (par exemple « L113-15-2 ») que \
l'utilisateur n'a pas tapée lui-même dans son dernier message.

Voici trois exemples.

Exemple 1 — reformulation anaphorique :
Historique :
- Utilisateur : Est-ce que la garantie dégât des eaux couvre les fuites de canalisation \
encastrée ?
- Assistant : Oui, si la fuite provient d'une canalisation encastrée dans un mur ou une \
dalle, elle est en général couverte par la garantie dégât des eaux, sous réserve des \
conditions de votre contrat.
Dernier message : Et si je suis locataire ?
Requête condensée : Est-ce que la garantie dégât des eaux couvre les fuites de \
canalisation encastrée si je suis locataire ?

Exemple 2 — déjà autonome, renvoyée telle quelle :
Historique :
- Utilisateur : Quelle est la différence entre une assurance au tiers et une assurance \
tous risques ?
- Assistant : L'assurance au tiers couvre les dommages causés aux autres ; l'assurance \
tous risques couvre en plus vos propres dommages.
Dernier message : Est-ce que la garantie bris de glace est incluse dans une assurance \
tous risques ?
Requête condensée : Est-ce que la garantie bris de glace est incluse dans une assurance \
tous risques ?

Exemple 3 — changement de sujet, déjà autonome :
Historique :
- Utilisateur : Ma résidence secondaire doit-elle être assurée même si elle est \
inoccupée une partie de l'année ?
- Assistant : Oui, un logement doit rester assuré même inoccupé, notamment pour la \
responsabilité civile et les dégâts qu'il pourrait causer ou subir.
Dernier message : Est-ce qu'une assurance est obligatoire pour un chien de catégorie 1 ?
Requête condensée : Est-ce qu'une assurance est obligatoire pour un chien de catégorie 1 ?
"""


def build_messages(raw_turn: str, history: Sequence[HistoryTurn] = ()) -> list[Message]:
    """The condenser's message list: the system prompt, then `history` verbatim, then
    `raw_turn` as the final `user` message — the one the model reformulates.

    Takes `history` as given rather than calling `trim_history` itself — trimming is
    `condense()`'s job (`rag.condensation.pipeline`), applied once before either the
    `history == []` check or this assembly, so this function stays a plain, trim-agnostic
    "history then turn" composition, the same shape whether or not its caller happens to
    enforce SPEC §8.7's window.
    """
    messages: list[Message] = [("system", SYSTEM_PROMPT)]
    messages.extend((turn.role, turn.content) for turn in history)
    messages.append(("user", raw_turn))
    return messages
