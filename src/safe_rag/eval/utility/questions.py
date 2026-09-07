from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


DEFAULT_QUESTION_TYPES = ("Fact Retrieval", "Complex Reasoning")
DEFAULT_PROTOCOLS = ("p1",)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    """
    the a an of for to in on and or is are was were be been being this that with by as
    from at into than then it its their his her what which who how does do did can may
    might when where why if not no yes most common type used include including such also
    often usually typically involve involves involved involving patient patients treatment
    treatments diagnosis diagnostic method methods risk factor factors cancer cancers
    cell cells tumor tumors disease diseases primary recommended standard main major
    role associated increase increases increased higher high low more less after before
    during between among using based listed given according common
    """.split()
)


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answer: str
    question_type: str
    evidence: str = ""
    evidence_relations: str = ""
    source: str = ""

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Question":
        question = str(record.get("question") or "").strip()
        answer = str(record.get("answer") or "").strip()
        if not question or not answer:
            raise ValueError("QA record is missing question or answer")
        return cls(
            id=str(record.get("id") or question),
            question=question,
            answer=answer,
            question_type=str(record.get("question_type") or "Fact Retrieval"),
            evidence=str(record.get("evidence") or ""),
            evidence_relations=str(record.get("evidence_relations") or ""),
            source=str(record.get("source") or ""),
        )


@dataclass(frozen=True)
class UtilityPlan:
    questions: list[Question]
    sessions: dict[str, list[list[Question]]]


def load_questions(path: str | Path) -> list[Question]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"QA file must be a JSON list: {path}")
    return [Question.from_record(item) for item in raw if isinstance(item, dict)]


def _normalize_question(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.casefold()))


def _content_tokens(item: Question) -> frozenset[str]:
    text = " ".join((item.question, item.answer, item.evidence_relations))
    return frozenset(
        token
        for token in _TOKEN_RE.findall(text.casefold())
        if token not in _STOP and len(token) > 2
    )


def _document_frequency(token_sets: Sequence[frozenset[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tokens in token_sets:
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
    return counts


def _distinctive_tokens(tokens: frozenset[str], df: dict[str, int]) -> frozenset[str]:
    if not tokens:
        return frozenset()
    ranked = sorted(tokens, key=lambda token: (df.get(token, 0), token))
    keep = max(1, (len(ranked) + 1) // 2)
    return frozenset(ranked[:keep])


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _dedupe(questions: Iterable[Question]) -> list[Question]:
    seen: set[str] = set()
    unique: list[Question] = []
    for item in questions:
        key = _normalize_question(item.question)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _related(
    seed_tokens: frozenset[str],
    other_tokens: frozenset[str],
    distinctive: frozenset[str],
    min_overlap: float,
) -> bool:
    if _jaccard(seed_tokens, other_tokens) < min_overlap:
        return False
    return bool(distinctive & other_tokens)


def _cluster_topics(
    questions: Sequence[Question],
    *,
    min_overlap: float,
    cluster_cap: int,
) -> list[list[Question]]:
    token_sets = [_content_tokens(item) for item in questions]
    df = _document_frequency(token_sets)
    used: set[int] = set()
    clusters: list[list[Question]] = []
    for index, item in enumerate(questions):
        if index in used:
            continue
        seed_tokens = token_sets[index]
        distinctive = _distinctive_tokens(seed_tokens, df)
        ranked: list[tuple[float, int]] = []
        for other, other_tokens in enumerate(token_sets):
            if other == index or other in used:
                continue
            if not _related(seed_tokens, other_tokens, distinctive, min_overlap):
                continue
            ranked.append((_jaccard(seed_tokens, other_tokens), other))
        ranked.sort(reverse=True)
        chosen = [index]
        chosen_tokens = [seed_tokens]
        for _, other in ranked:
            if len(chosen) >= cluster_cap:
                break
            other_tokens = token_sets[other]
            if any(_jaccard(other_tokens, existing) > 0.78 for existing in chosen_tokens):
                continue
            chosen.append(other)
            chosen_tokens.append(other_tokens)
        used.update(chosen)
        cluster = [questions[position] for position in chosen]
        cluster.sort(key=lambda row: (0 if row.question_type == "Fact Retrieval" else 1, row.id))
        clusters.append(cluster)
        del item
    return clusters


def pack_topic_sessions(
    questions: Sequence[Question],
    *,
    session_size: int,
    min_overlap: float,
    min_session_size: int | None = None,
) -> list[list[Question]]:
    if session_size < 2:
        raise ValueError("session_size must be at least 2")
    del min_session_size
    clusters = _cluster_topics(
        questions,
        min_overlap=min_overlap,
        cluster_cap=max(session_size * 8, 32),
    )
    sessions: list[list[Question]] = []
    for cluster in clusters:
        for start in range(0, len(cluster) - session_size + 1, session_size):
            sessions.append(cluster[start : start + session_size])
    if not sessions:
        sessions = [cluster[:session_size] for cluster in clusters if len(cluster) >= 2]
    return sessions


def _sample_sessions(
    sessions: Sequence[list[Question]],
    *,
    max_questions: int,
    session_size: int,
    seed: int,
) -> list[list[Question]]:
    target = max(1, max_questions // session_size)
    ordered = list(sessions)
    if seed >= 0:
        import random

        rng = random.Random(seed)
        rng.shuffle(ordered)
    return ordered[:target]


def build_utility_plan(
    questions: Sequence[Question] | Sequence[dict[str, Any]],
    *,
    question_types: Optional[Sequence[str]] = None,
    max_questions: int,
    session_size: int = 4,
    seed: int = 42,
    protocols: Optional[Sequence[str]] = None,
    min_overlap: float = 0.22,
    dedupe: bool = True,
) -> UtilityPlan:
    allowed = tuple(question_types) if question_types else DEFAULT_QUESTION_TYPES
    protocol_names = tuple(protocols) if protocols else DEFAULT_PROTOCOLS
    unknown = [name for name in protocol_names if name not in {"p0", "p1"}]
    if unknown:
        raise ValueError(f"Unsupported utility protocols: {unknown}")
    if max_questions <= 0:
        raise ValueError("max_questions must be positive")

    loaded = [
        item if isinstance(item, Question) else Question.from_record(item)
        for item in questions
    ]
    filtered = [item for item in loaded if item.question_type in allowed]
    if dedupe:
        filtered = _dedupe(filtered)
    if not filtered:
        raise ValueError("No questions left after type filter / dedupe")

    packed = pack_topic_sessions(
        filtered,
        session_size=session_size,
        min_overlap=min_overlap,
    )
    if not packed:
        raise ValueError("Could not form any related-question sessions")
    sampled = _sample_sessions(
        packed,
        max_questions=max_questions,
        session_size=session_size,
        seed=seed,
    )
    selected = [item for session in sampled for item in session]
    sessions: dict[str, list[list[Question]]] = {}
    if "p1" in protocol_names:
        sessions["p1"] = sampled
    if "p0" in protocol_names:
        sessions["p0"] = [[item] for item in selected]
    return UtilityPlan(questions=selected, sessions=sessions)
