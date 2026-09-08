from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional, Sequence

from safe_rag.defenses.base import DefenseContext
from safe_rag.defenses.retrieve.extract_hmm import ExtractHMM, novelty_rate, unexplored_rate
from safe_rag.defenses.retrieve.graph_isolation import (
    GraphIsolationDefense,
    _contains_token_phrase,
    _entity_label,
    _relation_endpoints,
    _require_record,
    _tokenize_for_phrase_match,
)
from safe_rag.systems.base import OriginalGraph, StructuredRetrieval


class AdaptiveIsolationDefense(GraphIsolationDefense):
    """Choose isolation radius each turn from an online extraction HMM."""

    name = "adaptive"
    requires_staged_query = True

    def __init__(
        self,
        window_size: int = 3,
        decide_after: int = 2,
        p_radius_1: float = 0.35,
        p_radius_2: float = 0.70,
        consecutive_extract: int = 2,
        hmm_start: Optional[Sequence[float]] = None,
        hmm_transition: Optional[Sequence[Sequence[float]]] = None,
        hmm_emit_mean: Optional[Sequence[Sequence[float]]] = None,
        hmm_emit_std: Optional[Sequence[Sequence[float]]] = None,
    ):
        super().__init__(radius=0)
        if decide_after < 0:
            raise ValueError("decide_after must be >= 0")
        if consecutive_extract < 1:
            raise ValueError("consecutive_extract must be >= 1")
        self.window_size = int(window_size)
        self.decide_after = int(decide_after)
        self.p_radius_1 = float(p_radius_1)
        self.p_radius_2 = float(p_radius_2)
        self.consecutive_extract = int(consecutive_extract)
        self.hmm_start = hmm_start
        self.hmm_transition = hmm_transition
        self.hmm_emit_mean = hmm_emit_mean
        self.hmm_emit_std = hmm_emit_std
        self.init_kwargs = {
            "window_size": self.window_size,
            "decide_after": self.decide_after,
            "p_radius_1": self.p_radius_1,
            "p_radius_2": self.p_radius_2,
            "consecutive_extract": self.consecutive_extract,
            "hmm_start": hmm_start,
            "hmm_transition": hmm_transition,
            "hmm_emit_mean": hmm_emit_mean,
            "hmm_emit_std": hmm_emit_std,
        }
        self.hmm = ExtractHMM(
            window_size=self.window_size,
            start=hmm_start,
            transition=hmm_transition,
            emit_mean=hmm_emit_mean,
            emit_std=hmm_emit_std,
        )
        self._consecutive_high = 0
        self._pending_retrieval: StructuredRetrieval | None = None
        self._pending_filtered: StructuredRetrieval | None = None

    def initialize(self, graph: OriginalGraph) -> None:
        super().initialize(graph)
        self.radius = 0
        self._consecutive_high = 0
        self._pending_retrieval = None
        self._pending_filtered = None
        self.hmm = ExtractHMM(
            window_size=self.window_size,
            start=self.hmm_start,
            transition=self.hmm_transition,
            emit_mean=self.hmm_emit_mean,
            emit_std=self.hmm_emit_std,
        )

    def _retrieved_labels(self, retrieval: StructuredRetrieval) -> set[str]:
        labels: set[str] = set()
        for entity in retrieval.entities:
            labels.add(_entity_label(_require_record(entity, "entity")))
        for relation in retrieval.relations:
            source, target = _relation_endpoints(_require_record(relation, "relationship"))
            labels.update((source, target))
        return labels

    def _radius_from_p(self, p_extract: float) -> int:
        if p_extract < self.p_radius_1:
            self._consecutive_high = 0
            return 0
        if p_extract < self.p_radius_2:
            self._consecutive_high = 0
            return 1
        self._consecutive_high += 1
        if self._consecutive_high >= self.consecutive_extract:
            return 2
        return 1

    def on_structured_retrieve(
        self,
        retrieval: StructuredRetrieval,
        ctx: DefenseContext,
    ) -> StructuredRetrieval:
        if not self._initialized:
            raise RuntimeError("Adaptive isolation defense has not been initialized")

        retrieved = self._retrieved_labels(retrieval)
        disclosed = set(self.released_graph.nodes)
        novelty = novelty_rate(retrieved, disclosed)
        unexplored = unexplored_rate(retrieved, disclosed, self.topology)
        detector = self.hmm.observe(novelty, unexplored)
        if ctx.turn <= self.decide_after:
            self._consecutive_high = 0
            self.radius = 0
        else:
            self.radius = self._radius_from_p(float(detector["p"]))
        detector["radius"] = self.radius
        detector["retrieved_nodes"] = len(retrieved)
        detector["disclosed_nodes"] = len(disclosed)
        detector["consecutive_high"] = self._consecutive_high

        self.forbidden_nodes = self._forbidden_neighborhood(disclosed, self.radius)
        filtered = super().on_structured_retrieve(retrieval, ctx)
        extra = dict(filtered.extra or {})
        extra["detector"] = detector
        isolation = extra.get("isolation")
        if isinstance(isolation, dict):
            isolation["radius"] = self.radius
        filtered.extra = extra
        self._pending_retrieval = retrieval
        self._pending_filtered = filtered
        return filtered

    def on_disclose(self, retrieval: StructuredRetrieval, ctx: DefenseContext) -> None:
        del ctx
        if not self._initialized:
            raise RuntimeError("Adaptive isolation defense has not been initialized")
        isolation_stats = retrieval.extra.get("isolation")
        if isinstance(isolation_stats, dict):
            isolation_stats.update(
                {
                    "forbidden_nodes": len(self.forbidden_nodes),
                    "forbidden_edges": self._forbidden_edge_count(),
                    "released_nodes": self.released_graph.number_of_nodes(),
                    "released_edges": self.released_graph.number_of_edges(),
                    "radius": self.radius,
                }
            )

    def on_response(self, response: str, ctx: DefenseContext) -> str:
        del ctx
        pending = self._pending_retrieval
        if pending is None:
            return response
        mentioned = self._mentioned_labels(pending, response or "")
        entity_records: dict[str, dict[str, Any]] = {}
        for entity in pending.entities:
            record = dict(_require_record(entity, "entity"))
            label = _entity_label(record)
            entity_records[label] = record
        for label in mentioned:
            self.released_graph.add_node(label)
            cached = entity_records.get(label)
            if cached is not None:
                self.released_entities.setdefault(label, deepcopy(cached))
        for relation in pending.relations:
            record = _require_record(relation, "relationship")
            source, target = _relation_endpoints(record)
            if source in mentioned and target in mentioned:
                self.released_graph.add_edge(source, target)
        detector = None
        if self._pending_filtered is not None:
            extra = self._pending_filtered.extra
            detector = extra.get("detector")
            isolation = extra.get("isolation")
            if isinstance(isolation, dict):
                isolation.update(
                    {
                        "released_nodes": self.released_graph.number_of_nodes(),
                        "released_edges": self.released_graph.number_of_edges(),
                    }
                )
        if isinstance(detector, dict):
            detector["disclosed_from_answer"] = len(mentioned)
            detector["disclosed_nodes"] = self.released_graph.number_of_nodes()
            detector["disclosed_edges"] = self.released_graph.number_of_edges()
        self._pending_retrieval = None
        return response

    def _mentioned_labels(self, retrieval: StructuredRetrieval, response: str) -> set[str]:
        labels = self._retrieved_labels(retrieval)
        text_tokens = _tokenize_for_phrase_match(response)
        mentioned: set[str] = set()
        for label in labels:
            phrase = _tokenize_for_phrase_match(label)
            if phrase and _contains_token_phrase(text_tokens, phrase):
                mentioned.add(label)
        return mentioned
