from __future__ import annotations

from copy import deepcopy
from typing import Any

from safe_rag.defenses.base import DefenseContext
from safe_rag.defenses.retrieve.extract_cusum import ExtractCUSUM, partition_retrieved
from safe_rag.defenses.retrieve.extract_hmm import novelty_rate
from safe_rag.defenses.retrieve.graph_isolation import (
    GraphIsolationDefense,
    _contains_token_phrase,
    _entity_label,
    _relation_endpoints,
    _require_record,
    _tokenize_for_phrase_match,
)
from safe_rag.systems.base import OriginalGraph, StructuredRetrieval


class CusumIsolationDefense(GraphIsolationDefense):
    """Isolation radius from CUSUM on D growth / novelty, with a post-warmup r=1 floor."""

    name = "cusum"
    requires_staged_query = True

    def __init__(
        self,
        decide_after: int = 2,
        hop: int = 1,
        mu0: float = 0.08,
        slack: float = 0.04,
        h_radius_2: float = 1.0,
        decay: float = 0.35,
        settle_eps: float = 0.05,
        settle_patience: int = 3,
    ):
        super().__init__(radius=0)
        if decide_after < 0:
            raise ValueError("decide_after must be >= 0")
        if hop not in {1, 2}:
            raise ValueError("CUSUM hop radius must be 1 or 2")
        if settle_patience < 1:
            raise ValueError("settle_patience must be >= 1")
        self.decide_after = int(decide_after)
        self.hop = int(hop)
        self.mu0 = float(mu0)
        self.slack = float(slack)
        self.h_radius_2 = float(h_radius_2)
        self.decay = float(decay)
        self.settle_eps = float(settle_eps)
        self.settle_patience = int(settle_patience)
        self.init_kwargs = {
            "decide_after": self.decide_after,
            "hop": self.hop,
            "mu0": self.mu0,
            "slack": self.slack,
            "h_radius_2": self.h_radius_2,
            "decay": self.decay,
            "settle_eps": self.settle_eps,
            "settle_patience": self.settle_patience,
        }
        self.cusum = ExtractCUSUM(
            mu0=self.mu0,
            slack=self.slack,
            h_radius_2=self.h_radius_2,
            decay=self.decay,
        )
        self._last_growth = 0.0
        self._stagnant = 0
        self._pending_retrieval: StructuredRetrieval | None = None
        self._pending_filtered: StructuredRetrieval | None = None

    def initialize(self, graph: OriginalGraph) -> None:
        super().initialize(graph)
        self.radius = 0
        self._last_growth = 0.0
        self._stagnant = 0
        self._pending_retrieval = None
        self._pending_filtered = None
        self.cusum = ExtractCUSUM(
            mu0=self.mu0,
            slack=self.slack,
            h_radius_2=self.h_radius_2,
            decay=self.decay,
        )

    def _radius_for_turn(self, turn: int, score: float, novelty: float) -> int:
        if turn <= self.decide_after:
            self._stagnant = 0
            return 0
        if self._last_growth < self.settle_eps and novelty < self.settle_eps:
            self._stagnant += 1
        else:
            self._stagnant = 0
        if score >= self.h_radius_2:
            return 2
        if self._stagnant >= self.settle_patience:
            return 0
        return 1

    def _retrieved_labels(self, retrieval: StructuredRetrieval) -> set[str]:
        labels: set[str] = set()
        for entity in retrieval.entities:
            labels.add(_entity_label(_require_record(entity, "entity")))
        for relation in retrieval.relations:
            source, target = _relation_endpoints(_require_record(relation, "relationship"))
            labels.update((source, target))
        return labels

    def on_structured_retrieve(
        self,
        retrieval: StructuredRetrieval,
        ctx: DefenseContext,
    ) -> StructuredRetrieval:
        if not self._initialized:
            raise RuntimeError("CUSUM isolation defense has not been initialized")

        retrieved = self._retrieved_labels(retrieval)
        disclosed = set(self.released_graph.nodes)
        parts = partition_retrieved(retrieved, disclosed, self.topology, radius=self.hop)
        novelty = parts["belt"] + parts["far"]
        observation = max(self._last_growth, novelty)
        accumulate = ctx.turn > self.decide_after
        detector = self.cusum.observe(observation, accumulate=accumulate)
        self.radius = self._radius_for_turn(ctx.turn, float(detector["score"]), novelty)
        detector.update(
            {
                "far": parts["far"],
                "belt": parts["belt"],
                "in_d": parts["in_d"],
                "growth": self._last_growth,
                "novelty": novelty,
                "label_novelty": novelty_rate(retrieved, disclosed),
                "hop": self.hop,
                "accumulate": accumulate,
                "stagnant": self._stagnant,
                "settle_eps": self.settle_eps,
                "settle_patience": self.settle_patience,
                "radius": self.radius,
                "retrieved_nodes": len(retrieved),
                "disclosed_nodes": len(disclosed),
            }
        )

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
            raise RuntimeError("CUSUM isolation defense has not been initialized")
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
        retrieved = self._retrieved_labels(pending)
        before = set(self.released_graph.nodes)
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
        new_disclosed = len(mentioned - before)
        self._last_growth = (
            new_disclosed / len(retrieved) if retrieved else 0.0
        )
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
            detector["new_disclosed"] = new_disclosed
            detector["next_growth"] = self._last_growth
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
