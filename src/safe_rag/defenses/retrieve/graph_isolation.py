from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any
import unicodedata

import networkx as nx

from safe_rag.attacks.agea.utils import normalize_node_label
from safe_rag.defenses.base import DefenseContext, PassthroughDefense
from safe_rag.systems.base import OriginalGraph, StructuredRetrieval


_ENTITY_KEYS = ("entity", "entity_name", "name")
_RELATION_ENDPOINT_KEYS = (
    ("entity1", "entity2"),
    ("src_id", "tgt_id"),
    ("source", "target"),
)
_CHUNK_TEXT_KEYS = ("content", "text")


def _require_record(record: Any, record_type: str) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"Structured {record_type} record must be a mapping")
    return record


def _require_nonempty_string(
    record: Mapping[str, Any],
    keys: tuple[str, ...],
    record_type: str,
) -> str:
    present = [key for key in keys if key in record]
    if not present:
        raise RuntimeError(
            f"Structured {record_type} record is missing a supported field: {keys}"
        )
    nonempty: list[str] = []
    for key in present:
        value = record[key]
        if not isinstance(value, str):
            raise RuntimeError(f"Structured {record_type} field '{key}' must be a string")
        if value.strip():
            nonempty.append(value)
    if nonempty:
        return nonempty[0]
    raise RuntimeError(
        f"Structured {record_type} record has no non-empty supported value"
    )


def _entity_label(record: Mapping[str, Any]) -> str:
    value = _require_nonempty_string(record, _ENTITY_KEYS, "entity")
    labels = {
        normalize_node_label(record[key])
        for key in _ENTITY_KEYS
        if key in record and isinstance(record[key], str) and record[key].strip()
    }
    if len(labels) != 1:
        raise RuntimeError("Structured entity record has conflicting label fields")
    label = normalize_node_label(value)
    if not _normalize_for_phrase_match(label):
        raise RuntimeError("Structured entity label has no matchable Unicode tokens")
    return label


def _relation_endpoints(record: Mapping[str, Any]) -> tuple[str, str]:
    endpoints: list[tuple[str, str]] = []
    for source_key, target_key in _RELATION_ENDPOINT_KEYS:
        if source_key not in record and target_key not in record:
            continue
        if source_key not in record or target_key not in record:
            raise RuntimeError(
                "Structured relationship record has an incomplete endpoint pair"
            )
        source = normalize_node_label(
            _require_nonempty_string(
                record,
                (source_key,),
                "relationship endpoint",
            )
        )
        target = normalize_node_label(
            _require_nonempty_string(
                record,
                (target_key,),
                "relationship endpoint",
            )
        )
        if not _normalize_for_phrase_match(source) or not _normalize_for_phrase_match(target):
            raise RuntimeError(
                "Structured relationship endpoint has no matchable Unicode tokens"
            )
        endpoints.append((source, target))
    if not endpoints:
        raise RuntimeError(
            "Structured relationship record is missing a supported endpoint pair"
        )
    if any(endpoint != endpoints[0] for endpoint in endpoints[1:]):
        raise RuntimeError(
            "Structured relationship record has conflicting endpoint pairs"
        )
    return endpoints[0]


def _chunk_text(record: Mapping[str, Any]) -> str:
    present = [key for key in _CHUNK_TEXT_KEYS if key in record]
    if not present:
        raise RuntimeError(
            f"Structured chunk record is missing a supported field: {_CHUNK_TEXT_KEYS}"
        )
    for key in present:
        value = record[key]
        if not isinstance(value, str):
            raise RuntimeError(f"Structured chunk field '{key}' must be a string")
    return "\n".join(record[key] for key in present)


def _is_cjk_token_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
        or 0x3040 <= codepoint <= 0x309F
        or 0x30A0 <= codepoint <= 0x30FF
        or 0x31F0 <= codepoint <= 0x31FF
        or 0xAC00 <= codepoint <= 0xD7A3
        or 0x1100 <= codepoint <= 0x11FF
        or 0x3130 <= codepoint <= 0x318F
        or 0xA960 <= codepoint <= 0xA97F
        or 0xD7B0 <= codepoint <= 0xD7FF
    )


def _tokenize_for_phrase_match(value: str) -> tuple[str, ...]:
    tokens: list[str] = []
    current: list[str] = []
    for character in unicodedata.normalize("NFKC", value).casefold():
        if _is_cjk_token_character(character):
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append(character)
        elif unicodedata.category(character)[0] in {"L", "M", "N"}:
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def _normalize_for_phrase_match(value: str) -> str:
    return " ".join(_tokenize_for_phrase_match(value))


def _contains_token_phrase(
    text_tokens: tuple[str, ...],
    phrase_tokens: tuple[str, ...],
) -> bool:
    if not phrase_tokens:
        return False

    def matches_from(phrase_index: int, text_index: int) -> bool:
        if phrase_index == len(phrase_tokens):
            return True
        if text_index == len(text_tokens):
            return False

        phrase_token = phrase_tokens[phrase_index]
        text_token = text_tokens[text_index]
        if phrase_token == text_token and matches_from(
            phrase_index + 1,
            text_index + 1,
        ):
            return True

        if len(phrase_token) > 1:
            compact = ""
            next_text_index = text_index
            while (
                next_text_index < len(text_tokens)
                and len(text_tokens[next_text_index]) == 1
                and len(compact) < len(phrase_token)
            ):
                compact += text_tokens[next_text_index]
                next_text_index += 1
                if compact == phrase_token and matches_from(
                    phrase_index + 1,
                    next_text_index,
                ):
                    return True

        if len(phrase_token) == 1:
            compact = phrase_token
            next_phrase_index = phrase_index + 1
            while (
                next_phrase_index < len(phrase_tokens)
                and len(phrase_tokens[next_phrase_index]) == 1
            ):
                compact += phrase_tokens[next_phrase_index]
                next_phrase_index += 1
                if compact == text_token and matches_from(
                    next_phrase_index,
                    text_index + 1,
                ):
                    return True

        return False

    return any(matches_from(0, start) for start in range(len(text_tokens)))


class GraphIsolationDefense(PassthroughDefense):
    """Block cumulative released graph regions at the generation boundary."""

    name = "isolation"
    requires_staged_query = True

    def __init__(self, radius: int = 1):
        if radius not in {1, 2}:
            raise ValueError("Graph isolation radius must be 1 or 2")
        self.radius = radius
        self.topology = nx.Graph()
        self.released_graph = nx.Graph()
        self.released_entities: dict[str, dict[str, Any]] = {}
        self.forbidden_nodes: set[str] = set()
        self._initialized = False

    def _forbidden_edge_count(self) -> int:
        return sum(
            1
            for source, target in self.topology.edges
            if source in self.forbidden_nodes or target in self.forbidden_nodes
        )

    def initialize(self, graph: OriginalGraph) -> None:
        topology = nx.Graph()
        topology.add_nodes_from(
            label
            for node in graph.original_nodes
            if (label := normalize_node_label(str(node)))
        )
        topology.add_edges_from(
            (source, target)
            for raw_source, raw_target in graph.original_edges
            if (source := normalize_node_label(str(raw_source)))
            and (target := normalize_node_label(str(raw_target)))
        )
        self.topology = topology
        self.released_graph = nx.Graph()
        self.released_entities = {}
        self.forbidden_nodes = set()
        self._initialized = True

    def on_structured_retrieve(
        self,
        retrieval: StructuredRetrieval,
        ctx: DefenseContext,
    ) -> StructuredRetrieval:
        del ctx
        if not self._initialized:
            raise RuntimeError("Graph isolation defense has not been initialized")

        allowed_entities: list[dict[str, Any]] = []
        allowed_relations: list[dict[str, Any]] = []
        allowed_chunks: list[dict[str, Any]] = []
        blocked_nodes = 0
        blocked_edges = 0
        blocked_chunks = 0
        cached_nodes_reused = 0
        candidate_labels: set[str] = set()
        unknown_edge_count = 0
        allowed_entity_labels: set[str] = set()

        for entity in retrieval.entities:
            entity_record = _require_record(entity, "entity")
            label = _entity_label(entity_record)
            candidate_labels.add(label)
            if label in self.forbidden_nodes:
                blocked_nodes += 1
                cached = self.released_entities.get(label)
                if cached is not None and label not in allowed_entity_labels:
                    allowed_entities.append(deepcopy(cached))
                    allowed_entity_labels.add(label)
                    cached_nodes_reused += 1
                continue
            if label not in allowed_entity_labels:
                allowed_entities.append(deepcopy(dict(entity_record)))
                allowed_entity_labels.add(label)

        topology_nodes = set(self.topology.nodes)
        for relation in retrieval.relations:
            relation_record = _require_record(relation, "relationship")
            source, target = _relation_endpoints(relation_record)
            candidate_labels.update((source, target))
            if source not in topology_nodes or target not in topology_nodes:
                unknown_edge_count += 1
            if source in self.forbidden_nodes or target in self.forbidden_nodes:
                blocked_edges += 1
                continue
            allowed_relations.append(deepcopy(dict(relation_record)))

        forbidden_phrase_tokens = tuple(
            phrase_tokens
            for label in self.forbidden_nodes
            if (phrase_tokens := _tokenize_for_phrase_match(label))
        )
        for chunk in retrieval.chunks:
            chunk_record = _require_record(chunk, "chunk")
            chunk_text = _chunk_text(chunk_record)
            text_tokens = _tokenize_for_phrase_match(chunk_text)
            if any(
                _contains_token_phrase(text_tokens, phrase_tokens)
                for phrase_tokens in forbidden_phrase_tokens
            ):
                blocked_chunks += 1
                continue
            allowed_chunks.append(deepcopy(dict(chunk_record)))

        unknown_labels = candidate_labels - topology_nodes
        isolation_stats = {
            "candidate_nodes": len(retrieval.entities),
            "allowed_nodes": len(allowed_entities),
            "blocked_nodes": blocked_nodes,
            "cached_nodes_reused": cached_nodes_reused,
            "candidate_edges": len(retrieval.relations),
            "allowed_edges": len(allowed_relations),
            "blocked_edges": blocked_edges,
            "candidate_chunks": len(retrieval.chunks),
            "allowed_chunks": len(allowed_chunks),
            "blocked_chunks": blocked_chunks,
            "unknown_topology_nodes": len(unknown_labels),
            "unknown_topology_edges": unknown_edge_count,
            "forbidden_nodes": len(self.forbidden_nodes),
            "forbidden_edges": self._forbidden_edge_count(),
            "released_nodes": self.released_graph.number_of_nodes(),
            "released_edges": self.released_graph.number_of_edges(),
            "radius": self.radius,
        }
        extra = deepcopy(retrieval.extra)
        extra["isolation"] = isolation_stats
        return StructuredRetrieval(
            entities=allowed_entities,
            relations=allowed_relations,
            chunks=allowed_chunks,
            context="",
            extra=extra,
        )

    def on_disclose(self, retrieval: StructuredRetrieval, ctx: DefenseContext) -> None:
        del ctx
        if not self._initialized:
            raise RuntimeError("Graph isolation defense has not been initialized")

        for entity in retrieval.entities:
            entity_record = _require_record(entity, "entity")
            label = _entity_label(entity_record)
            self.released_graph.add_node(label)
            self.released_entities.setdefault(label, deepcopy(dict(entity_record)))

        for relation in retrieval.relations:
            relation_record = _require_record(relation, "relationship")
            source, target = _relation_endpoints(relation_record)
            self.released_graph.add_edge(source, target)

        released_nodes = set(self.released_graph.nodes)
        forbidden = set(released_nodes)
        topology_nodes = set(self.topology.nodes)
        for node in released_nodes & topology_nodes:
            forbidden.update(
                nx.single_source_shortest_path_length(
                    self.topology,
                    node,
                    cutoff=self.radius,
                )
            )
        self.forbidden_nodes = forbidden
        isolation_stats = retrieval.extra.get("isolation")
        if isinstance(isolation_stats, dict):
            isolation_stats.update(
                {
                    "forbidden_nodes": len(self.forbidden_nodes),
                    "forbidden_edges": self._forbidden_edge_count(),
                    "released_nodes": self.released_graph.number_of_nodes(),
                    "released_edges": self.released_graph.number_of_edges(),
                }
            )
