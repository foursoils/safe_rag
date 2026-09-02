from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from typing import Any
import unicodedata

from safe_rag.systems.base import StructuredRetrieval


_ENTITY_KEYS = ("entity", "entity_name", "name")
_RELATION_ENDPOINT_KEYS = (
    ("entity1", "entity2"),
    ("src_id", "tgt_id"),
    ("source", "target"),
)
_CHUNK_TEXT_KEYS = ("content", "text")


def _require_nonempty_string(
    record: Mapping[str, Any],
    keys: tuple[str, ...],
    record_type: str,
) -> str:
    present = [key for key in keys if key in record]
    if not present:
        raise RuntimeError(
            f"LightRAG {record_type} record is missing a supported field: {keys}"
        )
    nonempty: list[str] = []
    for key in present:
        value = record[key]
        if not isinstance(value, str):
            raise RuntimeError(
                f"LightRAG {record_type} field '{key}' must be a string"
            )
        if value.strip():
            nonempty.append(value)
    if nonempty:
        return nonempty[0]
    raise RuntimeError(
        f"LightRAG {record_type} record has no non-empty supported value"
    )


def _canonical_identifier(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _validate_entity(record: Mapping[str, Any]) -> None:
    _require_nonempty_string(record, _ENTITY_KEYS, "entity")
    labels = {
        _canonical_identifier(record[key])
        for key in _ENTITY_KEYS
        if key in record and isinstance(record[key], str) and record[key].strip()
    }
    if len(labels) != 1:
        raise RuntimeError("LightRAG entity record has conflicting label fields")


def _validate_relation(record: Mapping[str, Any]) -> None:
    endpoints: list[tuple[str, str]] = []
    for source_key, target_key in _RELATION_ENDPOINT_KEYS:
        if source_key not in record and target_key not in record:
            continue
        if source_key not in record or target_key not in record:
            raise RuntimeError(
                "LightRAG relationship record has an incomplete endpoint pair"
            )
        _require_nonempty_string(record, (source_key,), "relationship endpoint")
        _require_nonempty_string(record, (target_key,), "relationship endpoint")
        endpoints.append(
            (
                _canonical_identifier(record[source_key]),
                _canonical_identifier(record[target_key]),
            )
        )
    if not endpoints:
        raise RuntimeError(
            "LightRAG relationship record is missing a supported endpoint pair"
        )
    if any(endpoint != endpoints[0] for endpoint in endpoints[1:]):
        raise RuntimeError(
            "LightRAG relationship record has conflicting endpoint pairs"
        )


def _validate_chunk(record: Mapping[str, Any]) -> None:
    present = [key for key in _CHUNK_TEXT_KEYS if key in record]
    if not present:
        raise RuntimeError(
            f"LightRAG chunk record is missing a supported field: {_CHUNK_TEXT_KEYS}"
        )
    for key in present:
        if not isinstance(record[key], str):
            raise RuntimeError(f"LightRAG chunk field '{key}' must be a string")


def parse_lightrag_result(result: Any) -> StructuredRetrieval:
    """Parse only LightRAG results that expose all structured context lists."""
    if not isinstance(result, Mapping):
        raise RuntimeError("LightRAG did not return a structured mapping")

    raw_data = result.get("raw_data")
    direct_data = result.get("data")
    if isinstance(direct_data, Mapping):
        payload = direct_data
    elif isinstance(raw_data, Mapping) and isinstance(raw_data.get("data"), Mapping):
        payload = raw_data["data"]
    elif isinstance(raw_data, Mapping):
        payload = raw_data
    else:
        raise RuntimeError("LightRAG result is missing structured data")

    entities = payload.get("entities")
    relations = payload.get("relationships")
    if relations is None:
        relations = payload.get("relations")
    chunks = payload.get("chunks")
    if not isinstance(entities, list) or not isinstance(relations, list) or not isinstance(chunks, list):
        raise RuntimeError(
            "LightRAG structured data must contain entities, relationships, and chunks lists"
        )
    if not all(isinstance(record, Mapping) for records in (entities, relations, chunks) for record in records):
        raise RuntimeError("LightRAG structured data lists must contain mapping records")
    for entity in entities:
        _validate_entity(entity)
    for relation in relations:
        _validate_relation(relation)
    for chunk in chunks:
        _validate_chunk(chunk)

    context = result.get("content")
    if not isinstance(context, str) and isinstance(raw_data, Mapping):
        context = raw_data.get("content")
    if not isinstance(context, str):
        context = ""

    return StructuredRetrieval(
        entities=deepcopy(entities),
        relations=deepcopy(relations),
        chunks=deepcopy(chunks),
        context=context,
    )


def render_lightrag_context(retrieval: StructuredRetrieval) -> str:
    """Render filtered records into deterministic JSON-lines sections."""
    if not retrieval.entities and not retrieval.relations and not retrieval.chunks:
        return ""

    lines = ["Entities"]
    lines.extend(
        json.dumps(record, ensure_ascii=False, sort_keys=True)
        for record in retrieval.entities
    )
    lines.append("Relationships")
    lines.extend(
        json.dumps(record, ensure_ascii=False, sort_keys=True)
        for record in retrieval.relations
    )
    lines.append("Sources")
    lines.extend(
        json.dumps(record, ensure_ascii=False, sort_keys=True)
        for record in retrieval.chunks
    )
    return "\n".join(lines)
