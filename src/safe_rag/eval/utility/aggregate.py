from __future__ import annotations

from collections import defaultdict
from typing import Any

from safe_rag.eval.utility.score import SCORE_KEYS


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _score_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        key: _mean([float(row["scores"][key]) for row in rows if key in row.get("scores", {})])
        for key in SCORE_KEYS
    }


def _field_means(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        stats = row.get(field)
        if not isinstance(stats, dict):
            continue
        for key, value in stats.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            buckets[key].append(float(value))
    return {key: _mean(values) for key, values in sorted(buckets.items())}


def _isolation_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    return _field_means(rows, "isolation")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "blocked": sum(1 for row in rows if row.get("blocked")),
        **_score_means(rows),
        "isolation": _isolation_means(rows),
        "detector": _field_means(rows, "detector"),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_protocol: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row.get("protocol") or "unknown")].append(row)

    for protocol, rows in grouped.items():
        by_type: dict[str, dict[str, Any]] = {}
        by_turn: dict[str, dict[str, Any]] = {}
        type_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        turn_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            type_groups[str(row.get("question_type") or "unknown")].append(row)
            turn_groups[int(row.get("session_turn") or 1)].append(row)
        for name, group in type_groups.items():
            by_type[name] = summarize_rows(group)
        for turn, group in sorted(turn_groups.items()):
            by_turn[str(turn)] = summarize_rows(group)
        by_protocol[protocol] = {
            **summarize_rows(rows),
            "by_type": by_type,
            "by_session_turn": by_turn,
        }

    return {
        "n": len(records),
        "overall": summarize_rows(records),
        "by_protocol": by_protocol,
    }


def relative_delta(baseline: dict[str, Any], treated: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key in SCORE_KEYS:
        base = float(baseline.get(key) or 0.0)
        value = float(treated.get(key) or 0.0)
        abs_delta = value - base
        rel = abs_delta / base if base else None
        delta[key] = {"baseline": base, "treated": value, "delta": abs_delta, "relative": rel}
    return delta
