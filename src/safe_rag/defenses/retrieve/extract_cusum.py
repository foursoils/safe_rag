from __future__ import annotations

from typing import Any

import networkx as nx


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def closed_neighborhood(topology: nx.Graph, nodes: set[str], radius: int) -> set[str]:
    """Nodes within `radius` hops of `nodes`, including `nodes` themselves."""
    if radius <= 0:
        return set(nodes)
    closed = set(nodes)
    topology_nodes = set(topology.nodes)
    for node in nodes & topology_nodes:
        closed.update(
            nx.single_source_shortest_path_length(topology, node, cutoff=radius)
        )
    return closed


def partition_retrieved(
    retrieved: set[str],
    disclosed: set[str],
    topology: nx.Graph,
    radius: int = 1,
) -> dict[str, float]:
    """Split R into disclosed / r-hop belt / outside the belt."""
    if not retrieved:
        return {"in_d": 0.0, "belt": 0.0, "far": 0.0}
    total = len(retrieved)
    in_d = retrieved & disclosed
    closed = closed_neighborhood(topology, disclosed, radius)
    belt = retrieved & (closed - disclosed)
    far = retrieved - closed
    return {
        "in_d": len(in_d) / total,
        "belt": len(belt) / total,
        "far": len(far) / total,
    }


class ExtractCUSUM:
    """One-sided CUSUM. Negative drift is scaled by `decay` so a dip does not wipe the score."""

    def __init__(
        self,
        mu0: float = 0.08,
        slack: float = 0.04,
        h_radius_2: float = 1.0,
        decay: float = 0.35,
    ):
        if slack < 0:
            raise ValueError("CUSUM slack must be >= 0")
        if h_radius_2 <= 0:
            raise ValueError("CUSUM h_radius_2 must be > 0")
        if not 0.0 < decay <= 1.0:
            raise ValueError("CUSUM decay must be in (0, 1]")
        self.mu0 = float(mu0)
        self.slack = float(slack)
        self.h_radius_2 = float(h_radius_2)
        self.decay = float(decay)
        self.reset()

    def reset(self) -> None:
        self.score = 0.0

    def observe(self, value: float, *, accumulate: bool = True) -> dict[str, Any]:
        value = _clip01(value)
        drift = value - self.mu0 - self.slack
        if accumulate:
            step = drift if drift >= 0 else self.decay * drift
            self.score = max(0.0, self.score + step)
        return {
            "x": value,
            "drift": drift,
            "score": self.score,
            "p": _clip01(self.score / self.h_radius_2),
            "mu0": self.mu0,
            "slack": self.slack,
            "h_radius_2": self.h_radius_2,
            "decay": self.decay,
        }
