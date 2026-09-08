from __future__ import annotations

from collections import deque
import math
from typing import Any, Optional, Sequence

import networkx as nx
import numpy as np

SETTLE = 0
SWITCH = 1
EXTRACT = 2
STATE_NAMES = ("settle", "switch", "extract")

# Hand-set priors: switch is a brief high-novelty bump; extract stays high.
DEFAULT_START = (0.45, 0.45, 0.10)
DEFAULT_TRANSITION = (
    (0.70, 0.25, 0.05),
    (0.55, 0.20, 0.25),
    (0.08, 0.07, 0.85),
)
# Columns: novelty, unexplored-outside-R. Settle is low novelty; the other two are high.
DEFAULT_EMIT_MEAN = (
    (0.25, 0.80),
    (0.80, 0.85),
    (0.75, 0.85),
)
DEFAULT_EMIT_STD = (
    (0.15, 0.15),
    (0.15, 0.12),
    (0.12, 0.12),
)


def novelty_rate(retrieved: set[str], disclosed: set[str]) -> float:
    if not retrieved:
        return 0.0
    return len(retrieved - disclosed) / len(retrieved)


def open_neighborhood(topology: nx.Graph, nodes: set[str]) -> set[str]:
    neighbors: set[str] = set()
    for node in nodes:
        if node in topology:
            neighbors.update(topology.neighbors(node))
    return neighbors - nodes


def unexplored_rate(retrieved: set[str], disclosed: set[str], topology: nx.Graph) -> float:
    neighbors = open_neighborhood(topology, retrieved)
    if not neighbors:
        return 0.0
    return len(neighbors - disclosed) / len(neighbors)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _log_normal(value: float, mean: float, std: float) -> float:
    std = max(float(std), 1e-6)
    z = (value - mean) / std
    return -0.5 * math.log(2.0 * math.pi) - math.log(std) - 0.5 * z * z


def _logsumexp(values: np.ndarray) -> float:
    top = float(np.max(values))
    return top + math.log(math.fsum(np.exp(values - top)))


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=1, keepdims=True)
    totals = np.where(totals <= 0, 1.0, totals)
    return matrix / totals


class ExtractHMM:
    """Online 3-state HMM. Observation is the sliding-window mean of (novelty, unexplored)."""

    def __init__(
        self,
        window_size: int = 3,
        start: Optional[Sequence[float]] = None,
        transition: Optional[Sequence[Sequence[float]]] = None,
        emit_mean: Optional[Sequence[Sequence[float]]] = None,
        emit_std: Optional[Sequence[Sequence[float]]] = None,
    ):
        if window_size < 1:
            raise ValueError("HMM window_size must be >= 1")
        self.window_size = int(window_size)
        start_vec = np.asarray(DEFAULT_START if start is None else start, dtype=float)
        if start_vec.shape != (3,):
            raise ValueError("HMM start must have 3 state probabilities")
        start_vec = np.clip(start_vec, 0.0, None)
        total = float(start_vec.sum())
        if total <= 0:
            raise ValueError("HMM start probabilities must sum to a positive value")
        self.start = start_vec / total
        transition_mat = np.asarray(
            DEFAULT_TRANSITION if transition is None else transition, dtype=float
        )
        if transition_mat.shape != (3, 3):
            raise ValueError("HMM transition must be 3x3")
        self.transition = _row_normalize(np.clip(transition_mat, 0.0, None))
        means = np.asarray(DEFAULT_EMIT_MEAN if emit_mean is None else emit_mean, dtype=float)
        stds = np.asarray(DEFAULT_EMIT_STD if emit_std is None else emit_std, dtype=float)
        if means.shape != (3, 2) or stds.shape != (3, 2):
            raise ValueError("HMM emit_mean and emit_std must be 3x2")
        self.emit_mean = means
        self.emit_std = np.clip(stds, 1e-6, None)
        self.reset()

    def reset(self) -> None:
        self.window: deque[tuple[float, float]] = deque(maxlen=self.window_size)
        self._log_alpha: Optional[np.ndarray] = None

    def observe(self, novelty: float, unexplored: float) -> dict[str, Any]:
        novelty = _clip01(novelty)
        unexplored = _clip01(unexplored)
        self.window.append((novelty, unexplored))
        window_novelty = sum(item[0] for item in self.window) / len(self.window)
        window_unexplored = sum(item[1] for item in self.window) / len(self.window)
        log_emit = np.array(
            [
                _log_normal(window_novelty, self.emit_mean[state, 0], self.emit_std[state, 0])
                + _log_normal(
                    window_unexplored, self.emit_mean[state, 1], self.emit_std[state, 1]
                )
                for state in range(3)
            ],
            dtype=float,
        )
        if self._log_alpha is None:
            log_alpha = np.log(self.start + 1e-12) + log_emit
        else:
            log_alpha = np.empty(3, dtype=float)
            log_transition = np.log(self.transition + 1e-12)
            for next_state in range(3):
                log_alpha[next_state] = (
                    _logsumexp(self._log_alpha + log_transition[:, next_state])
                    + log_emit[next_state]
                )
        self._log_alpha = log_alpha - _logsumexp(log_alpha)
        probs = np.exp(self._log_alpha)
        probs = probs / max(float(probs.sum()), 1e-12)
        return {
            "novelty": novelty,
            "unexplored": unexplored,
            "window_novelty": window_novelty,
            "window_unexplored": window_unexplored,
            "window_size": len(self.window),
            "p_settle": float(probs[SETTLE]),
            "p_switch": float(probs[SWITCH]),
            "p_extract": float(probs[EXTRACT]),
            "p": float(probs[EXTRACT]),
        }
