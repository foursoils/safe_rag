from __future__ import annotations

import unittest

from safe_rag.defenses.base import DefenseContext
from safe_rag.defenses.registry import get_defense, list_defenses
from safe_rag.defenses.retrieve.extract_cusum import ExtractCUSUM, partition_retrieved
from safe_rag.systems.base import OriginalGraph, StructuredRetrieval


def _graph():
    import networkx as nx

    g = nx.Graph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")])
    return g


class PartitionTests(unittest.TestCase):
    def test_empty_d_is_all_far(self):
        parts = partition_retrieved({"D", "E"}, set(), _graph(), radius=1)
        self.assertEqual(parts["far"], 1.0)
        self.assertEqual(parts["belt"], 0.0)
        self.assertEqual(parts["in_d"], 0.0)

    def test_belt_vs_far(self):
        disclosed = {"B"}
        belt = partition_retrieved({"A", "C"}, disclosed, _graph(), radius=1)
        self.assertEqual(belt["belt"], 1.0)
        self.assertEqual(belt["far"], 0.0)
        far = partition_retrieved({"D", "E"}, disclosed, _graph(), radius=1)
        self.assertEqual(far["far"], 1.0)
        self.assertEqual(far["belt"], 0.0)
        inside = partition_retrieved({"B"}, disclosed, _graph(), radius=1)
        self.assertEqual(inside["in_d"], 1.0)


class CusumScoreTests(unittest.TestCase):
    def test_negative_drift_is_damped(self):
        fast = ExtractCUSUM(mu0=0.25, slack=0.08, h_radius_2=1.2, decay=1.0)
        slow = ExtractCUSUM(mu0=0.25, slack=0.08, h_radius_2=1.2, decay=0.35)
        fast.observe(0.9)
        slow.observe(0.9)
        fast.observe(0.1)
        slow.observe(0.1)
        self.assertGreater(slow.score, fast.score)

    def test_sustained_high_grows_score(self):
        detector = ExtractCUSUM(mu0=0.08, slack=0.04, h_radius_2=1.0, decay=0.35)
        for _ in range(12):
            detector.observe(0.25)
        self.assertGreaterEqual(detector.score, 1.0)

    def test_warmup_does_not_accumulate(self):
        detector = ExtractCUSUM()
        detector.observe(1.0, accumulate=False)
        detector.observe(1.0, accumulate=False)
        self.assertEqual(detector.score, 0.0)
        stepped = detector.observe(0.5, accumulate=True)
        self.assertGreater(stepped["score"], 0)

    def test_rejects_bad_thresholds(self):
        with self.assertRaises(ValueError):
            ExtractCUSUM(h_radius_2=0.0)
        with self.assertRaises(ValueError):
            ExtractCUSUM(decay=0.0)


def _entities(*names: str) -> list[dict[str, str]]:
    return [{"entity": name} for name in names]


def _line_graph() -> OriginalGraph:
    return OriginalGraph(
        filtered_nodes={"A", "B", "C", "D", "E"},
        original_nodes={"A", "B", "C", "D", "E"},
        original_edges={("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")},
    )


class CusumDefenseTests(unittest.TestCase):
    def test_warmup_then_floor_then_r2_on_far(self):
        defense = get_defense("cusum", decide_after=2, hop=1, settle_patience=3)
        defense.initialize(_line_graph())
        local = StructuredRetrieval(entities=_entities("A", "B"))
        far = StructuredRetrieval(entities=_entities("D", "E"))

        first = defense.on_structured_retrieve(local, DefenseContext(turn=1))
        self.assertEqual(first.extra["detector"]["radius"], 0)
        defense.on_response("mention A and B here", DefenseContext(turn=1))

        second = defense.on_structured_retrieve(local, DefenseContext(turn=2))
        self.assertEqual(second.extra["detector"]["radius"], 0)
        defense.on_response("still A and B", DefenseContext(turn=2))

        radii = []
        for turn in range(3, 12):
            filtered = defense.on_structured_retrieve(far, DefenseContext(turn=turn))
            radii.append(filtered.extra["detector"]["radius"])
            defense.on_response("no matching labels", DefenseContext(turn=turn))
        self.assertGreaterEqual(radii[0], 1)
        self.assertEqual(radii[-1], 2)
        self.assertIn("cusum", list_defenses())

    def test_floor_r1_until_stagnant(self):
        defense = get_defense(
            "cusum",
            decide_after=2,
            hop=1,
            settle_eps=0.05,
            settle_patience=3,
        )
        defense.initialize(_line_graph())
        local = StructuredRetrieval(entities=_entities("A", "B"))
        defense.on_structured_retrieve(local, DefenseContext(turn=1))
        defense.on_response("mention A and B here", DefenseContext(turn=1))
        defense.on_structured_retrieve(local, DefenseContext(turn=2))
        defense.on_response("still A and B", DefenseContext(turn=2))

        radii = []
        for turn in range(3, 7):
            filtered = defense.on_structured_retrieve(local, DefenseContext(turn=turn))
            radii.append(filtered.extra["detector"]["radius"])
            defense.on_response("mention A and B here", DefenseContext(turn=turn))
        self.assertEqual(radii[:2], [1, 1])
        self.assertEqual(radii[-1], 0)


if __name__ == "__main__":
    unittest.main()
