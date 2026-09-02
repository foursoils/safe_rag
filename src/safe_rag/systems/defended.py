from safe_rag.defenses.base import Defense, QueryDecision
from safe_rag.systems.base import OriginalGraph, QueryResult, VictimSystem


class DefendedSystem:
    """Apply defense hooks around a victim RAG system."""

    def __init__(self, system: VictimSystem, defense: Defense):
        self.system = system
        self.defense = defense
        self.name = getattr(system, "name", "unknown")
        self.graph_backend = getattr(system, "graph_backend", "graphrag")
        self._turn = 0
        self._initialized = False
        self._original_graph: OriginalGraph | None = None

    def _require_staged_methods(self) -> None:
        if not callable(getattr(self.system, "retrieve", None)) or not callable(
            getattr(self.system, "generate", None)
        ):
            raise RuntimeError(
                f"Defense '{getattr(self.defense, 'name', 'unknown')}' requires "
                "a victim with staged retrieve() and generate() methods"
            )

    def query(self, text: str, method: str = "local") -> QueryResult:
        from safe_rag.defenses.base import DefenseContext

        self._turn += 1
        ctx = DefenseContext(turn=self._turn, method=method)
        decision = self.defense.on_query(text, ctx)
        if decision is None:
            decision = QueryDecision(action="allow", query=text)
        if decision.action == "block":
            return QueryResult(
                response="",
                blocked=True,
                block_reason=decision.reason or "blocked by defense",
                extra={"defense": {"turn": self._turn}},
            )

        outgoing = decision.query if decision.action == "rewrite" else text
        if not getattr(self.defense, "requires_staged_query", False):
            result = self.system.query(outgoing, method=method)
            result.retrieved_context = self.defense.on_retrieve(result.retrieved_context, ctx)
            result.response = self.defense.on_response(result.response, ctx)
            return result

        self._require_staged_methods()
        retrieval = self.system.retrieve(outgoing, method=method)
        filtered = self.defense.on_structured_retrieve(retrieval, ctx)
        result = self.system.generate(outgoing, filtered, method=method)
        self.defense.on_disclose(filtered, ctx)
        result.response = self.defense.on_response(result.response, ctx)
        result.extra["defense"] = {**filtered.extra, "turn": self._turn}
        return result

    def load_original_graph(self, dataset: str) -> OriginalGraph:
        if self._initialized and self._original_graph is not None:
            return self._original_graph

        original = self.system.load_original_graph(dataset)
        if getattr(self.defense, "requires_staged_query", False):
            self._require_staged_methods()
        self.defense.initialize(original)
        self._initialized = True
        self._original_graph = original
        return original
