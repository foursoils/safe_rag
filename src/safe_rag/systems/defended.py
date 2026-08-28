from safe_rag.defenses.base import Defense, QueryDecision
from safe_rag.systems.base import QueryResult, VictimSystem


class DefendedSystem:
    """Apply defense hooks around a victim RAG system."""

    def __init__(self, system: VictimSystem, defense: Defense):
        self.system = system
        self.defense = defense
        self.name = getattr(system, "name", "unknown")
        self.graph_backend = getattr(system, "graph_backend", "graphrag")

    def query(self, text: str, method: str = "local") -> QueryResult:
        from safe_rag.defenses.base import DefenseContext

        ctx = DefenseContext(method=method)
        decision = self.defense.on_query(text, ctx)
        if decision is None:
            decision = QueryDecision(action="allow", query=text)
        if decision.action == "block":
            return QueryResult(
                response="",
                blocked=True,
                block_reason=decision.reason or "blocked by defense",
            )

        outgoing = decision.query if decision.action == "rewrite" else text
        result = self.system.query(outgoing, method=method)
        result.retrieved_context = self.defense.on_retrieve(result.retrieved_context, ctx)
        result.response = self.defense.on_response(result.response, ctx)
        return result

    def load_original_graph(self, dataset: str):
        return self.system.load_original_graph(dataset)
