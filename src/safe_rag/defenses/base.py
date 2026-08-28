from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol


@dataclass
class DefenseContext:
    turn: int = 0
    method: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryDecision:
    action: Literal["allow", "rewrite", "block"]
    query: str
    reason: str = ""


class Defense(Protocol):
    name: str

    def on_query(self, query: str, ctx: DefenseContext) -> QueryDecision: ...

    def on_retrieve(self, context: str, ctx: DefenseContext) -> str: ...

    def on_response(self, response: str, ctx: DefenseContext) -> str: ...

    def on_index(self, graph: Any, ctx: DefenseContext) -> Any: ...


class PassthroughDefense:
    """Default no-op hooks. Concrete defenses override only the layers they use."""

    name = "passthrough"

    def on_query(self, query: str, ctx: DefenseContext) -> QueryDecision:
        del ctx
        return QueryDecision(action="allow", query=query)

    def on_retrieve(self, context: str, ctx: DefenseContext) -> str:
        del ctx
        return context

    def on_response(self, response: str, ctx: DefenseContext) -> str:
        del ctx
        return response

    def on_index(self, graph: Any, ctx: DefenseContext) -> Any:
        del ctx
        return graph
