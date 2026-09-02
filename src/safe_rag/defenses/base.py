from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol

from safe_rag.systems.base import OriginalGraph, StructuredRetrieval


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
    requires_staged_query: bool

    def initialize(self, graph: OriginalGraph) -> None: ...

    def on_query(self, query: str, ctx: DefenseContext) -> QueryDecision: ...

    def on_retrieve(self, context: str, ctx: DefenseContext) -> str: ...

    def on_structured_retrieve(
        self,
        retrieval: StructuredRetrieval,
        ctx: DefenseContext,
    ) -> StructuredRetrieval: ...

    def on_disclose(self, retrieval: StructuredRetrieval, ctx: DefenseContext) -> None: ...

    def on_response(self, response: str, ctx: DefenseContext) -> str: ...

    def on_index(self, graph: Any, ctx: DefenseContext) -> Any: ...


class PassthroughDefense:
    """Default no-op hooks. Concrete defenses override only the layers they use."""

    name = "passthrough"
    requires_staged_query = False

    def initialize(self, graph: OriginalGraph) -> None:
        del graph

    def on_query(self, query: str, ctx: DefenseContext) -> QueryDecision:
        del ctx
        return QueryDecision(action="allow", query=query)

    def on_retrieve(self, context: str, ctx: DefenseContext) -> str:
        del ctx
        return context

    def on_structured_retrieve(
        self,
        retrieval: StructuredRetrieval,
        ctx: DefenseContext,
    ) -> StructuredRetrieval:
        del ctx
        return retrieval

    def on_disclose(self, retrieval: StructuredRetrieval, ctx: DefenseContext) -> None:
        del retrieval, ctx

    def on_response(self, response: str, ctx: DefenseContext) -> str:
        del ctx
        return response

    def on_index(self, graph: Any, ctx: DefenseContext) -> Any:
        del ctx
        return graph
