from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class QueryResult:
    response: str
    retrieved_context: str = ""
    stderr: str = ""
    blocked: bool = False
    block_reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredRetrieval:
    entities: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    context: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OriginalGraph:
    filtered_nodes: set[str]
    original_nodes: set[str]
    original_edges: set[tuple[str, str]]
    stats: dict[str, Any] = field(default_factory=dict)


class VictimSystem(Protocol):
    name: str
    graph_backend: str

    def query(self, text: str, method: str = "local") -> QueryResult: ...

    def load_original_graph(self, dataset: str) -> OriginalGraph: ...


class StagedVictimSystem(VictimSystem, Protocol):
    def retrieve(self, text: str, method: str = "local") -> StructuredRetrieval: ...

    def generate(
        self,
        text: str,
        retrieval: StructuredRetrieval,
        method: str = "local",
    ) -> QueryResult: ...
