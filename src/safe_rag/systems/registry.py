"""Victim RAG system registry."""

from importlib import import_module
from typing import Any

from safe_rag.systems.base import VictimSystem

_SYSTEMS: dict[str, str] = {
    "graphrag": "safe_rag.systems.graphrag.system:GraphRAGSystem",
    "lightrag": "safe_rag.systems.lightrag.system:LightRAGSystem",
}


def register_system(name: str, dotted: str) -> None:
    _SYSTEMS[name] = dotted


def list_systems() -> list[str]:
    return sorted(_SYSTEMS)


def get_system(name: str, **kwargs: Any) -> VictimSystem:
    if name not in _SYSTEMS:
        raise KeyError(f"Unknown system '{name}'. Available: {list_systems()}")
    module_name, class_name = _SYSTEMS[name].split(":")
    cls = getattr(import_module(module_name), class_name)
    return cls(**kwargs)
