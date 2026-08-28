"""Defense registry."""

from typing import Any

from safe_rag.defenses.base import Defense
from safe_rag.defenses.none.passthrough import NoneDefense

_DEFENSES: dict[str, type] = {
    "none": NoneDefense,
}


def register_defense(name: str, cls: type) -> None:
    _DEFENSES[name] = cls


def list_defenses() -> list[str]:
    return sorted(_DEFENSES)


def get_defense(name: str, **kwargs: Any) -> Defense:
    if name not in _DEFENSES:
        raise KeyError(f"Unknown defense '{name}'. Available: {list_defenses()}")
    return _DEFENSES[name](**kwargs)
