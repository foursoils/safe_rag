"""Attack registry."""

from importlib import import_module
from typing import Any

from safe_rag.attacks.base import Attack

_ATTACKS: dict[str, str] = {
    "agea": "safe_rag.attacks.agea.attack:AgeaAttack",
}


def register_attack(name: str, dotted: str) -> None:
    _ATTACKS[name] = dotted


def list_attacks() -> list[str]:
    return sorted(_ATTACKS)


def get_attack(name: str, **kwargs: Any) -> Attack:
    if name not in _ATTACKS:
        raise KeyError(f"Unknown attack '{name}'. Available: {list_attacks()}")
    module_name, class_name = _ATTACKS[name].split(":")
    cls = getattr(import_module(module_name), class_name)
    return cls(**kwargs)
