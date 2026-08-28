from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from safe_rag.attacks.base import AttackResult
from safe_rag.attacks.registry import get_attack
from safe_rag.defenses.registry import get_defense
from safe_rag.eval.report import write_run_summary
from safe_rag.paths import RESULTS_ROOT
from safe_rag.systems.registry import get_system

KNOWN_KEYS = {"attack", "defense", "system", "dataset", "turns"}


@dataclass
class ExperimentSpec:
    attack: str
    defense: str
    system: str
    dataset: str
    turns: int = 50
    extra: dict[str, Any] = field(default_factory=dict)


def load_experiment(path: str | Path) -> ExperimentSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    missing = [key for key in ("attack", "defense", "system", "dataset") if key not in raw]
    if missing:
        raise ValueError(f"Experiment config missing keys: {missing}")
    extra = {key: value for key, value in raw.items() if key not in KNOWN_KEYS}
    return ExperimentSpec(
        attack=raw["attack"],
        defense=raw["defense"],
        system=raw["system"],
        dataset=raw["dataset"],
        turns=int(raw.get("turns", 50)),
        extra=extra,
    )


def run_experiment(
    config_path: str | Path,
    results_root: Optional[str | Path] = None,
    system=None,
) -> AttackResult:
    spec = load_experiment(config_path)
    results_root = Path(results_root) if results_root is not None else RESULTS_ROOT
    output_dir = results_root / f"{spec.attack}_{spec.defense}_{spec.system}_{spec.dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)

    attack = get_attack(spec.attack)
    defense = get_defense(spec.defense)
    if system is None:
        system_kwargs = dict(spec.extra.get("system_kwargs") or {})
        system_kwargs.setdefault("dataset", spec.dataset)
        system = get_system(spec.system, **system_kwargs)

    result = attack.run(
        system=system,
        defense=defense,
        dataset=spec.dataset,
        budget=spec.turns,
        output_dir=output_dir,
        config=spec.extra,
    )
    write_run_summary(result.output_dir, spec, result.metrics)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a safe_RAG attack × defense × system experiment")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument("--results-root", default=None, help="Override results directory")
    args = parser.parse_args()
    result = run_experiment(args.config, results_root=args.results_root)
    print(json.dumps({"output_dir": str(result.output_dir), "metrics": result.metrics}, indent=2, default=str))


if __name__ == "__main__":
    main()
