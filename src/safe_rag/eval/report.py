from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from safe_rag.eval.runner import ExperimentSpec


def write_run_summary(output_dir: Path, spec: ExperimentSpec, metrics: dict[str, Any]) -> Path:
    summary = {
        "attack": spec.attack,
        "defense": spec.defense,
        "system": spec.system,
        "dataset": spec.dataset,
        "turns": spec.turns,
        "metrics": metrics,
    }
    path = Path(output_dir) / "run_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def compare_runs(result_dirs: list[Path], dest: Path) -> Path:
    rows = []
    for directory in result_dirs:
        summary_path = Path(directory) / "run_summary.json"
        analysis_path = Path(directory) / "extraction_analysis.json"
        payload: dict[str, Any] = {}
        if summary_path.exists():
            payload.update(json.loads(summary_path.read_text(encoding="utf-8")))
        elif analysis_path.exists():
            payload.update(json.loads(analysis_path.read_text(encoding="utf-8")))
        payload["path"] = str(directory)
        rows.append(payload)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"runs": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest
