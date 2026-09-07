from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

from safe_rag.eval.utility.aggregate import relative_delta

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


def _load_utility_metrics(directory: Path) -> dict[str, Any]:
    metrics_path = Path(directory) / "utility_metrics.json"
    summary_path = Path(directory) / "run_summary.json"
    payload: dict[str, Any] = {}
    if metrics_path.is_file():
        payload.update(json.loads(metrics_path.read_text(encoding="utf-8")))
    elif summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        payload.update(summary.get("metrics") or {})
        payload.setdefault("defense", summary.get("defense"))
    else:
        raise FileNotFoundError(f"No utility_metrics.json or run_summary.json in {directory}")
    payload["path"] = str(directory)
    return payload


def compare_utility_runs(result_dirs: list[Path], dest: Path) -> Path:
    runs = [_load_utility_metrics(Path(directory)) for directory in result_dirs]
    payload: dict[str, Any] = {"runs": runs}
    if len(runs) >= 2:
        baseline = runs[0]
        treated = runs[1]
        protocol_delta = {}
        for protocol in sorted(set(baseline.get("by_protocol", {})) & set(treated.get("by_protocol", {}))):
            base_proto = baseline["by_protocol"][protocol]
            treated_proto = treated["by_protocol"][protocol]
            turn_delta = {}
            for turn in sorted(
                set(base_proto.get("by_session_turn", {})) & set(treated_proto.get("by_session_turn", {})),
                key=int,
            ):
                turn_delta[turn] = relative_delta(
                    base_proto["by_session_turn"][turn],
                    treated_proto["by_session_turn"][turn],
                )
            protocol_delta[protocol] = {
                "overall": relative_delta(base_proto, treated_proto),
                "by_session_turn": turn_delta,
            }
        payload["delta"] = {
            "baseline": baseline.get("defense") or baseline.get("path"),
            "treated": treated.get("defense") or treated.get("path"),
            "overall": relative_delta(baseline.get("overall") or {}, treated.get("overall") or {}),
            "by_protocol": protocol_delta,
        }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare safe_RAG experiment outputs")
    parser.add_argument("--compare", nargs="+", required=True, help="Result directories")
    parser.add_argument("--dest", required=True, help="Output JSON path")
    parser.add_argument(
        "--utility",
        action="store_true",
        help="Compare utility_metrics.json and write relative deltas",
    )
    args = parser.parse_args()
    dest = Path(args.dest)
    directories = [Path(item) for item in args.compare]
    if args.utility:
        compare_utility_runs(directories, dest)
    else:
        compare_runs(directories, dest)
    print(dest)


if __name__ == "__main__":
    main()
