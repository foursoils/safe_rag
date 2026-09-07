from typing import Any

__all__ = ["ExperimentSpec", "load_experiment", "run_experiment"]


def __getattr__(name: str) -> Any:
    if name in {"ExperimentSpec", "load_experiment", "run_experiment"}:
        from safe_rag.eval.runner import ExperimentSpec, load_experiment, run_experiment

        mapping = {
            "ExperimentSpec": ExperimentSpec,
            "load_experiment": load_experiment,
            "run_experiment": run_experiment,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
