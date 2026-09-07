from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from safe_rag.attacks.base import AttackResult
from safe_rag.defenses.base import DefenseContext, QueryDecision
from safe_rag.eval.utility.aggregate import summarize_records
from safe_rag.eval.utility.questions import Question, build_utility_plan, load_questions
from safe_rag.eval.utility.score import score_answer
from safe_rag.logutil import redirect_library_logs
from safe_rag.paths import DATA_ROOT, LOGS_ROOT
from safe_rag.systems.base import OriginalGraph, QueryResult
from safe_rag.systems.defended import DefendedSystem


def _default_questions_path(dataset: str) -> Path:
    return DATA_ROOT / "qa" / f"{dataset}_questions.json"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _session_complete(history: list[dict[str, Any]], protocol: str, session_id: str, size: int) -> bool:
    turns = {
        int(row["session_turn"])
        for row in history
        if row.get("protocol") == protocol and row.get("session_id") == session_id
    }
    return size > 0 and turns == set(range(1, size + 1))


def _drop_incomplete_session(history: list[dict[str, Any]], protocol: str, session_id: str) -> list[dict[str, Any]]:
    return [
        row
        for row in history
        if not (row.get("protocol") == protocol and row.get("session_id") == session_id)
    ]


def _clone_defense(defense):
    kwargs: dict[str, Any] = {}
    radius = getattr(defense, "radius", None)
    if radius is not None:
        kwargs["radius"] = radius
    return type(defense)(**kwargs)


def _query_turn(system, defense, text: str, method: str, turn: int) -> QueryResult:
    ctx = DefenseContext(turn=turn, method=method)
    decision = defense.on_query(text, ctx)
    if decision is None:
        decision = QueryDecision(action="allow", query=text)
    if decision.action == "block":
        return QueryResult(
            response="",
            blocked=True,
            block_reason=decision.reason or "blocked by defense",
            extra={"defense": {"turn": turn}},
        )
    outgoing = decision.query if decision.action == "rewrite" else text
    if callable(getattr(system, "retrieve", None)) and callable(getattr(system, "generate", None)):
        retrieval = system.retrieve(outgoing, method=method)
        filtered = defense.on_structured_retrieve(retrieval, ctx)
        result = system.generate(outgoing, filtered, method=method)
        defense.on_disclose(filtered, ctx)
        result.response = defense.on_response(result.response, ctx)
        result.extra = dict(result.extra or {})
        result.extra["defense"] = {**(filtered.extra or {}), "turn": turn}
        return result
    result = system.query(outgoing, method=method)
    result.retrieved_context = defense.on_retrieve(result.retrieved_context, ctx)
    result.response = defense.on_response(result.response, ctx)
    result.extra = dict(result.extra or {})
    result.extra["defense"] = {"turn": turn}
    return result


def _score_turn(
    *,
    protocol: str,
    session_id: str,
    session_turn: int,
    session_size: int,
    item: Question,
    result: QueryResult,
    latency_s: float,
) -> dict[str, Any]:
    prediction = "" if result.blocked else (result.response or "")
    defense_extra = result.extra.get("defense") if isinstance(result.extra, dict) else {}
    isolation = defense_extra.get("isolation") if isinstance(defense_extra, dict) else None
    return {
        "protocol": protocol,
        "session_id": session_id,
        "session_turn": session_turn,
        "session_size": session_size,
        "question_id": item.id,
        "question_type": item.question_type,
        "question": item.question,
        "gold_answer": item.answer,
        "prediction": prediction,
        "blocked": bool(result.blocked),
        "block_reason": result.block_reason,
        "scores": score_answer(prediction, item.answer),
        "isolation": isolation,
        "defense_stats": defense_extra or {},
        "stderr": result.stderr,
        "latency_s": latency_s,
    }


class UtilityAttack:
    name = "utility"

    def run(
        self,
        system,
        defense,
        dataset: str,
        budget: int,
        output_dir: Path,
        config: Optional[dict[str, Any]] = None,
    ) -> AttackResult:
        config = config or {}
        utility_cfg = dict(config.get("utility") or {})
        questions_path = Path(utility_cfg["questions"]) if utility_cfg.get("questions") else _default_questions_path(dataset)
        if not questions_path.is_file():
            raise FileNotFoundError(f"Utility QA file not found: {questions_path}")

        query_method = utility_cfg.get("query_method", "hybrid")
        enable_resume = bool(utility_cfg.get("resume", False))
        protocols = utility_cfg.get("protocols") or ["p1"]
        if isinstance(protocols, str):
            protocols = [protocols]
        session_size = int(utility_cfg.get("session_size", 4))
        session_workers = max(1, int(utility_cfg.get("session_workers", 4)))
        seed = int(utility_cfg.get("seed", 42))
        min_overlap = float(utility_cfg.get("min_overlap", 0.22))
        dedupe = bool(utility_cfg.get("dedupe", True))
        question_types = utility_cfg.get("question_types")

        plan = build_utility_plan(
            load_questions(questions_path),
            question_types=question_types,
            max_questions=budget,
            session_size=session_size,
            seed=seed,
            protocols=protocols,
            min_overlap=min_overlap,
            dedupe=dedupe,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = Path(utility_cfg["log_dir"]) if utility_cfg.get("log_dir") else LOGS_ROOT / output_dir.name
        os.environ.setdefault("LOG_DIR", str(log_dir))
        redirect_library_logs(log_dir / "lightrag.log")
        turn_log_dir = log_dir / "turn_logs"
        turn_log_dir.mkdir(parents=True, exist_ok=True)

        history_path = output_dir / "query_history.json"
        history: list[dict[str, Any]] = []
        if enable_resume and history_path.is_file():
            loaded = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                history = loaded

        if getattr(system, "name", "") == "lightrag":
            from safe_rag.systems.lightrag.clients import warmup_backends

            warmup_backends()
        defended = DefendedSystem(system, defense)
        original: OriginalGraph = defended.load_original_graph(dataset)
        jobs: list[tuple[str, str, list[Question]]] = []
        for protocol in protocols:
            for index, session in enumerate(plan.sessions.get(protocol, []), start=1):
                session_id = f"{protocol}-{index:04d}"
                if enable_resume and _session_complete(history, protocol, session_id, len(session)):
                    continue
                if enable_resume:
                    history = _drop_incomplete_session(history, protocol, session_id)
                jobs.append((protocol, session_id, session))

        progress = tqdm(
            total=len(jobs),
            desc="Utility",
            unit="session",
            dynamic_ncols=True,
            mininterval=0.5,
            file=sys.stderr,
        )
        history_lock = threading.Lock()
        turn_idx = len(history)

        def run_session(job: tuple[str, str, list[Question]]) -> list[dict[str, Any]]:
            protocol, session_id, session = job
            session_defense = _clone_defense(defense)
            session_defense.initialize(original)
            records = []
            for session_turn, item in enumerate(session, start=1):
                started = time.time()
                result = _query_turn(
                    system,
                    session_defense,
                    item.question,
                    query_method,
                    session_turn,
                )
                records.append(
                    _score_turn(
                        protocol=protocol,
                        session_id=session_id,
                        session_turn=session_turn,
                        session_size=len(session),
                        item=item,
                        result=result,
                        latency_s=time.time() - started,
                    )
                )
            return records

        def commit(records: list[dict[str, Any]]) -> None:
            nonlocal turn_idx
            with history_lock:
                for record in records:
                    turn_idx += 1
                    record["turn"] = turn_idx
                    history.append(record)
                    (turn_log_dir / f"query_{turn_idx:04d}.json").write_text(
                        json.dumps(record, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    scores = record["scores"]
                    progress.set_postfix(
                        protocol=record["protocol"],
                        session=record["session_id"],
                        q=f"{record['session_turn']}/{record['session_size']}",
                        hit=f"{scores['lexical_hit']:.0f}",
                        rouge=f"{scores['rouge_l']:.2f}",
                    )
                progress.update(1)
                _write_json(history_path, history)

        try:
            if session_workers == 1 or len(jobs) <= 1:
                for job in jobs:
                    commit(run_session(job))
            else:
                with ThreadPoolExecutor(max_workers=min(session_workers, len(jobs))) as pool:
                    futures = [pool.submit(run_session, job) for job in jobs]
                    for future in as_completed(futures):
                        commit(future.result())
        finally:
            progress.close()

        metrics = summarize_records(history)
        metrics.update(
            {
                "dataset": dataset,
                "system": getattr(system, "name", "unknown"),
                "defense": getattr(defense, "name", "unknown"),
                "questions_path": str(questions_path),
                "n_selected_questions": len(plan.questions),
                "protocols": list(protocols),
                "session_size": session_size,
                "session_workers": session_workers,
                "seed": seed,
            }
        )
        _write_json(output_dir / "utility_metrics.json", metrics)
        return AttackResult(
            output_dir=output_dir,
            metrics=metrics,
            query_history_path=history_path,
        )
