import asyncio
import threading
from pathlib import Path
from typing import Any, Optional
from dataclasses import replace

from safe_rag.paths import DATA_ROOT
from safe_rag.systems.base import OriginalGraph, QueryResult, StructuredRetrieval

_LOOP: asyncio.AbstractEventLoop | None = None
_LOOP_THREAD: threading.Thread | None = None
_LOOP_GUARD = threading.Lock()


def _lightrag_loop() -> asyncio.AbstractEventLoop:
    """One process-wide loop. LightRAG's asyncio.Lock objects bind to it."""
    global _LOOP, _LOOP_THREAD
    with _LOOP_GUARD:
        if _LOOP is not None and _LOOP.is_running():
            return _LOOP
        ready = threading.Event()

        def _run() -> None:
            global _LOOP
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _LOOP = loop
            ready.set()
            loop.run_forever()

        _LOOP_THREAD = threading.Thread(target=_run, name="lightrag-loop", daemon=True)
        _LOOP_THREAD.start()
        if not ready.wait(timeout=10):
            raise RuntimeError("LightRAG event loop failed to start")
        if _LOOP is None:
            raise RuntimeError("LightRAG event loop was not created")
        return _LOOP


def run_on_lightrag_loop(coro):
    loop = _lightrag_loop()
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        raise RuntimeError("cannot block the LightRAG event loop from inside itself")
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


class LightRAGSystem:
    name = "lightrag"
    graph_backend = "lightrag"

    def __init__(
        self,
        dataset: str = "",
        root_dir: Optional[Path] = None,
        response_type: str = "Multiple Paragraphs",
        **kwargs: Any,
    ):
        del kwargs
        self.dataset = dataset
        self.root_dir = Path(root_dir) if root_dir is not None else DATA_ROOT / "lightrag"
        self.response_type = response_type
        from safe_rag.systems.lightrag.settings import get_lightrag_settings

        replicas = max(1, get_lightrag_settings().embedding_replicas)
        self._retrieve_slots = threading.Semaphore(replicas)
        self._rag = None
        self._rag_lock: asyncio.Lock | None = None

    def working_dir(self, dataset: Optional[str] = None) -> Path:
        name = dataset or self.dataset
        return self.root_dir / name

    @staticmethod
    def _run_async(coro):
        return run_on_lightrag_loop(coro)

    def _new_rag(self):
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc

        from safe_rag.systems.lightrag.clients import (
            embedding_dim,
            embedding_func,
            llm_model_func,
        )

        working_dir = self.working_dir()
        if not working_dir.exists():
            raise FileNotFoundError(f"LightRAG workspace not found: {working_dir}")
        return LightRAG(
            working_dir=str(working_dir),
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dim(),
                max_token_size=8192,
                func=embedding_func,
            ),
        )

    async def _rag_instance(self):
        from lightrag.kg.shared_storage import initialize_pipeline_status

        if self._rag_lock is None:
            self._rag_lock = asyncio.Lock()
        async with self._rag_lock:
            if self._rag is None:
                rag = self._new_rag()
                await rag.initialize_storages()
                await initialize_pipeline_status()
                self._rag = rag
            return self._rag

    def query(self, text: str, method: str = "hybrid") -> QueryResult:
        from lightrag import QueryParam

        from safe_rag.systems.lightrag.settings import get_lightrag_settings

        get_lightrag_settings()

        async def run_query() -> tuple[str, str, str]:
            rag = await self._rag_instance()
            query_param = QueryParam(
                mode=method,
                top_k=10,
                chunk_top_k=5,
                max_entity_tokens=2000,
                max_relation_tokens=2000,
                max_total_tokens=8192,
                enable_rerank=False,
            )
            try:
                context = await rag.aquery(
                    text,
                    param=replace(query_param, only_need_context=True),
                )
                response = await rag.aquery(
                    text,
                    param=replace(query_param, stream=False),
                )
                answer = (response or "").strip()
                error = ""
                if not answer:
                    error = (
                        "LightRAG returned an empty answer. Check logs/*/lightrag.log "
                        "for vLLM context-length errors."
                    )
                return answer, error, (context or "")
            except Exception as exc:
                return "", str(exc), ""

        response, error, context = self._run_async(run_query())
        return QueryResult(response=response, retrieved_context=context, stderr=error)

    def retrieve(self, text: str, method: str = "hybrid") -> StructuredRetrieval:
        from lightrag import QueryParam

        from safe_rag.systems.lightrag.context import parse_lightrag_result
        from safe_rag.systems.lightrag.settings import get_lightrag_settings

        get_lightrag_settings()

        async def run_retrieve():
            rag = await self._rag_instance()
            aquery_llm = getattr(rag, "aquery_llm", None)
            if not callable(aquery_llm):
                raise RuntimeError(
                    "Installed LightRAG does not expose structured aquery_llm retrieval"
                )
            result = await aquery_llm(
                text,
                param=QueryParam(
                    mode=method,
                    only_need_context=True,
                    top_k=10,
                    chunk_top_k=5,
                    max_entity_tokens=2000,
                    max_relation_tokens=2000,
                    max_total_tokens=8192,
                    enable_rerank=False,
                ),
            )
            return parse_lightrag_result(result)

        with self._retrieve_slots:
            return self._run_async(run_retrieve())

    def generate(
        self,
        text: str,
        retrieval: StructuredRetrieval,
        method: str = "hybrid",
    ) -> QueryResult:
        del method
        from safe_rag.systems.lightrag.context import render_lightrag_context

        context = render_lightrag_context(retrieval)
        if not context:
            return QueryResult(
                response="No safe context is available for this query.",
                retrieved_context="",
            )

        from lightrag.prompt import PROMPTS

        from safe_rag.systems.lightrag.clients import llm_model_func

        system_prompt = PROMPTS["rag_response"].format(
            response_type=self.response_type,
            user_prompt="",
            context_data=context,
        )

        async def run_generate() -> str:
            return await llm_model_func(text, system_prompt=system_prompt)

        response = (self._run_async(run_generate()) or "").strip()
        error = ""
        if not response:
            error = (
                "LightRAG generation returned an empty answer. Check logs/*/lightrag.log "
                "for vLLM context-length errors."
            )
        return QueryResult(
            response=response,
            retrieved_context=context,
            stderr=error,
        )

    def load_original_graph(self, dataset: str) -> OriginalGraph:
        from safe_rag.attacks.agea.utils import load_original_graph_data as load_graph

        filtered, original_nodes, original_edges, stats = load_graph(
            dataset,
            filter_isolated=False,
            dataset_base_path=str(self.root_dir),
            graph_backend="lightrag",
        )
        return OriginalGraph(
            filtered_nodes=filtered,
            original_nodes=original_nodes,
            original_edges=original_edges,
            stats=stats,
        )
