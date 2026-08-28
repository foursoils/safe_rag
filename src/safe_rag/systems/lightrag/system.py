from pathlib import Path
from typing import Any, Optional
from dataclasses import replace

from safe_rag.paths import DATA_ROOT
from safe_rag.systems.base import OriginalGraph, QueryResult


class LightRAGSystem:
    name = "lightrag"
    graph_backend = "lightrag"

    def __init__(
        self,
        dataset: str = "",
        root_dir: Optional[Path] = None,
        **kwargs: Any,
    ):
        del kwargs
        self.dataset = dataset
        self.root_dir = Path(root_dir) if root_dir is not None else DATA_ROOT / "lightrag"

    def working_dir(self, dataset: Optional[str] = None) -> Path:
        name = dataset or self.dataset
        return self.root_dir / name

    def query(self, text: str, method: str = "hybrid") -> QueryResult:
        import asyncio

        from lightrag import LightRAG, QueryParam
        from lightrag.kg.shared_storage import initialize_pipeline_status
        from lightrag.utils import EmbeddingFunc

        from safe_rag.systems.lightrag.clients import (
            embedding_dim,
            embedding_func,
            llm_model_func,
            load_lightrag_config,
        )

        load_lightrag_config()

        working_dir = self.working_dir()
        if not working_dir.exists():
            raise FileNotFoundError(f"LightRAG workspace not found: {working_dir}")

        async def run_query() -> tuple[str, str, str]:
            rag = LightRAG(
                working_dir=str(working_dir),
                llm_model_func=llm_model_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=embedding_dim(),
                    max_token_size=8192,
                    func=embedding_func,
                ),
            )
            await rag.initialize_storages()
            await initialize_pipeline_status()
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

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            response, error, context = asyncio.run(run_query())
        else:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                response, error, context = pool.submit(asyncio.run, run_query()).result()

        return QueryResult(response=response, retrieved_context=context, stderr=error)

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
