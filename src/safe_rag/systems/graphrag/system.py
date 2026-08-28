from pathlib import Path
from typing import Any, Optional

from safe_rag.paths import DATA_ROOT
from safe_rag.systems.base import OriginalGraph, QueryResult


class GraphRAGSystem:
    name = "graphrag"
    graph_backend = "graphrag"

    def __init__(
        self,
        dataset: str = "",
        root_dir: Optional[Path] = None,
        community_level: int = 2,
        response_type: str = "Multiple Paragraphs",
        **kwargs: Any,
    ):
        del kwargs
        self.dataset = dataset
        self.root_dir = Path(root_dir) if root_dir is not None else DATA_ROOT / "graphrag"
        self.community_level = community_level
        self.response_type = response_type

    def dataset_path(self, dataset: Optional[str] = None) -> Path:
        name = dataset or self.dataset
        return self.root_dir / name

    def query(self, text: str, method: str = "local") -> QueryResult:
        import os
        import subprocess
        import tempfile

        dataset_root = self.dataset_path()
        data_dir = dataset_root / "output"
        if not dataset_root.exists():
            raise FileNotFoundError(f"GraphRAG dataset directory not found: {dataset_root}")
        if not data_dir.exists():
            raise FileNotFoundError(f"GraphRAG output directory not found: {data_dir}")

        with tempfile.TemporaryDirectory(prefix="agea_graphrag_") as tmp:
            log_path = os.path.join(tmp, "retrieved_context.json")
            env = os.environ.copy()
            env["GRAPHRAG_LOG_PATH"] = log_path
            command = [
                "python",
                "-m",
                "graphrag",
                "query",
                "--root",
                str(dataset_root.resolve()),
                "--data",
                str(data_dir.resolve()),
                "--community-level",
                str(self.community_level),
                "--response-type",
                self.response_type,
                "--method",
                method,
                "--query",
                text,
            ]
            result = subprocess.run(command, capture_output=True, text=True, env=env)
            retrieved = ""
            if os.path.exists(log_path):
                with open(log_path, encoding="utf-8") as handle:
                    retrieved = handle.read()
            return QueryResult(
                response=result.stdout or "",
                retrieved_context=retrieved,
                stderr=result.stderr or "",
            )

    def load_original_graph(self, dataset: str) -> OriginalGraph:
        from safe_rag.attacks.agea.utils import load_original_graph_data as load_graph

        filtered, original_nodes, original_edges, stats = load_graph(
            dataset,
            filter_isolated=False,
            dataset_base_path=str(self.root_dir),
            graph_backend="graphrag",
        )
        return OriginalGraph(
            filtered_nodes=filtered,
            original_nodes=original_nodes,
            original_edges=original_edges,
            stats=stats,
        )
