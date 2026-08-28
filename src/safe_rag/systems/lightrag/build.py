import os
import asyncio
import json
import argparse
import shutil
from datetime import datetime
from pathlib import Path
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from tqdm import tqdm

from safe_rag.logutil import redirect_library_logs
from safe_rag.paths import DATA_ROOT, LIGHTRAG_BUILD_CONFIG
from safe_rag.systems.lightrag.chunking import chunk_documents
from safe_rag.systems.lightrag.clients import (
    chat_model,
    detect_embedding_dimension,
    embedding_func,
    embedding_model,
    llm_model_func,
    load_lightrag_config,
)
from safe_rag.systems.lightrag.settings import get_lightrag_settings
from safe_rag.systems.lightrag.vllm_server import is_ready

DATASET_INPUT_ROOT = DATA_ROOT / "corpus"
GRAPH_OUTPUT_ROOT = DATA_ROOT / "lightrag"
_build_log_file = None


def setup_build_log(path: Path) -> None:
    global _build_log_file
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LOG_DIR", str(path.parent))
    _build_log_file = path.open("w", encoding="utf-8")
    _build_log_file.write(f"===== {datetime.now().isoformat(timespec='seconds')} =====\n")
    _build_log_file.flush()
    # LightRAG attaches its own console handler (propagate=False). Keep INFO in the
    # file; leave the terminal for tqdm and our log() lines.
    redirect_library_logs(path, mode="a")


def log(message: str) -> None:
    try:
        tqdm.write(message)
    except Exception:
        print(message, flush=True)
    if _build_log_file is not None:
        _build_log_file.write(message + "\n")
        _build_log_file.flush()


def corpus_path_for(dataset_name: str) -> Path:
    """Raw corpus lives at data/corpus/<dataset>.json."""
    return DATASET_INPUT_ROOT / f"{dataset_name}.json"


def load_corpus_data(dataset_name):
    """
    Load corpus data from the specified dataset.
    Handles multiple formats:
    - Medical JSON: [{"corpus_name": "Medical", "context": "...", ...}] or a single object
    - Novel format: [{"idx": ..., "title": "...", "text": "...", ...}]
    - Agriculture format: {"context": "..."} separated by newlines
    - Plain text format: raw text content
    """
    corpus_path = corpus_path_for(dataset_name)

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found for dataset: {dataset_name} at {corpus_path}")

    log(f"Loading corpus from: {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    contexts = []

    try:
        data = json.loads(content)
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if "context" in item:
                        contexts.append(item["context"])
                    elif "text" in item:
                        contexts.append(item["text"])
                    else:
                        text_fields = [v for k, v in item.items() if isinstance(v, str) and len(v) > 100]
                        if text_fields:
                            contexts.append(max(text_fields, key=len))
            if contexts:
                log(f"Loaded {len(contexts)} contexts from JSON array format")
                return contexts
    except json.JSONDecodeError:
        pass

    try:
        parts = content.split("}{")
        for i, part in enumerate(parts):
            if i == 0:
                part = part + "}"
            elif i == len(parts) - 1:
                part = "{" + part
            else:
                part = "{" + part + "}"
            try:
                json_obj = json.loads(part)
                if isinstance(json_obj, dict):
                    if "context" in json_obj:
                        contexts.append(json_obj["context"])
                    elif "text" in json_obj:
                        contexts.append(json_obj["text"])
            except json.JSONDecodeError:
                continue
        if contexts:
            log(f"Loaded {len(contexts)} contexts from separate JSON objects format")
            return contexts
    except Exception as exc:
        log(f"Error parsing separate JSON objects: {exc}")

    if content:
        contexts.append(content)
        log("Loaded 1 context from raw text format")
        return contexts

    raise ValueError(f"Could not parse corpus data from {corpus_path}")


async def test_funcs():
    log(f"LLM: {get_lightrag_settings().chat_display_url}  model={chat_model()}")
    log(f"Embedding: {get_lightrag_settings().embedding_display_url}  model={embedding_model()}")
    result = await llm_model_func("How are you?")
    log("LLM Response: " + (result or ""))

    result = await embedding_func(["How are you?"])
    log(f"Embedding Result: {result.shape}")
    log(f"Embedding Dimension: {result.shape[1]}")


async def initialize_rag(working_dir, settings):
    embedding_dim = await detect_embedding_dimension()
    log(f"Detected embedding dimension: {embedding_dim} (from model: {embedding_model()})")

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=embedding_func,
        ),
        chunk_token_size=settings.chunk_token_size,
        chunk_overlap_token_size=settings.chunk_overlap_token_size,
        llm_model_max_async=settings.llm_max_async,
        max_parallel_insert=settings.llm_max_async,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    parser = argparse.ArgumentParser(
        description="Insert corpus into a LightRAG workspace. Start vLLM with scripts/build_lightrag.sh."
    )
    parser.add_argument(
        "--config",
        default=str(LIGHTRAG_BUILD_CONFIG),
        help="Path to configs/lightrag/build.yaml",
    )
    parser.add_argument("--dataset", default=None, help="Override dataset name from the YAML config")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the existing workspace and build from scratch",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep the workspace and retry failed documents",
    )
    parser.add_argument("--test-llm", action="store_true", help="Ping LLM and embedding before indexing")
    args = parser.parse_args()
    if args.rebuild and args.resume:
        raise SystemExit("use only one of --rebuild or --resume")

    settings = load_lightrag_config(args.config)
    setup_build_log(settings.build_log_path)
    dataset = args.dataset or settings.dataset

    log(f"Config: {settings.path}")
    log(f"Logs: {settings.log_dir}")
    log(f"Dataset: {dataset}")
    log(f"LLM: {settings.chat_display_url}  model={settings.served_model_name}")
    log(f"Embedding: {settings.embedding_display_url}  model={settings.embedding_model}")

    if not is_ready(settings, "chat"):
        raise RuntimeError(
            f"Chat vLLM is not reachable at {settings.chat_display_url}. "
            "Start it with: bash scripts/build_lightrag.sh"
        )
    if not is_ready(settings, "embed"):
        raise RuntimeError(
            f"Embedding vLLM is not reachable at {settings.embedding_display_url}. "
            "Start it with: bash scripts/build_lightrag.sh"
        )
    log("vLLM chat and embedding are ready")

    if args.test_llm:
        log("Testing LLM and embedding endpoints...")
        asyncio.run(test_funcs())
        log("Endpoint test completed.")

    contexts = load_corpus_data(dataset)
    chunks = chunk_documents(
        contexts,
        chunk_token_size=settings.chunk_token_size,
        overlap_token_size=settings.chunk_overlap_token_size,
        chars_per_token=settings.chars_per_token,
    )
    log(
        f"Loaded {len(contexts)} document(s) → {len(chunks)} chunk(s) "
        f"(~{settings.chunk_token_size} tokens, overlap {settings.chunk_overlap_token_size})"
    )
    if not chunks:
        raise ValueError("No text chunks produced from the corpus")

    working_dir = GRAPH_OUTPUT_ROOT / dataset
    GRAPH_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    workspace_exists = working_dir.exists()
    rebuild = args.rebuild or (not workspace_exists and not args.resume)
    if rebuild and workspace_exists:
        log(f"Removing existing workspace: {working_dir}")
        shutil.rmtree(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    if rebuild:
        log(f"Workspace: {working_dir} (new)")
    else:
        log(f"Workspace: {working_dir} (resume; processed docs are skipped)")

    log("Initializing LightRAG...")
    rag = asyncio.run(initialize_rag(working_dir, settings))

    batch_size = max(settings.insert_batch_size, settings.llm_max_async)
    log(
        f"Building knowledge graph ({len(chunks)} chunks, "
        f"max_async={settings.llm_max_async}, batch={batch_size})..."
    )
    with tqdm(
        total=len(chunks),
        desc="Building knowledge graph",
        unit="chunk",
        dynamic_ncols=True,
        mininterval=0.5,
    ) as bar:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            rag.insert(batch)
            bar.update(len(batch))

    graphml = working_dir / "graph_chunk_entity_relation.graphml"
    log("Knowledge graph construction completed.")
    log(f"Workspace: {working_dir}")
    if graphml.exists():
        log(f"Graph: {graphml}")


if __name__ == "__main__":
    main()
