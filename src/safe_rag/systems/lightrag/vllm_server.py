"""vLLM helpers for the shell orchestrator: chat + embedding."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import httpx

from safe_rag.paths import LIGHTRAG_BUILD_CONFIG
from safe_rag.systems.lightrag.settings import LightRAGSettings, cuda_visible_devices, load_lightrag_settings

ROLES = ("chat", "embed")


def _chat_entrypoint(settings: LightRAGSettings) -> str:
    """Gemma 4 needs the head-dim startup patch; Qwen3.5 uses stock vLLM."""
    extra = " ".join(settings.extra_args).lower()
    name = f"{settings.served_model_name} {settings.model_path}".lower()
    if "gemma4" in extra or "gemma" in name:
        return "safe_rag.systems.lightrag.vllm_compat"
    return "vllm.entrypoints.openai.api_server"


def vllm_command(settings: LightRAGSettings, role: str = "chat", replica: int = 0) -> list[str]:
    if role == "embed":
        if settings.embedding_backend == "cpu":
            raise RuntimeError("Embedding backend is cpu; do not start an embedding vLLM")
        if not settings.embedding_looks_downloaded():
            raise FileNotFoundError(
                f"Embedding weights not found in {settings.embedding_model_path}."
            )
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(settings.embedding_model_path),
            "--served-model-name",
            settings.embedding_model,
            "--runner",
            "pooling",
            "--convert",
            "embed",
            "--gpu-memory-utilization",
            str(settings.embedding_gpu_memory_utilization),
            "--max-model-len",
            str(settings.embedding_max_model_len),
            "--max-num-seqs",
            str(settings.embedding_max_num_seqs),
            "--tensor-parallel-size",
            str(settings.embedding_tensor_parallel_size),
            *settings.embedding_extra_args,
        ]
        if settings.embedding_uds:
            command.extend(["--uds", settings.embedding_uds])
        else:
            host = str(settings.raw.get("embedding", {}).get("host") or "127.0.0.1")
            port = int(settings.raw.get("embedding", {}).get("port") or 8001)
            command.extend(["--host", host, "--port", str(port)])
        return command

    if not settings.model_looks_downloaded():
        raise FileNotFoundError(
            f"LLM weights not found in {settings.model_path}. "
            "Download the model into that folder, then rerun."
        )
    command = [
        sys.executable,
        "-m",
        _chat_entrypoint(settings),
        "--model",
        str(settings.model_path),
        "--served-model-name",
        settings.served_model_name,
        "--gpu-memory-utilization",
        str(settings.gpu_memory_utilization),
        "--max-model-len",
        str(settings.max_model_len),
        "--max-num-seqs",
        str(settings.max_num_seqs),
        "--tensor-parallel-size",
        str(settings.chat_tensor_parallel_size),
        "--dtype",
        settings.dtype,
        *settings.extra_args,
    ]
    uds = settings.chat_replica_uds(replica)
    if uds:
        command.extend(["--uds", uds])
    else:
        command.extend(
            ["--host", settings.vllm_host, "--port", str(settings.chat_replica_port(replica))]
        )
    return command


def _http_client(settings: LightRAGSettings, role: str, replica: int = 0) -> httpx.Client:
    kwargs: dict = {"trust_env": False, "timeout": 5.0}
    if role == "embed":
        uds = settings.embedding_uds
    else:
        uds = settings.chat_replica_uds(replica)
    if uds:
        kwargs["transport"] = httpx.HTTPTransport(uds=uds)
    return httpx.Client(**kwargs)


def _base_url(settings: LightRAGSettings, role: str, replica: int = 0) -> str:
    if role == "embed":
        return settings.embedding_base_url
    return settings.chat_replica_base_url(replica)


def display_url(settings: LightRAGSettings, role: str, replica: int = 0) -> str:
    if role == "embed":
        return settings.embedding_display_url
    return settings.chat_replica_display_url(replica)


def is_ready(settings: LightRAGSettings, role: str = "chat", replica: int | None = None) -> bool:
    if role == "embed" and settings.embedding_backend == "cpu":
        return settings.embedding_looks_downloaded()
    if role == "chat" and replica is None and settings.chat_replicate:
        return all(is_ready(settings, "chat", index) for index in range(settings.chat_replica_count))
    replica = 0 if replica is None else replica
    try:
        with _http_client(settings, role, replica) as client:
            response = client.get(f"{_base_url(settings, role, replica)}/models")
            return 200 <= response.status_code < 300
    except Exception:
        return False


def _fmt_elapsed(seconds: float) -> str:
    total = max(int(seconds), 0)
    if total < 60:
        return f"{total}s"
    return f"{total // 60}m{total % 60:02d}s"


def _write_status(message: str) -> None:
    sys.stdout.write(f"\r{message}\033[K")
    sys.stdout.flush()


def _clear_status_line() -> None:
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def wait_until_ready(
    settings: LightRAGSettings,
    role: str = "chat",
    pid: int | None = None,
    replica: int = 0,
) -> None:
    deadline = time.time() + settings.startup_timeout_s
    started = time.time()
    next_check = 0.0
    label = "embed" if role == "embed" else "chat"
    frames = "|/-\\"
    try:
        while time.time() < deadline:
            elapsed = time.time() - started
            if pid is not None:
                try:
                    os.kill(pid, 0)
                except OSError as exc:
                    raise RuntimeError(
                        f"vLLM {label} process {pid} exited before becoming ready"
                    ) from exc
            if elapsed >= next_check:
                if is_ready(settings, role, replica if role == "chat" else None):
                    return
                next_check = elapsed + 1.0
            frame = frames[int(elapsed * 10) % len(frames)]
            _write_status(
                f"[vLLM {label}] loading weights  {_fmt_elapsed(elapsed)}  {frame}"
            )
            time.sleep(0.1)
        raise TimeoutError(
            f"vLLM {label} did not become ready within {settings.startup_timeout_s:.0f}s"
        )
    finally:
        _clear_status_line()


def prepare_socket(settings: LightRAGSettings, role: str = "chat", replica: int = 0) -> None:
    if role == "embed":
        uds = settings.embedding_uds
    else:
        uds = settings.chat_replica_uds(replica)
    if uds:
        sock = Path(uds)
        if sock.exists():
            sock.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="vLLM helpers for scripts/build_lightrag.sh")
    parser.add_argument(
        "action",
        choices=(
            "argv",
            "devices",
            "ready",
            "wait",
            "prepare-socket",
            "log-dir",
            "log-file",
            "name",
            "replica-count",
            "embedding-backend",
        ),
    )
    parser.add_argument("--role", choices=ROLES, default="chat")
    parser.add_argument("--replica", type=int, default=0)
    parser.add_argument("--config", default=str(LIGHTRAG_BUILD_CONFIG))
    parser.add_argument("--pid", type=int, default=None)
    args = parser.parse_args(argv)
    settings = load_lightrag_settings(args.config)

    if args.action == "replica-count":
        print(settings.chat_replica_count if args.role == "chat" else 1)
        return 0
    if args.action == "embedding-backend":
        print(settings.embedding_backend)
        return 0
    if args.action == "argv":
        for part in vllm_command(settings, args.role, replica=args.replica):
            print(part)
        return 0
    if args.action == "log-dir":
        print(settings.log_dir)
        return 0
    if args.action == "log-file":
        if args.role == "embed":
            print(settings.embed_log_path)
        elif settings.chat_replicate:
            print(settings.log_dir / f"vllm-chat-{args.replica}.log")
        else:
            print(settings.chat_log_path)
        return 0
    if args.action == "name":
        name = settings.embedding_model if args.role == "embed" else settings.served_model_name
        print(name)
        return 0
    if args.action == "devices":
        if args.role == "embed":
            print(cuda_visible_devices(settings.embedding_devices))
        else:
            print(cuda_visible_devices(settings.chat_replica_devices(args.replica)))
        return 0
    if args.action == "ready":
        replica = args.replica if args.role == "chat" else None
        return 0 if is_ready(settings, args.role, replica) else 1
    if args.action == "prepare-socket":
        prepare_socket(settings, args.role, replica=args.replica)
        return 0
    wait_until_ready(settings, args.role, pid=args.pid, replica=args.replica)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[vLLM] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
