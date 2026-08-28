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


def vllm_command(settings: LightRAGSettings, role: str = "chat") -> list[str]:
    if role == "embed":
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
        "safe_rag.systems.lightrag.vllm_compat",
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
    if settings.vllm_uds:
        command.extend(["--uds", settings.vllm_uds])
    else:
        command.extend(["--host", settings.vllm_host, "--port", str(settings.vllm_port)])
    return command


def _http_client(settings: LightRAGSettings, role: str) -> httpx.Client:
    kwargs: dict = {"trust_env": False, "timeout": 5.0}
    uds = settings.embedding_uds if role == "embed" else settings.vllm_uds
    if uds:
        kwargs["transport"] = httpx.HTTPTransport(uds=uds)
    return httpx.Client(**kwargs)


def _base_url(settings: LightRAGSettings, role: str) -> str:
    if role == "embed":
        return settings.embedding_base_url
    return settings.chat_base_url


def display_url(settings: LightRAGSettings, role: str) -> str:
    if role == "embed":
        return settings.embedding_display_url
    return settings.chat_display_url


def is_ready(settings: LightRAGSettings, role: str = "chat") -> bool:
    try:
        with _http_client(settings, role) as client:
            response = client.get(f"{_base_url(settings, role)}/models")
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
                if is_ready(settings, role):
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


def prepare_socket(settings: LightRAGSettings, role: str = "chat") -> None:
    uds = settings.embedding_uds if role == "embed" else settings.vllm_uds
    if uds:
        sock = Path(uds)
        if sock.exists():
            sock.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="vLLM helpers for scripts/build_lightrag.sh")
    parser.add_argument(
        "action",
        choices=("argv", "devices", "ready", "wait", "prepare-socket", "log-dir", "log-file", "name"),
    )
    parser.add_argument("--role", choices=ROLES, default="chat")
    parser.add_argument("--config", default=str(LIGHTRAG_BUILD_CONFIG))
    parser.add_argument("--pid", type=int, default=None)
    args = parser.parse_args(argv)
    settings = load_lightrag_settings(args.config)

    if args.action == "argv":
        for part in vllm_command(settings, args.role):
            print(part)
        return 0
    if args.action == "log-dir":
        print(settings.log_dir)
        return 0
    if args.action == "log-file":
        path = settings.embed_log_path if args.role == "embed" else settings.chat_log_path
        print(path)
        return 0
    if args.action == "name":
        name = settings.embedding_model if args.role == "embed" else settings.served_model_name
        print(name)
        return 0
    if args.action == "devices":
        devices = settings.embedding_devices if args.role == "embed" else settings.chat_devices
        print(cuda_visible_devices(devices))
        return 0
    if args.action == "ready":
        return 0 if is_ready(settings, args.role) else 1
    if args.action == "prepare-socket":
        prepare_socket(settings, args.role)
        return 0
    wait_until_ready(settings, args.role, pid=args.pid)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[vLLM] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
