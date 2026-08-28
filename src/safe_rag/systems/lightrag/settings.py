"""Load LightRAG build/runtime settings from configs/lightrag/build.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from safe_rag.paths import LIGHTRAG_BUILD_CONFIG, REPO_ROOT


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def parse_cuda_devices(value: Any, default: list[int] | None = None) -> list[int]:
    """Accept [0, 1], '0,1', '0', or 0."""
    if value is None:
        return list(default or [0])
    if isinstance(value, bool):
        return list(default or [0])
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return list(default or [0])
        return [int(part.strip()) for part in text.split(",") if part.strip() != ""]
    if isinstance(value, (list, tuple)):
        return [int(part) for part in value]
    raise ValueError(f"Invalid CUDA devices value: {value!r}")


def cuda_visible_devices(devices: list[int]) -> str:
    return ",".join(str(index) for index in devices)


@dataclass
class LightRAGSettings:
    raw: dict[str, Any]
    path: Path

    @property
    def dataset(self) -> str:
        return str(self.raw.get("dataset") or "medical")

    @property
    def log_dir(self) -> Path:
        return _resolve(self.raw.get("log_dir") or "logs/lightrag")

    @property
    def chat_log_path(self) -> Path:
        return self.log_dir / "vllm-chat.log"

    @property
    def embed_log_path(self) -> Path:
        return self.log_dir / "vllm-embed.log"

    @property
    def build_log_path(self) -> Path:
        return self.log_dir / "build.log"

    @property
    def model_path(self) -> Path:
        return _resolve(self.raw.get("llm", {}).get("model_path") or "models/llm/gemma-4-31B-it")

    @property
    def served_model_name(self) -> str:
        return str(self.raw.get("llm", {}).get("served_model_name") or "gemma-4-31B-it")

    @property
    def llm_api_key(self) -> str:
        return str(self.raw.get("llm", {}).get("api_key") or "EMPTY")

    @property
    def vllm_host(self) -> str:
        return str(self.raw.get("vllm", {}).get("host") or "127.0.0.1")

    @property
    def vllm_port(self) -> int:
        return int(self.raw.get("vllm", {}).get("port") or 8000)

    @property
    def vllm_uds(self) -> Optional[str]:
        """Unix socket path. Empty/false disables it and falls back to TCP.

        Default is on: WSL + Clash TUN commonly black-holes TCP to 127.0.0.1
        after the port starts listening.
        """
        block = self.raw.get("vllm", {}) or {}
        if "uds" not in block:
            return "/tmp/safe-rag-vllm.sock"
        value = block.get("uds")
        if value in (None, False, ""):
            return None
        return str(value)

    @property
    def chat_base_url(self) -> str:
        if self.vllm_uds:
            return "http://localhost/v1"
        probe_host = "127.0.0.1" if self.vllm_host in {"0.0.0.0", "::"} else self.vllm_host
        return f"http://{probe_host}:{self.vllm_port}/v1"

    @property
    def chat_display_url(self) -> str:
        if self.vllm_uds:
            return f"unix:{self.vllm_uds}"
        return f"http://{self.vllm_host}:{self.vllm_port}/v1"

    @property
    def gpu_memory_utilization(self) -> float:
        return float(self.raw.get("vllm", {}).get("gpu_memory_utilization") or 0.85)

    @property
    def chat_devices(self) -> list[int]:
        return parse_cuda_devices(self.raw.get("vllm", {}).get("devices"), default=[0])

    @property
    def chat_tensor_parallel_size(self) -> int:
        return max(len(self.chat_devices), 1)

    @property
    def max_model_len(self) -> int:
        return int(self.raw.get("vllm", {}).get("max_model_len") or 10240)

    @property
    def max_num_seqs(self) -> int:
        return int(self.raw.get("vllm", {}).get("max_num_seqs") or 64)

    @property
    def dtype(self) -> str:
        return str(self.raw.get("vllm", {}).get("dtype") or "auto")

    @property
    def extra_args(self) -> list[str]:
        value = self.raw.get("vllm", {}).get("extra_args") or []
        return [str(item) for item in value]

    @property
    def startup_timeout_s(self) -> float:
        return float(self.raw.get("vllm", {}).get("startup_timeout_s") or 600)

    @property
    def stop_after_build(self) -> bool:
        return bool(self.raw.get("vllm", {}).get("stop_after_build", True))

    @property
    def embedding_base_url(self) -> str:
        if self.embedding_uds:
            return "http://localhost/v1"
        explicit = str(self.raw.get("embedding", {}).get("base_url") or "").rstrip("/")
        if explicit:
            return explicit
        host = str(self.raw.get("embedding", {}).get("host") or "127.0.0.1")
        port = int(self.raw.get("embedding", {}).get("port") or 8001)
        probe = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        return f"http://{probe}:{port}/v1"

    @property
    def embedding_display_url(self) -> str:
        if self.embedding_uds:
            return f"unix:{self.embedding_uds}"
        return self.embedding_base_url

    @property
    def embedding_model_path(self) -> Path:
        return _resolve(
            self.raw.get("embedding", {}).get("model_path") or "models/embedding/Qwen3-Embedding-4B"
        )

    @property
    def embedding_uds(self) -> Optional[str]:
        block = self.raw.get("embedding", {}) or {}
        if "uds" not in block:
            return "/tmp/safe-rag-embed.sock"
        value = block.get("uds")
        if value in (None, False, ""):
            return None
        return str(value)

    @property
    def embedding_gpu_memory_utilization(self) -> float:
        return float(self.raw.get("embedding", {}).get("gpu_memory_utilization") or 0.2)

    @property
    def embedding_devices(self) -> list[int]:
        return parse_cuda_devices(self.raw.get("embedding", {}).get("devices"), default=[0])

    @property
    def embedding_tensor_parallel_size(self) -> int:
        return max(len(self.embedding_devices), 1)

    @property
    def embedding_max_model_len(self) -> int:
        return int(self.raw.get("embedding", {}).get("max_model_len") or 8192)

    @property
    def embedding_max_num_seqs(self) -> int:
        return int(self.raw.get("embedding", {}).get("max_num_seqs") or 32)

    @property
    def embedding_extra_args(self) -> list[str]:
        value = self.raw.get("embedding", {}).get("extra_args") or []
        return [str(item) for item in value]

    @property
    def embedding_model(self) -> str:
        return str(self.raw.get("embedding", {}).get("model") or "Qwen3-Embedding-4B")

    @property
    def embedding_dim(self) -> int:
        return int(self.raw.get("embedding", {}).get("dim") or 2560)

    @property
    def embedding_api_key(self) -> str:
        return str(self.raw.get("embedding", {}).get("api_key") or "EMPTY")

    @property
    def chunk_token_size(self) -> int:
        return int(self.raw.get("chunking", {}).get("chunk_token_size") or 1200)

    @property
    def chunk_overlap_token_size(self) -> int:
        return int(self.raw.get("chunking", {}).get("chunk_overlap_token_size") or 100)

    @property
    def chars_per_token(self) -> int:
        return int(self.raw.get("chunking", {}).get("chars_per_token") or 4)

    @property
    def temperature(self) -> float:
        return float(self.raw.get("llm", {}).get("temperature", 1.0))

    @property
    def top_p(self) -> float:
        return float(self.raw.get("llm", {}).get("top_p", 0.95))

    @property
    def top_k(self) -> int:
        return int(self.raw.get("llm", {}).get("top_k", 64))

    @property
    def min_p(self) -> float:
        return float(self.raw.get("llm", {}).get("min_p", 0.0))

    @property
    def presence_penalty(self) -> float:
        return float(self.raw.get("llm", {}).get("presence_penalty", 0.0))

    @property
    def repetition_penalty(self) -> float:
        return float(self.raw.get("llm", {}).get("repetition_penalty", 1.0))

    @property
    def max_tokens(self) -> Optional[int]:
        """Optional output cap. None = use whatever remains after the prompt."""
        value = self.raw.get("llm", {}).get("max_tokens")
        if value in (None, "", False):
            return None
        return int(value)

    @property
    def enable_thinking(self) -> bool:
        return bool(self.raw.get("llm", {}).get("enable_thinking", True))

    @property
    def llm_max_async(self) -> int:
        return int(self.raw.get("insert", {}).get("max_async") or 8)

    @property
    def insert_batch_size(self) -> int:
        return int(self.raw.get("insert", {}).get("batch_size") or 16)

    def model_looks_downloaded(self) -> bool:
        return self._weights_present(self.model_path)

    def embedding_looks_downloaded(self) -> bool:
        return self._weights_present(self.embedding_model_path)

    @staticmethod
    def _weights_present(path: Path) -> bool:
        if not path.exists():
            return False
        markers = ("config.json", "tokenizer.json", "tokenizer_config.json")
        if any((path / name).exists() for name in markers):
            return True
        return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))


_SETTINGS: Optional[LightRAGSettings] = None


def load_lightrag_settings(path: str | Path | None = None) -> LightRAGSettings:
    global _SETTINGS
    config_path = Path(path) if path is not None else LIGHTRAG_BUILD_CONFIG
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    _SETTINGS = LightRAGSettings(raw=raw, path=config_path)
    return _SETTINGS


def get_lightrag_settings() -> LightRAGSettings:
    if _SETTINGS is None:
        return load_lightrag_settings()
    return _SETTINGS
