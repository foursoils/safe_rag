from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from openai import OpenAI

from safe_rag.paths import AGEA_LLM_CONFIG, REPO_ROOT

_PLACEHOLDERS = {
    "",
    "sk-your_key",
    "sk-xxx",
    "sk-replace",
    "your_key",
    "https://your_console_api_host/v1",
    "https://your_console_api_host",
}


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _normalize_base_url(url: str) -> str:
    text = (url or "").strip().rstrip("/")
    if text and not text.endswith("/v1"):
        text = f"{text}/v1"
    return text


def _is_placeholder(value: str) -> bool:
    return (value or "").strip().lower() in _PLACEHOLDERS


@dataclass
class AgeaLLMSettings:
    base_url: str
    api_key: str
    query_generator_model: str
    graph_filter_model: str
    max_retries: int
    timeout: float
    path: Path


_SETTINGS: Optional[AgeaLLMSettings] = None


def load_agea_llm_settings(path: str | Path | None = None) -> AgeaLLMSettings:
    global _SETTINGS
    config_path = _resolve(path) if path is not None else AGEA_LLM_CONFIG
    if not config_path.exists():
        example = config_path.with_name("llm.yaml.example")
        raise FileNotFoundError(
            f"AGEA LLM config not found: {config_path}. "
            f"Copy {example if example.exists() else 'configs/agea/llm.yaml.example'} "
            "to configs/agea/llm.yaml and fill in base_url and api_key."
        )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base_url = _normalize_base_url(str(raw.get("base_url") or ""))
    api_key = str(raw.get("api_key") or "").strip()
    if _is_placeholder(base_url) or not base_url:
        raise ValueError(
            f"Set base_url in {config_path} to the console API address "
            "(OpenAI-compatible, should end with /v1)."
        )
    if _is_placeholder(api_key) or not api_key:
        raise ValueError(f"Set api_key in {config_path}.")
    _SETTINGS = AgeaLLMSettings(
        base_url=base_url,
        api_key=api_key,
        query_generator_model=str(raw.get("query_generator_model") or "gpt-4o-mini"),
        graph_filter_model=str(raw.get("graph_filter_model") or "gpt-4o-mini"),
        max_retries=int(raw.get("max_retries") or 10),
        timeout=float(raw.get("timeout") or 120),
        path=config_path,
    )
    get_openai_client.cache_clear()
    return _SETTINGS


def get_agea_llm_settings() -> AgeaLLMSettings:
    if _SETTINGS is None:
        return load_agea_llm_settings()
    return _SETTINGS


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    settings = get_agea_llm_settings()
    return OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        max_retries=settings.max_retries,
        timeout=settings.timeout,
    )


def graph_filter_deployment(default: str = "gpt-4o-mini") -> str:
    return default or get_agea_llm_settings().graph_filter_model


def query_generator_deployment(default: str = "gpt-4o-mini") -> str:
    return default or get_agea_llm_settings().query_generator_model
