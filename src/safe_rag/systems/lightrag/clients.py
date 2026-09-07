"""OpenAI-compatible LightRAG backends: local vLLM chat + embeddings."""

from __future__ import annotations

import itertools
import os
import queue
import threading
from functools import lru_cache
import re

import httpx
import numpy as np
from openai import AsyncOpenAI, OpenAI

from safe_rag.systems.lightrag.settings import get_lightrag_settings, load_lightrag_settings

_chat_rr = itertools.count()
_cpu_pool_lock = threading.Lock()
_cpu_embed_pool: queue.Queue | None = None


def reset_http_clients() -> None:
    _chat_clients.cache_clear()
    embedding_client.cache_clear()


def reset_clients() -> None:
    global _cpu_embed_pool
    reset_http_clients()
    with _cpu_pool_lock:
        _cpu_embed_pool = None


def load_lightrag_config(path=None):
    settings = load_lightrag_settings(path)
    reset_http_clients()
    return settings


def warmup_backends() -> None:
    """Load settings and the CPU embedding replicas before the first query."""
    settings = get_lightrag_settings()
    if settings.embedding_backend == "cpu":
        _ensure_cpu_embed_pool()


def chat_base_url() -> str:
    return get_lightrag_settings().chat_base_url


def embedding_base_url() -> str:
    return get_lightrag_settings().embedding_base_url


def chat_model() -> str:
    return get_lightrag_settings().served_model_name


def embedding_model() -> str:
    return get_lightrag_settings().embedding_model


def embedding_dim() -> int:
    return get_lightrag_settings().embedding_dim


def _chat_http_client(replica: int = 0) -> httpx.AsyncClient:
    settings = get_lightrag_settings()
    kwargs: dict = {"trust_env": False, "timeout": None}
    uds = settings.chat_replica_uds(replica)
    if uds:
        kwargs["transport"] = httpx.AsyncHTTPTransport(uds=uds)
    return httpx.AsyncClient(**kwargs)


@lru_cache(maxsize=1)
def _chat_clients() -> tuple[AsyncOpenAI, ...]:
    settings = get_lightrag_settings()
    return tuple(
        AsyncOpenAI(
            base_url=settings.chat_replica_base_url(replica),
            api_key=settings.llm_api_key,
            timeout=None,
            max_retries=0,
            http_client=_chat_http_client(replica),
        )
        for replica in range(settings.chat_replica_count)
    )


def chat_client() -> AsyncOpenAI:
    clients = _chat_clients()
    if len(clients) == 1:
        return clients[0]
    return clients[next(_chat_rr) % len(clients)]


@lru_cache(maxsize=1)
def embedding_client() -> OpenAI:
    settings = get_lightrag_settings()
    kwargs: dict = {"trust_env": False, "timeout": 120.0}
    if settings.embedding_uds:
        kwargs["transport"] = httpx.HTTPTransport(uds=settings.embedding_uds)
    http_client = httpx.Client(**kwargs)
    return OpenAI(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        timeout=120.0,
        max_retries=5,
        http_client=http_client,
    )


def _prompt_char_len(messages: list[dict]) -> int:
    prompt_chars = 0
    for message in messages:
        content = message.get("content") or ""
        if isinstance(content, str):
            prompt_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    prompt_chars += len(str(part.get("text") or ""))
    return prompt_chars


def _output_token_budget(messages: list[dict]) -> int:
    """Leave remaining context for generation. Never force a floor that overflows max_model_len."""
    settings = get_lightrag_settings()
    # English/JSON is ~4 chars/token; pad for the chat template.
    prompt_tokens = _prompt_char_len(messages) // 4 + 256
    room = settings.max_model_len - prompt_tokens - 32
    cap = settings.max_tokens
    if cap is not None:
        room = min(room, cap)
    return max(1, min(room, settings.max_model_len - 1))


def _message_text(message) -> str:
    content = message.content or ""
    if content.strip():
        return content
    reasoning = getattr(message, "reasoning_content", None) or ""
    return reasoning or ""


def _context_overflow_input_tokens(exc: BaseException) -> int | None:
    match = re.search(r"prompt contains at least (\d+) input tokens", str(exc))
    if not match:
        return None
    return int(match.group(1))


async def llm_model_func(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
    del keyword_extraction
    settings = get_lightrag_settings()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    max_tokens = _output_token_budget(messages)
    extra_body = {
        "top_k": settings.top_k,
        "min_p": settings.min_p,
        "repetition_penalty": settings.repetition_penalty,
        "chat_template_kwargs": {"enable_thinking": settings.enable_thinking},
    }

    async def _complete(output_tokens: int):
        return await chat_client().chat.completions.create(
            model=chat_model(),
            messages=messages,
            temperature=settings.temperature,
            top_p=settings.top_p,
            presence_penalty=settings.presence_penalty,
            max_tokens=output_tokens,
            n=kwargs.get("n", 1),
            extra_body=extra_body,
        )

    try:
        chat_completion = await _complete(max_tokens)
    except Exception as exc:
        input_tokens = _context_overflow_input_tokens(exc)
        if input_tokens is None:
            raise
        retry_tokens = max(1, settings.max_model_len - input_tokens - 8)
        if settings.max_tokens is not None:
            retry_tokens = min(retry_tokens, settings.max_tokens)
        chat_completion = await _complete(retry_tokens)
    return _message_text(chat_completion.choices[0].message)


def _last_token_pool(last_hidden_states, attention_mask):
    import torch

    if attention_mask[:, -1].sum() == attention_mask.shape[0]:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch, device=last_hidden_states.device), sequence_lengths]


def _cpu_thread_budget(replicas: int) -> int:
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    return max(1, cores // max(replicas, 1))


def _load_one_cpu_model(path: str):
    import torch
    from transformers import AutoModel

    try:
        from transformers.utils.logging import disable_progress_bar

        disable_progress_bar()
    except Exception:
        pass
    model = AutoModel.from_pretrained(
        path,
        dtype=torch.float16,
        trust_remote_code=True,
    ).to("cpu")
    model.eval()
    return model


def _ensure_cpu_embed_pool() -> queue.Queue:
    global _cpu_embed_pool
    with _cpu_pool_lock:
        if _cpu_embed_pool is not None:
            return _cpu_embed_pool
        import torch
        from transformers import AutoTokenizer

        settings = get_lightrag_settings()
        replicas = settings.embedding_replicas
        threads = _cpu_thread_budget(replicas)
        torch.set_num_threads(threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        path = str(settings.embedding_model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            padding_side="left",
            trust_remote_code=True,
        )
        pool: queue.Queue = queue.Queue()
        for _ in range(replicas):
            pool.put((tokenizer, _load_one_cpu_model(path)))
        _cpu_embed_pool = pool
        return pool


def _cpu_embed(texts: list[str]) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    settings = get_lightrag_settings()
    pool = _ensure_cpu_embed_pool()
    tokenizer, model = pool.get()
    try:
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=settings.embedding_max_model_len,
        )
        encoded = {key: value.to("cpu") for key, value in encoded.items()}
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state
            pooled = _last_token_pool(hidden, encoded["attention_mask"])
            pooled = F.normalize(pooled.float(), p=2, dim=1)
        return pooled.cpu().numpy()
    finally:
        pool.put((tokenizer, model))


async def embedding_func(texts: list[str]) -> np.ndarray:
    settings = get_lightrag_settings()
    if settings.embedding_backend == "cpu":
        import asyncio

        return await asyncio.to_thread(_cpu_embed, list(texts))
    embedding = embedding_client().embeddings.create(
        model=embedding_model(),
        input=texts,
        encoding_format="float",
    )
    return np.array([item.embedding for item in embedding.data])


async def detect_embedding_dimension() -> int:
    try:
        test_result = await embedding_func(["test"])
        if test_result.ndim == 2:
            return int(test_result.shape[1])
        if test_result.ndim == 1:
            return int(test_result.shape[0])
    except Exception as exc:
        print(f"Warning: could not auto-detect embedding dimension ({exc}); using {embedding_dim()}")
    return embedding_dim()
