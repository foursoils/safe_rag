"""vLLM OpenAI server with a Gemma 4 / transformers 5.15 startup patch.

vLLM 0.27.1 still reads flat `config.head_dim` / `config.global_head_dim`.
Transformers >= 5.15 folds those into per-layer values (sliding 256 vs full
512) and raises AmbiguousGlobalPerLayerAttributeError on a global read.

Allowing global access is enough to get past config init, but then every
layer is built at 256 and weight load fails:

    AssertionError: Attempted to load weight (torch.Size([512]))
    into parameter (torch.Size([256]))

This module (1) sizes KV buffers to the max per-layer head_dim and (2)
wraps Gemma4DecoderLayer so each layer sees a homogeneous config. Engine
and worker processes load the same patch via the `vllm.general_plugins`
entry point and a PYTHONPATH sitecustomize hook.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_PATCHED = False


def _allow_global_per_layer_access(config: Any) -> None:
    if config is None:
        return
    try:
        config.allow_global_per_layer_attribute_access = True
    except Exception:
        pass
    for name in ("text_config", "vision_config", "audio_config", "language_config"):
        inner = getattr(config, name, None)
        if inner is not None and inner is not config:
            _allow_global_per_layer_access(inner)


def _cfg_get(config: Any, name: str, default: int = 0) -> int:
    getter = getattr(config, "_getattr_without_heterogeneous_validation", None)
    if callable(getter):
        try:
            value = getter(name, default)
            return default if value is None else int(value)
        except Exception:
            pass
    try:
        config.allow_global_per_layer_attribute_access = True
        value = getattr(config, name, default)
        return default if value is None else int(value)
    except Exception:
        return default


def homogeneous_layer_config(config: Any, layer_idx: int) -> Any:
    """Return a per-layer config whose `head_dim` / KV heads are unambiguous.

    Backport of vLLM 0.28 `gemma4_layer_config`: transformers 5.15 exposes
    `config.per_layer_config[i]`; older configs keep flat global aliases.
    """
    if getattr(config, "is_heterogeneous", False):
        try:
            return config.per_layer_config[layer_idx]
        except Exception:
            pass

    from copy import copy

    layer = copy(config)
    layer_types = getattr(config, "layer_types", None) or []
    if layer_idx < len(layer_types) and layer_types[layer_idx] == "full_attention":
        global_head_dim = _cfg_get(config, "global_head_dim")
        layer.head_dim = global_head_dim or _cfg_get(config, "head_dim")
        if getattr(config, "attention_k_eq_v", False):
            global_kv = _cfg_get(config, "num_global_key_value_heads")
            layer.num_key_value_heads = global_kv or _cfg_get(
                config, "num_key_value_heads"
            )
    return layer


def max_head_dim(config: Any) -> int:
    values = [_cfg_get(config, "head_dim"), _cfg_get(config, "global_head_dim")]
    try:
        if getattr(config, "is_heterogeneous", False) or hasattr(
            config, "per_layer_config"
        ):
            values.extend(int(layer.head_dim) for layer in config.per_layer_config)
    except Exception:
        pass
    return max((value for value in values if value), default=0)


def _install_worker_sitecustomize() -> None:
    """Engine/worker processes spawn with a clean interpreter; inherit this hook."""
    hook = Path(__file__).resolve().parent / "_vllm_pth"
    current = os.environ.get("PYTHONPATH", "")
    prefix = str(hook)
    if current.split(os.pathsep)[0] == prefix:
        return
    os.environ["PYTHONPATH"] = prefix + (os.pathsep + current if current else "")


def patch_gemma4_head_dim() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    import vllm.config.model as vllm_model_config
    from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer
    from vllm.model_executor.models.utils import extract_layer_index
    from vllm.transformers_utils import config as vllm_hf_config
    from vllm.transformers_utils.model_arch_config_convertor import (
        Gemma4ModelArchConfigConvertor,
    )

    original_get_config = vllm_hf_config.get_config

    def get_config(*args, **kwargs):
        config = original_get_config(*args, **kwargs)
        _allow_global_per_layer_access(config)
        return config

    vllm_hf_config.get_config = get_config
    vllm_model_config.get_config = get_config

    def get_head_size(self) -> int:
        cfg = self.hf_text_config
        _allow_global_per_layer_access(cfg)
        return max_head_dim(cfg) or super(
            Gemma4ModelArchConfigConvertor, self
        ).get_head_size()

    Gemma4ModelArchConfigConvertor.get_head_size = get_head_size

    original_init = Gemma4DecoderLayer.__init__

    def decoder_init(
        self,
        config,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        layer_idx = extract_layer_index(prefix)
        config = homogeneous_layer_config(config, layer_idx)
        original_init(self, config, cache_config, quant_config, prefix)

    Gemma4DecoderLayer.__init__ = decoder_init


def main() -> None:
    _install_worker_sitecustomize()
    patch_gemma4_head_dim()

    import uvloop
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
