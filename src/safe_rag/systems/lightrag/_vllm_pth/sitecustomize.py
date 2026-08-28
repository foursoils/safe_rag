"""Imported automatically in vLLM engine/worker subprocesses.

`vllm_compat.main` prepends this directory to PYTHONPATH so spawned
interpreters apply the Gemma 4 per-layer head_dim patch before model init.
"""

from safe_rag.systems.lightrag.vllm_compat import patch_gemma4_head_dim

patch_gemma4_head_dim()
