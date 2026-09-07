# Local models

```
models/llm/Qwen3.5-9B/                  # graph-extraction chat model
models/embedding/Qwen3-Embedding-4B/    # embeddings (2560-d)
```

Index build uses `configs/lightrag/build.yaml`. Utility QA can use
`configs/lightrag/query_dual.yaml` (two chat replicas + CPU embeddings).
