Datasets and built indexes. Tracked with Git LFS (`data/**` in `.gitattributes`).

```
data/corpus/<dataset>.json           # raw corpus
data/qa/<dataset>_questions.json     # GraphRAG-Bench gold QA (utility eval)
data/graphrag/<dataset>/             # GraphRAG index
data/lightrag/<dataset>/             # LightRAG workspace after build
```

Model weights stay local and untracked (`models/llm/`, `models/embedding/`). Paths: `configs/lightrag/build.yaml`.
