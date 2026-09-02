# safe_RAG

Attack–defense evaluation for GraphRAG systems. One experiment is a four-tuple `(attack, defense, system, dataset)`.

AGEA ([Query-Efficient Agentic Graph Extraction Attacks on GraphRAG Systems](https://arxiv.org/pdf/2601.14662), ACL 2026) is the first attack. The original paper snapshot lives in `third_party/agea/` and is not the runtime entrypoint.

## Layout

```
src/safe_rag/
  attacks/          # Attack interface + implementations (AGEA)
  defenses/         # Defense hooks; none/ is the baseline
    query/ retrieve/ generate/ index/   # reserved insertion points
  systems/          # Victim RAG adapters (GraphRAG, LightRAG)
  eval/             # runner, metrics, comparison reports
configs/experiments/
configs/lightrag/
data/               # corpora and indexes (Git LFS)
models/llm/         # local gemma-4-31B-it weights
results/            # run outputs (Git LFS)
third_party/agea/   # unmodified paper code
```

Defense hooks (all no-op in `none`):

- `on_query` — detect / rewrite / block extraction queries
- `on_retrieve` — filter or perturb recalled context
- `on_response` — limit structured entity/relation dumps
- `on_index` — perturb or partition the stored graph at build time

## Setup

```bash
python -m pip install -e .
# GraphRAG victim (optional):  python -m pip install -e ".[graphrag]"
```

LightRAG indexing is configured in `configs/lightrag/build.yaml`:

- Chat / graph extraction: local vLLM, weights in `models/llm/gemma-4-31B-it/`
- Embeddings: local vLLM, weights in `models/embedding/Qwen3-Embedding-4B/` (2560-d)

Put the LLM snapshot in `models/llm/gemma-4-31B-it/` and the embedding snapshot in `models/embedding/Qwen3-Embedding-4B/`, then from WSL:

```bash
bash scripts/build_lightrag.sh            # resume if workspace exists
bash scripts/build_lightrag.sh --rebuild  # wipe and rebuild
```

The script starts two vLLM processes (chat + embed). Logs go to `logs/lightrag/<dataset>/` (`log_dir` in the YAML plus the dataset name):

- `logs/lightrag/<dataset>/vllm-chat.log`
- `logs/lightrag/<dataset>/vllm-embed.log`
- `logs/lightrag/<dataset>/build.log`

The terminal shows stage banners and the insert progress bar. Output: `data/lightrag/<dataset>/`.

GPU assignment is in `configs/lightrag/build.yaml` (`vllm.devices` and `embedding.devices`). Tensor parallel size follows the device list: one card `[0]`; two-way TP `[0, 1]`; three-way TP `[0, 1, 2]`. The 31B chat model and 4B embed model currently share `[0, 1, 2, 3]`.

## Run

```bash
python -m safe_rag.eval.runner --config configs/experiments/agea_none_graphrag_medical.yaml
python -m safe_rag.eval.runner --config configs/experiments/agea_none_lightrag_medical.yaml
```

Outputs go to `results/agea_none_<system>_<dataset>/`.

To compare a new defense, add `src/safe_rag/defenses/<layer>/...`, register it, copy an experiment YAML, change `defense:`, and rerun. The runner does not need to change.

## Session graph isolation

The LightRAG experiment can isolate graph regions cumulatively across one AGEA
run:

```bash
python -m safe_rag.eval.runner --config configs/experiments/agea_isolation_lightrag_medical.yaml
```

`isolation` keeps state only for one `AgeaAttack.run` invocation. Its
default radius is one hop; `defense_kwargs.radius` accepts `1` or `2`. After a
turn, released entities and relations become forbidden together with the
configured-radius neighborhood in the original graph. If retrieval later
matches an already released entity, the defense may reuse that entity's cached
description, but it does not release newly retrieved relations through the
forbidden entity.

This initial implementation enforces isolation at the generation boundary:
LightRAG performs structured retrieval first, then only filtered entities,
relationships, and source chunks are rendered into the answer context. It does
not prevent LightRAG's storage layer from traversing forbidden graph data.
LightRAG must expose structured `aquery_llm` results with entity, relationship,
and chunk lists; unsupported or unstructured results fail closed instead of
falling back to raw context.
