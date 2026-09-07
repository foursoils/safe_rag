# Experiment outputs

Tracked with Git LFS (`results/**` in `.gitattributes`). Each run writes to
`results/<attack>_<defense>_<system>_<dataset>/`.

Typical artifacts:
- `extracted_graph.json` / `.graphml`
- `query_history.json`
- `extraction_analysis.json`
- `run_summary.json`
- `utility_metrics.json` (benign QA runs)

Per-turn dumps live under `logs/<attack>_<defense>_<system>_<dataset>/turn_logs/`.
