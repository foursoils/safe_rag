from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"
LOGS_ROOT = REPO_ROOT / "logs"
THIRD_PARTY_AGEA = REPO_ROOT / "third_party" / "agea"
CONFIGS_ROOT = REPO_ROOT / "configs"
MODELS_ROOT = REPO_ROOT / "models"
LIGHTRAG_BUILD_CONFIG = CONFIGS_ROOT / "lightrag" / "build.yaml"
AGEA_LLM_CONFIG = CONFIGS_ROOT / "agea" / "llm.yaml"
