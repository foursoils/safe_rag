from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol


@dataclass
class AttackResult:
    output_dir: Path
    metrics: dict[str, Any] = field(default_factory=dict)
    query_history_path: Optional[Path] = None


class Attack(Protocol):
    name: str

    def run(
        self,
        system: Any,
        defense: Any,
        dataset: str,
        budget: int,
        output_dir: Path,
        config: Optional[dict[str, Any]] = None,
    ) -> AttackResult: ...
