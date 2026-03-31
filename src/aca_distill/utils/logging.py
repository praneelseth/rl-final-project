from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


class MetricAverager:
    def __init__(self) -> None:
        self.storage: dict[str, list[float]] = defaultdict(list)

    def update(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.storage[key].append(float(value))

    def compute(self) -> dict[str, float]:
        return {
            key: sum(values) / max(len(values), 1)
            for key, values in self.storage.items()
        }

    def reset(self) -> None:
        self.storage.clear()

