from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch


def measure_latency_ms(action_fn: Callable[[torch.Tensor], torch.Tensor], obs: torch.Tensor, repeats: int = 100) -> dict[str, float]:
    durations: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = action_fn(obs)
        if torch.cuda.is_available() and obs.is_cuda:
            torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) * 1000.0)
    return {
        "latency/mean_ms": float(np.mean(durations)),
        "latency/p50_ms": float(np.percentile(durations, 50)),
        "latency/p95_ms": float(np.percentile(durations, 95)),
    }


def iqm(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = np.sort(np.asarray(values, dtype=np.float32))
    n = len(ordered)
    lo = int(0.25 * n)
    hi = max(lo + 1, int(0.75 * n))
    return float(ordered[lo:hi].mean())

