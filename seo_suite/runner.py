"""Concurrent fan-out runner with retries and partial-result handling."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from seo_suite.models import ProviderResult, QueryTarget
from seo_suite.providers import SearchProvider


ProgressCallback = Callable[[int, int, ProviderResult], None]


def run_queries(
    targets: list[QueryTarget],
    provider: SearchProvider,
    runs_count: int,
    max_workers: int = 4,
    retries: int = 1,
    on_progress: ProgressCallback | None = None,
) -> list[ProviderResult]:
    """Run all target/run pairs with bounded concurrency and return ordered results."""

    runs_count = max(1, int(runs_count))
    max_workers = max(1, min(int(max_workers), len(targets) * runs_count if targets else 1))
    tasks = [(target_index, target, run_index) for target_index, target in enumerate(targets) for run_index in range(runs_count)]
    total = len(tasks)
    completed = 0
    ordered_results: list[tuple[tuple[int, int], ProviderResult]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_one, provider, target, run_index, retries): (target_index, run_index)
            for target_index, target, run_index in tasks
        }
        for future in as_completed(futures):
            target_index, run_index = futures[future]
            result = future.result()
            ordered_results.append(((target_index, run_index), result))
            completed += 1
            if on_progress:
                on_progress(completed, total, result)

    ordered_results.sort(key=lambda item: item[0])
    return [result for _, result in ordered_results]


def _run_one(
    provider: SearchProvider,
    target: QueryTarget,
    run_index: int,
    retries: int,
) -> ProviderResult:
    attempt = 0
    while True:
        try:
            return provider.run_query(target, run_index)
        except Exception as exc:
            if attempt >= retries:
                return ProviderResult(
                    query=target.query,
                    run_index=run_index,
                    provider=getattr(provider, "name", "unknown"),
                    model=getattr(provider, "model", ""),
                    response_text="",
                    sources=[],
                    fanout_queries=[],
                    grounding_metadata={},
                    error=f"{exc.__class__.__name__}: {exc}",
                )
            time.sleep(0.75 * (2**attempt))
            attempt += 1
