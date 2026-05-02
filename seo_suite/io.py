"""Local file loading and writing helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from seo_suite.models import QueryTarget


REQUIRED_QUERY_COLUMNS = {
    "query",
    "query_cluster",
    "intent",
    "target_url",
    "acceptable_url_pattern",
}


def load_query_targets(path: str | Path) -> list[QueryTarget]:
    df = pd.read_csv(path).fillna("")
    missing = REQUIRED_QUERY_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required query columns: {', '.join(sorted(missing))}")

    targets: list[QueryTarget] = []
    for row in df.to_dict("records"):
        priority_raw = row.get("priority", 3)
        try:
            priority = int(priority_raw)
        except (TypeError, ValueError):
            priority = 3
        targets.append(
            QueryTarget(
                query=str(row["query"]).strip(),
                query_cluster=str(row["query_cluster"]).strip(),
                intent=str(row["intent"]).strip(),
                target_url=str(row["target_url"]).strip(),
                acceptable_url_pattern=str(row["acceptable_url_pattern"]).strip(),
                priority=max(1, min(priority, 5)),
            )
        )
    return targets


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_outputs(output_dir: str | Path, tables: dict[str, pd.DataFrame], brief: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "client_brief.md").write_text(brief, encoding="utf-8")

