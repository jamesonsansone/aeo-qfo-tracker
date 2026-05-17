"""Local file loading and writing helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from seo_suite.models import QueryTarget


REQUIRED_QUERY_COLUMNS = {"query"}

COLUMN_ALIASES = {
    "search_query": "query",
    "seed_query": "query",
    "seed_queries": "query",
    "keyword": "query",
    "prompt": "query",
    "question": "query",
    "type": "intent",
    "query_type": "intent",
    "strategic_category": "query_cluster",
    "category": "query_cluster",
    "cluster": "query_cluster",
    "domain": "target_domain",
    "url_pattern": "acceptable_url_pattern",
    "target_url_pattern": "acceptable_url_pattern",
}


def load_query_targets(
    path: str | Path,
    default_target_domain: str = "",
    default_target_url: str = "",
    default_acceptable_url_pattern: str = "",
    default_brand_aliases: str = "",
) -> list[QueryTarget]:
    df = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    return query_targets_from_dataframe(
        df,
        default_target_domain=default_target_domain,
        default_target_url=default_target_url,
        default_acceptable_url_pattern=default_acceptable_url_pattern,
        default_brand_aliases=default_brand_aliases,
    )


def query_targets_from_dataframe(
    df: pd.DataFrame,
    default_target_domain: str = "",
    default_target_url: str = "",
    default_acceptable_url_pattern: str = "",
    default_brand_aliases: str = "",
) -> list[QueryTarget]:
    df = df.fillna("")
    available_columns = {_canonical_column_key(column) for column in df.columns}
    missing = REQUIRED_QUERY_COLUMNS - available_columns
    if missing:
        found = ", ".join(str(column) for column in df.columns) or "(none)"
        raise ValueError(
            "Missing required query column. Include one of: Query, query, search_query, seed_query. "
            f"Found columns: {found}"
        )

    targets: list[QueryTarget] = []
    for row in df.to_dict("records"):
        row = _canonical_row(row)
        priority_raw = row.get("priority", 3)
        try:
            priority = int(priority_raw)
        except (TypeError, ValueError):
            priority = 3
        target_url = _first_present(row, "target_url") or default_target_url
        acceptable_url_pattern = (
            _first_present(row, "acceptable_url_pattern", "target_url_pattern", "url_pattern")
            or default_acceptable_url_pattern
            or target_url
        )
        targets.append(
            QueryTarget(
                query=str(row["query"]).strip(),
                query_cluster=_first_present(row, "query_cluster", "strategic_category", "category", "cluster"),
                intent=_first_present(row, "intent", "type", "query_type"),
                target_url=str(target_url).strip(),
                acceptable_url_pattern=str(acceptable_url_pattern).strip(),
                priority=max(1, min(priority, 5)),
                target_domain=_first_present(row, "target_domain", "domain") or default_target_domain,
                target_brand_aliases=(
                    _first_present(row, "target_brand_aliases", "brand_aliases", "target_brand")
                    or default_brand_aliases
                ),
            )
        )
    return targets


def _canonical_row(row: dict) -> dict:
    canonical = {}
    for column, value in row.items():
        key = _canonical_column_key(column)
        existing = canonical.get(key, "")
        if key not in canonical or (not str(existing).strip() and str(value).strip()):
            canonical[key] = value
    return canonical


def _canonical_column_key(column: str) -> str:
    key = _column_key(column)
    return COLUMN_ALIASES.get(key, key)


def _column_key(column: str) -> str:
    clean = str(column).replace("\ufeff", "").strip().lower()
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", clean)).strip("_")


def _first_present(row: dict, *columns: str) -> str:
    for column in columns:
        value = str(row.get(column, "") or "").strip()
        if value:
            return value
    return ""


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
