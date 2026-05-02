"""SQLite persistence for daily AI search tracking snapshots."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from seo_suite.models import QueryTarget


def init_db(db_path: str) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)


def save_run_batch(
    db_path: str,
    query_set_name: str,
    target_domain: str,
    tables: dict[str, pd.DataFrame],
    targets: list[QueryTarget],
    provider: str = "",
    model: str = "",
    runs_per_query: int = 0,
    notes: str = "",
    run_date: str | None = None,
) -> int:
    init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    batch_date = run_date or datetime.now(timezone.utc).date().isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        query_set_id = _get_or_create_query_set(conn, query_set_name, target_domain, now)
        target_ids = _upsert_targets(conn, query_set_id, targets, now)
        cursor = conn.execute(
            """
            INSERT INTO run_batches
              (query_set_id, run_date, provider, model, runs_per_query, created_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (query_set_id, batch_date, provider, model, runs_per_query, now, notes),
        )
        batch_id = int(cursor.lastrowid)
        run_ids = _insert_runs(conn, batch_id, target_ids, tables.get("runs", pd.DataFrame()), now)
        _insert_citations(conn, run_ids, tables.get("citations", pd.DataFrame()))
        _insert_query_metrics(conn, batch_id, target_ids, tables.get("query_metrics", pd.DataFrame()))
        _insert_page_diagnostics(conn, batch_id, tables.get("page_diagnostics", pd.DataFrame()))
        _insert_content_alignment(conn, batch_id, target_ids, tables.get("content_alignment", pd.DataFrame()))
        _insert_weekly_queue(conn, batch_id, target_ids, tables.get("weekly_queue", pd.DataFrame()))
        return batch_id


def load_recent_batches(db_path: str, query_set_name: str, days: int = 56) -> dict[str, pd.DataFrame]:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        params = {"name": query_set_name, "days": f"-{days} days"}
        batches = pd.read_sql_query(
            """
            SELECT rb.*, qs.name AS query_set_name, qs.target_domain
            FROM run_batches rb
            JOIN query_sets qs ON qs.id = rb.query_set_id
            WHERE qs.name = :name AND date(rb.run_date) >= date('now', :days)
            ORDER BY rb.run_date DESC, rb.id DESC
            """,
            conn,
            params=params,
        )
        if batches.empty:
            return _empty_recent()
        batch_ids = tuple(int(value) for value in batches["id"].tolist())
        placeholders = ",".join("?" for _ in batch_ids)
        return {
            "run_batches": batches,
            "query_metrics": pd.read_sql_query(
                f"""
                SELECT qms.*, qt.query, qt.query_cluster, qt.intent, qt.target_url
                FROM query_metric_snapshots qms
                JOIN query_targets qt ON qt.id = qms.query_target_id
                WHERE qms.batch_id IN ({placeholders})
                """,
                conn,
                params=batch_ids,
            ),
            "weekly_queue": pd.read_sql_query(
                f"""
                SELECT wqi.*, qt.query, qt.query_cluster, qt.target_url
                FROM weekly_queue_items wqi
                JOIN query_targets qt ON qt.id = wqi.query_target_id
                WHERE wqi.batch_id IN ({placeholders})
                """,
                conn,
                params=batch_ids,
            ),
            "content_alignment": pd.read_sql_query(
                f"""
                SELECT cas.*, qt.query, qt.query_cluster
                FROM content_alignment_snapshots cas
                JOIN query_targets qt ON qt.id = cas.query_target_id
                WHERE cas.batch_id IN ({placeholders})
                """,
                conn,
                params=batch_ids,
            ),
        }


def build_daily_trends(db_path: str, query_set_name: str) -> pd.DataFrame:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT
              rb.id AS batch_id,
              rb.run_date,
              AVG(qms.mention_rate) AS mention_rate,
              AVG(qms.domain_citation_rate) AS domain_citation_rate,
              AVG(qms.approved_page_citation_rate) AS approved_page_citation_rate,
              AVG(qms.approved_page_rrf_score) AS approved_page_rrf_score,
              SUM(CASE WHEN qms.approved_page_citation_rate = 0 THEN 1 ELSE 0 END) AS zero_visibility_queries,
              SUM(CASE WHEN qms.wrong_client_page_cited = 1 THEN 1 ELSE 0 END) AS wrong_page_queries,
              AVG(qms.opportunity_score) AS avg_opportunity_score
            FROM run_batches rb
            JOIN query_sets qs ON qs.id = rb.query_set_id
            JOIN query_metric_snapshots qms ON qms.batch_id = rb.id
            WHERE qs.name = ?
            GROUP BY rb.id, rb.run_date
            ORDER BY rb.run_date ASC, rb.id ASC
            """,
            conn,
            params=(query_set_name,),
        )
    return df


def build_weekly_rollups(db_path: str, query_set_name: str, weeks: int = 8) -> pd.DataFrame:
    daily = build_daily_trends(db_path, query_set_name)
    if daily.empty:
        return daily
    daily["run_date"] = pd.to_datetime(daily["run_date"])
    cutoff = daily["run_date"].max() - pd.Timedelta(weeks=weeks)
    daily = daily[daily["run_date"] >= cutoff].copy()
    daily["week_start"] = daily["run_date"].dt.to_period("W-SUN").dt.start_time.dt.date.astype(str)
    return (
        daily.groupby("week_start", as_index=False)
        .agg(
            mention_rate=("mention_rate", "mean"),
            domain_citation_rate=("domain_citation_rate", "mean"),
            approved_page_citation_rate=("approved_page_citation_rate", "mean"),
            approved_page_rrf_score=("approved_page_rrf_score", "mean"),
            zero_visibility_queries=("zero_visibility_queries", "mean"),
            wrong_page_queries=("wrong_page_queries", "mean"),
            avg_opportunity_score=("avg_opportunity_score", "mean"),
        )
        .round(3)
    )


def build_historical_queue(db_path: str, query_set_name: str) -> pd.DataFrame:
    recent = load_recent_batches(db_path, query_set_name, days=56)
    batches = recent["run_batches"]
    queue = recent["weekly_queue"]
    if batches.empty or queue.empty:
        return pd.DataFrame()
    latest_batch = int(batches.iloc[0]["id"])
    previous_batches = batches[batches["id"] != latest_batch]
    latest = queue[queue["batch_id"] == latest_batch].copy()
    if previous_batches.empty:
        latest["issue_status"] = "new issue"
        return latest.sort_values("weekly_priority_score", ascending=False)
    previous_batch = int(previous_batches.iloc[0]["id"])
    previous = queue[queue["batch_id"] == previous_batch].set_index("query").to_dict("index")
    statuses = []
    for row in latest.to_dict("records"):
        old = previous.get(row["query"])
        if old is None:
            statuses.append("new issue")
        elif old["recommended_follow_up"] == row["recommended_follow_up"]:
            statuses.append("repeated issue")
        elif row["weekly_priority_score"] < old["weekly_priority_score"]:
            statuses.append("improved")
        else:
            statuses.append("worsened")
    latest["issue_status"] = statuses
    return latest.sort_values("weekly_priority_score", ascending=False)


def _get_or_create_query_set(conn: sqlite3.Connection, name: str, target_domain: str, now: str) -> int:
    row = conn.execute("SELECT id FROM query_sets WHERE name = ? AND target_domain = ?", (name, target_domain)).fetchone()
    if row:
        return int(row["id"])
    cursor = conn.execute(
        "INSERT INTO query_sets (name, target_domain, created_at) VALUES (?, ?, ?)",
        (name, target_domain, now),
    )
    return int(cursor.lastrowid)


def _upsert_targets(conn: sqlite3.Connection, query_set_id: int, targets: list[QueryTarget], now: str) -> dict[str, int]:
    ids = {}
    for target in targets:
        row = conn.execute(
            "SELECT id FROM query_targets WHERE query_set_id = ? AND query = ?",
            (query_set_id, target.query),
        ).fetchone()
        if row:
            target_id = int(row["id"])
            conn.execute(
                """
                UPDATE query_targets
                SET query_cluster = ?, intent = ?, target_url = ?, acceptable_url_pattern = ?, priority = ?, active = 1
                WHERE id = ?
                """,
                (
                    target.query_cluster,
                    target.intent,
                    target.target_url,
                    target.acceptable_url_pattern,
                    target.priority,
                    target_id,
                ),
            )
        else:
            cursor = conn.execute(
                """
                INSERT INTO query_targets
                  (query_set_id, query, query_cluster, intent, target_url, acceptable_url_pattern, priority, active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    query_set_id,
                    target.query,
                    target.query_cluster,
                    target.intent,
                    target.target_url,
                    target.acceptable_url_pattern,
                    target.priority,
                    now,
                ),
            )
            target_id = int(cursor.lastrowid)
        ids[target.query] = target_id
    return ids


def _insert_runs(conn: sqlite3.Connection, batch_id: int, target_ids: dict[str, int], runs: pd.DataFrame, now: str) -> dict[tuple[str, int], int]:
    run_ids = {}
    for row in runs.to_dict("records"):
        query = row["query"]
        run_index = int(row["run_index"])
        cursor = conn.execute(
            """
            INSERT INTO query_runs
              (batch_id, query_target_id, run_index, response_text, source_urls_cited, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                target_ids[query],
                run_index,
                row.get("response_text", ""),
                int(row.get("citation_count", 0)),
                now,
            ),
        )
        run_ids[(query, run_index)] = int(cursor.lastrowid)
    return run_ids


def _insert_citations(conn: sqlite3.Connection, run_ids: dict[tuple[str, int], int], citations: pd.DataFrame) -> None:
    for row in citations.to_dict("records"):
        conn.execute(
            """
            INSERT INTO citations
              (query_run_id, cited_url, cited_domain, title, citation_position,
               is_target_domain, is_exact_target_url, is_approved_target_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_ids[(row["query"], int(row["run_index"]))],
                row.get("cited_url", ""),
                row.get("cited_domain", ""),
                row.get("title", ""),
                int(row.get("citation_position", 0)),
                _bool(row.get("is_target_domain")),
                _bool(row.get("is_exact_target_url")),
                _bool(row.get("is_acceptable_target_url")),
            ),
        )


def _insert_query_metrics(conn: sqlite3.Connection, batch_id: int, target_ids: dict[str, int], metrics: pd.DataFrame) -> None:
    for row in metrics.to_dict("records"):
        conn.execute(
            """
            INSERT INTO query_metric_snapshots
              (batch_id, query_target_id, mention_rate, domain_citation_rate,
               exact_url_citation_rate, approved_page_citation_rate, approved_page_rrf_score,
               competitor_domain, competitor_url, wrong_client_page_cited, opportunity_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                target_ids[row["query"]],
                float(row.get("target_mention_rate", 0)),
                float(row.get("domain_citation_rate", 0)),
                float(row.get("exact_url_citation_rate", 0)),
                float(row.get("approved_page_citation_rate", 0)),
                float(row.get("acceptable_mrr", 0)),
                row.get("top_competitor_domain", ""),
                row.get("top_competitor_url", ""),
                _bool(row.get("target_mismatch")),
                float(row.get("opportunity_score", 0)),
            ),
        )


def _insert_page_diagnostics(conn: sqlite3.Connection, batch_id: int, diagnostics: pd.DataFrame) -> None:
    for row in diagnostics.to_dict("records"):
        conn.execute(
            """
            INSERT INTO page_diagnostic_snapshots
              (batch_id, url, domain, raw_html_available, rendered_html_available,
               product_schema_raw, product_schema_rendered, offer_schema_raw, offer_schema_rendered,
               raw_render_delta_type, word_count_raw, word_count_rendered,
               internal_links_count, external_outlinks_count, lcp, inp, cls,
               pagespeed_performance_score, diagnostic_flags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                row.get("url", ""),
                row.get("domain", ""),
                _bool(row.get("raw_html_available")),
                _bool(row.get("rendered_html_available")),
                _bool(row.get("product_schema_raw")),
                _bool(row.get("product_schema_rendered")),
                _bool(row.get("offer_schema_raw")),
                _bool(row.get("offer_schema_rendered")),
                row.get("raw_render_delta_type", ""),
                _int(row.get("word_count_raw")),
                _int(row.get("word_count_rendered")),
                _int(row.get("internal_links_count")),
                _int(row.get("external_outlinks_count")),
                _float_or_none(row.get("lcp")),
                _float_or_none(row.get("inp")),
                _float_or_none(row.get("cls")),
                _float_or_none(row.get("pagespeed_performance_score")),
                row.get("diagnostic_flags", ""),
            ),
        )


def _insert_content_alignment(conn: sqlite3.Connection, batch_id: int, target_ids: dict[str, int], alignment: pd.DataFrame) -> None:
    for row in alignment.to_dict("records"):
        conn.execute(
            """
            INSERT INTO content_alignment_snapshots
              (batch_id, query_target_id, url, url_role, source_text_mode,
               best_matching_heading, best_chunk_text, tfidf_score, embedding_score,
               answer_capsule_present, alignment_gap_vs_top_competitor)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                target_ids[row["query"]],
                row.get("url", ""),
                row.get("url_role", ""),
                row.get("source_text_mode", ""),
                row.get("best_matching_heading", ""),
                row.get("best_chunk_text", ""),
                float(row.get("tfidf_score", 0)),
                _float_or_none(row.get("embedding_score")),
                _bool(row.get("answer_capsule_present")),
                float(row.get("alignment_gap_vs_top_competitor", 0)),
            ),
        )


def _insert_weekly_queue(conn: sqlite3.Connection, batch_id: int, target_ids: dict[str, int], queue: pd.DataFrame) -> None:
    for row in queue.to_dict("records"):
        conn.execute(
            """
            INSERT INTO weekly_queue_items
              (batch_id, query_target_id, recommended_follow_up, weekly_priority_score, diagnostic_flags)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                target_ids[row["query"]],
                row.get("recommended_follow_up", ""),
                float(row.get("weekly_priority_score", 0)),
                row.get("diagnostic_flags", ""),
            ),
        )


def _bool(value) -> int:
    return 1 if bool(value) else 0


def _int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_or_none(value):
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def _empty_recent() -> dict[str, pd.DataFrame]:
    return {
        "run_batches": pd.DataFrame(),
        "query_metrics": pd.DataFrame(),
        "weekly_queue": pd.DataFrame(),
        "content_alignment": pd.DataFrame(),
    }


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS query_sets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  target_domain TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(name, target_domain)
);

CREATE TABLE IF NOT EXISTS query_targets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  query_set_id INTEGER NOT NULL,
  query TEXT NOT NULL,
  query_cluster TEXT,
  intent TEXT,
  target_url TEXT,
  acceptable_url_pattern TEXT,
  priority INTEGER,
  active INTEGER DEFAULT 1,
  created_at TEXT NOT NULL,
  UNIQUE(query_set_id, query),
  FOREIGN KEY(query_set_id) REFERENCES query_sets(id)
);

CREATE TABLE IF NOT EXISTS run_batches (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  query_set_id INTEGER NOT NULL,
  run_date TEXT NOT NULL,
  provider TEXT,
  model TEXT,
  runs_per_query INTEGER,
  created_at TEXT NOT NULL,
  notes TEXT,
  FOREIGN KEY(query_set_id) REFERENCES query_sets(id)
);

CREATE TABLE IF NOT EXISTS query_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id INTEGER NOT NULL,
  query_target_id INTEGER NOT NULL,
  run_index INTEGER NOT NULL,
  response_text TEXT,
  source_urls_cited INTEGER,
  created_at TEXT NOT NULL,
  FOREIGN KEY(batch_id) REFERENCES run_batches(id),
  FOREIGN KEY(query_target_id) REFERENCES query_targets(id)
);

CREATE TABLE IF NOT EXISTS citations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  query_run_id INTEGER NOT NULL,
  cited_url TEXT,
  cited_domain TEXT,
  title TEXT,
  citation_position INTEGER,
  is_target_domain INTEGER,
  is_exact_target_url INTEGER,
  is_approved_target_url INTEGER,
  FOREIGN KEY(query_run_id) REFERENCES query_runs(id)
);

CREATE TABLE IF NOT EXISTS query_metric_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id INTEGER NOT NULL,
  query_target_id INTEGER NOT NULL,
  mention_rate REAL,
  domain_citation_rate REAL,
  exact_url_citation_rate REAL,
  approved_page_citation_rate REAL,
  approved_page_rrf_score REAL,
  competitor_domain TEXT,
  competitor_url TEXT,
  wrong_client_page_cited INTEGER,
  opportunity_score REAL,
  FOREIGN KEY(batch_id) REFERENCES run_batches(id),
  FOREIGN KEY(query_target_id) REFERENCES query_targets(id)
);

CREATE TABLE IF NOT EXISTS page_diagnostic_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id INTEGER NOT NULL,
  url TEXT,
  domain TEXT,
  raw_html_available INTEGER,
  rendered_html_available INTEGER,
  product_schema_raw INTEGER,
  product_schema_rendered INTEGER,
  offer_schema_raw INTEGER,
  offer_schema_rendered INTEGER,
  raw_render_delta_type TEXT,
  word_count_raw INTEGER,
  word_count_rendered INTEGER,
  internal_links_count INTEGER,
  external_outlinks_count INTEGER,
  lcp REAL,
  inp REAL,
  cls REAL,
  pagespeed_performance_score REAL,
  diagnostic_flags TEXT,
  FOREIGN KEY(batch_id) REFERENCES run_batches(id)
);

CREATE TABLE IF NOT EXISTS content_alignment_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id INTEGER NOT NULL,
  query_target_id INTEGER NOT NULL,
  url TEXT,
  url_role TEXT,
  source_text_mode TEXT,
  best_matching_heading TEXT,
  best_chunk_text TEXT,
  tfidf_score REAL,
  embedding_score REAL,
  answer_capsule_present INTEGER,
  alignment_gap_vs_top_competitor REAL,
  FOREIGN KEY(batch_id) REFERENCES run_batches(id),
  FOREIGN KEY(query_target_id) REFERENCES query_targets(id)
);

CREATE TABLE IF NOT EXISTS weekly_queue_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id INTEGER NOT NULL,
  query_target_id INTEGER NOT NULL,
  recommended_follow_up TEXT,
  weekly_priority_score REAL,
  diagnostic_flags TEXT,
  FOREIGN KEY(batch_id) REFERENCES run_batches(id),
  FOREIGN KEY(query_target_id) REFERENCES query_targets(id)
);
"""
