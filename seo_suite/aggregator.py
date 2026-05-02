"""Aggregation and scoring for query opportunity analysis."""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from seo_suite.matching import (
    extract_domain,
    is_exact_url_match,
    is_pattern_match,
    is_same_or_subdomain,
)
from seo_suite.models import Citation, ProviderResult, QueryTarget, QueryRun


def build_query_runs(results: list[ProviderResult], targets: list[QueryTarget], target_domain: str) -> list[QueryRun]:
    target_by_query = {target.query: target for target in targets}
    runs: list[QueryRun] = []
    for result in results:
        target = target_by_query[result.query]
        citations: list[Citation] = []
        for index, source in enumerate(result.sources, start=1):
            cited_url = source.get("url") or source.get("uri") or ""
            cited_domain = extract_domain(cited_url)
            citations.append(
                Citation(
                    query=result.query,
                    run_index=result.run_index,
                    cited_url=cited_url,
                    cited_domain=cited_domain,
                    citation_position=index,
                    title=source.get("title", ""),
                    is_target_domain=is_same_or_subdomain(cited_domain, target_domain),
                    is_exact_target_url=is_exact_url_match(cited_url, target.target_url),
                    is_acceptable_target_url=is_pattern_match(cited_url, target.acceptable_url_pattern),
                )
            )
        runs.append(
            QueryRun(
                query=result.query,
                run_index=result.run_index,
                provider=result.provider,
                model=result.model,
                response_text=result.response_text,
                citations=citations,
            )
        )
    return runs


def build_tables(targets: list[QueryTarget], runs: list[QueryRun], target_domain: str) -> dict[str, pd.DataFrame]:
    citations = _citation_rows(runs)
    citations_df = pd.DataFrame(
        citations,
        columns=[
            "query",
            "run_index",
            "cited_url",
            "cited_domain",
            "title",
            "citation_position",
            "is_target_domain",
            "is_exact_target_url",
            "is_acceptable_target_url",
        ],
    )
    runs_df = pd.DataFrame(
        [
            {
                "query": run.query,
                "run_index": run.run_index,
                "provider": run.provider,
                "model": run.model,
                "response_text": run.response_text,
                "citation_count": len(run.citations),
            }
            for run in runs
        ]
    )
    query_metrics_df = _query_metrics(targets, runs, citations_df, target_domain)
    domain_metrics_df = _domain_metrics(citations_df)
    url_metrics_df = _url_metrics(citations_df)
    overview_df = _overview(query_metrics_df, domain_metrics_df, target_domain)
    return {
        "runs": runs_df,
        "citations": citations_df,
        "query_metrics": query_metrics_df,
        "domain_metrics": domain_metrics_df,
        "url_metrics": url_metrics_df,
        "overview": overview_df,
    }


def _citation_rows(runs: list[QueryRun]) -> list[dict]:
    rows: list[dict] = []
    for run in runs:
        for citation in run.citations:
            rows.append(
                {
                    "query": citation.query,
                    "run_index": citation.run_index,
                    "cited_url": citation.cited_url,
                    "cited_domain": citation.cited_domain,
                    "title": citation.title,
                    "citation_position": citation.citation_position,
                    "is_target_domain": citation.is_target_domain,
                    "is_exact_target_url": citation.is_exact_target_url,
                    "is_acceptable_target_url": citation.is_acceptable_target_url,
                }
            )
    return rows


def _query_metrics(targets: list[QueryTarget], runs: list[QueryRun], citations_df: pd.DataFrame, target_domain: str) -> pd.DataFrame:
    runs_by_query: dict[str, list[QueryRun]] = defaultdict(list)
    for run in runs:
        runs_by_query[run.query].append(run)

    rows = []
    for target in targets:
        query_runs = runs_by_query.get(target.query, [])
        run_count = len(query_runs)
        query_citations = citations_df[citations_df["query"] == target.query] if not citations_df.empty else pd.DataFrame()
        exact_positions = _first_positions(query_runs, "is_exact_target_url")
        acceptable_positions = _first_positions(query_runs, "is_acceptable_target_url")
        domain_positions = _first_positions(query_runs, "is_target_domain")
        competitor = _top_competitor(query_citations)
        citation_total = int(len(query_citations)) if not query_citations.empty else 0
        target_citation_total = int(query_citations["is_target_domain"].sum()) if not query_citations.empty else 0
        acceptable_seen = _presence_rate(acceptable_positions, run_count)
        domain_seen = _presence_rate(domain_positions, run_count)
        exact_seen = _presence_rate(exact_positions, run_count)
        mention_rate = _mention_rate(query_runs, target_domain, target.target_url)
        target_mismatch = 1 if domain_seen > 0 and acceptable_seen == 0 else 0
        competitor_pressure = min(35.0, competitor["citations"] * 4.0)
        visibility_gap = (1.0 - acceptable_seen / 100.0) * 45.0
        priority_weight = float(target.priority) * 4.0
        mismatch_penalty = 15.0 if target_mismatch else 0.0
        opportunity_score = round(min(100.0, visibility_gap + competitor_pressure + priority_weight + mismatch_penalty), 1)

        rows.append(
            {
                "query": target.query,
                "query_cluster": target.query_cluster,
                "intent": target.intent,
                "priority": target.priority,
                "target_url": target.target_url,
                "acceptable_url_pattern": target.acceptable_url_pattern,
                "runs_count": run_count,
                "target_domain_presence_rate": round(domain_seen, 1),
                "exact_target_url_presence_rate": round(exact_seen, 1),
                "acceptable_target_presence_rate": round(acceptable_seen, 1),
                "target_mention_rate": round(mention_rate, 1),
                "domain_citation_rate": round(domain_seen, 1),
                "exact_url_citation_rate": round(exact_seen, 1),
                "approved_page_citation_rate": round(acceptable_seen, 1),
                "best_exact_position": _best_position(exact_positions),
                "best_acceptable_position": _best_position(acceptable_positions),
                "best_domain_position": _best_position(domain_positions),
                "exact_mrr": round(_mrr(exact_positions, run_count), 3),
                "acceptable_mrr": round(_mrr(acceptable_positions, run_count), 3),
                "domain_mrr": round(_mrr(domain_positions, run_count), 3),
                "citation_share": round((target_citation_total / citation_total * 100.0) if citation_total else 0.0, 1),
                "top_competitor_domain": competitor["domain"],
                "top_competitor_url": competitor["url"],
                "top_competitor_citations": competitor["citations"],
                "target_mismatch": bool(target_mismatch),
                "opportunity_score": opportunity_score,
            }
        )
    return pd.DataFrame(rows).sort_values("opportunity_score", ascending=False).reset_index(drop=True)


def _first_positions(runs: list[QueryRun], attr: str) -> list[int | None]:
    positions: list[int | None] = []
    for run in runs:
        matching = [citation.citation_position for citation in run.citations if getattr(citation, attr)]
        positions.append(min(matching) if matching else None)
    return positions


def _presence_rate(positions: list[int | None], run_count: int) -> float:
    if run_count == 0:
        return 0.0
    return sum(position is not None for position in positions) / run_count * 100.0


def _best_position(positions: list[int | None]) -> int | None:
    seen = [position for position in positions if position is not None]
    return min(seen) if seen else None


def _mrr(positions: list[int | None], run_count: int) -> float:
    if run_count == 0:
        return 0.0
    return sum((1.0 / position) if position else 0.0 for position in positions) / run_count


def _mention_rate(runs: list[QueryRun], target_domain: str, target_url: str) -> float:
    if not runs:
        return 0.0
    tokens = _mention_tokens(target_domain, target_url)
    if not tokens:
        return 0.0
    mentioned = 0
    for run in runs:
        text = (run.response_text or "").lower()
        if any(token in text for token in tokens):
            mentioned += 1
    return mentioned / len(runs) * 100.0


def _mention_tokens(target_domain: str, target_url: str) -> set[str]:
    domains = [extract_domain(target_domain), extract_domain(target_url)]
    tokens: set[str] = set()
    for domain in domains:
        if not domain:
            continue
        tokens.add(domain)
        first_label = domain.split(".")[0].replace("-", "")
        if first_label:
            tokens.add(first_label)
    return tokens


def _top_competitor(query_citations: pd.DataFrame) -> dict:
    if query_citations.empty:
        return {"domain": "-", "url": "-", "citations": 0}
    competitors = query_citations[~query_citations["is_target_domain"]]
    if competitors.empty:
        return {"domain": "-", "url": "-", "citations": 0}
    top_url = competitors["cited_url"].value_counts().index[0]
    top_url_rows = competitors[competitors["cited_url"] == top_url]
    return {
        "domain": str(top_url_rows.iloc[0]["cited_domain"]),
        "url": str(top_url),
        "citations": int(len(top_url_rows)),
    }


def _domain_metrics(citations_df: pd.DataFrame) -> pd.DataFrame:
    if citations_df.empty:
        return pd.DataFrame(columns=["cited_domain", "category", "total_citations", "queries_count", "queries_list"])
    rows = []
    for domain, group in citations_df.groupby("cited_domain"):
        rows.append(
            {
                "cited_domain": domain,
                "category": "Target" if bool(group["is_target_domain"].any()) else "Other",
                "total_citations": int(len(group)),
                "queries_count": int(group["query"].nunique()),
                "queries_list": " | ".join(sorted(group["query"].unique())),
            }
        )
    return pd.DataFrame(rows).sort_values("total_citations", ascending=False).reset_index(drop=True)


def _url_metrics(citations_df: pd.DataFrame) -> pd.DataFrame:
    if citations_df.empty:
        return pd.DataFrame(columns=["cited_url", "cited_domain", "category", "total_citations", "queries_count", "queries_list"])
    rows = []
    for url, group in citations_df.groupby("cited_url"):
        rows.append(
            {
                "cited_url": url,
                "cited_domain": group.iloc[0]["cited_domain"],
                "title": group.iloc[0].get("title", ""),
                "category": "Target" if bool(group["is_target_domain"].any()) else "Other",
                "total_citations": int(len(group)),
                "queries_count": int(group["query"].nunique()),
                "queries_list": " | ".join(sorted(group["query"].unique())),
            }
        )
    return pd.DataFrame(rows).sort_values("total_citations", ascending=False).reset_index(drop=True)


def _overview(query_metrics_df: pd.DataFrame, domain_metrics_df: pd.DataFrame, target_domain: str) -> pd.DataFrame:
    if query_metrics_df.empty:
        return pd.DataFrame()
    top_competitor = "-"
    competitors = domain_metrics_df[domain_metrics_df["category"] != "Target"] if not domain_metrics_df.empty else pd.DataFrame()
    if not competitors.empty:
        top_competitor = str(competitors.iloc[0]["cited_domain"])
    values = {
        "target_domain": target_domain,
        "queries": int(len(query_metrics_df)),
        "avg_opportunity_score": round(float(query_metrics_df["opportunity_score"].mean()), 1),
        "avg_acceptable_presence_rate": round(float(query_metrics_df["acceptable_target_presence_rate"].mean()), 1),
        "avg_domain_presence_rate": round(float(query_metrics_df["target_domain_presence_rate"].mean()), 1),
        "avg_acceptable_mrr": round(float(query_metrics_df["acceptable_mrr"].mean()), 3),
        "avg_target_mention_rate": round(float(query_metrics_df["target_mention_rate"].mean()), 1),
        "avg_approved_page_citation_rate": round(float(query_metrics_df["approved_page_citation_rate"].mean()), 1),
        "zero_visibility_queries": int((query_metrics_df["acceptable_target_presence_rate"] == 0).sum()),
        "target_mismatch_queries": int(query_metrics_df["target_mismatch"].sum()),
        "top_competitor_domain": top_competitor,
    }
    return pd.DataFrame([values])


def make_citation_matrix(citations_df: pd.DataFrame, group_by: str = "cited_domain") -> pd.DataFrame:
    if citations_df.empty:
        return pd.DataFrame()
    matrix = citations_df.pivot_table(
        index="query",
        columns=group_by,
        values="cited_url",
        aggfunc="count",
        fill_value=0,
    )
    matrix["total_citations"] = matrix.sum(axis=1)
    matrix = matrix.sort_values("total_citations", ascending=False)
    return matrix.reset_index()
