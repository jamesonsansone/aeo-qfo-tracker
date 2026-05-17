"""Aggregation and scoring for AI search fan-out analysis."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import pandas as pd

from seo_suite.brand_mentions import (
    brand_from_domain,
    build_brand_mentions_table,
    canonical_brand,
    split_aliases,
)
from seo_suite.matching import (
    extract_domain,
    is_exact_url_match,
    is_pattern_match,
    is_same_or_subdomain,
)
from seo_suite.models import Citation, ProviderResult, QueryRun, QueryTarget


def build_query_runs(
    results: list[ProviderResult],
    targets: list[QueryTarget],
    target_domain: str = "",
) -> list[QueryRun]:
    target_by_query = {target.query: target for target in targets}
    runs: list[QueryRun] = []
    for result in results:
        target = target_by_query[result.query]
        active_target_domain = target.target_domain or target_domain
        citations: list[Citation] = []
        for index, source in enumerate(result.sources, start=1):
            citation_kind = source.get("citation_kind") or ""
            fallback_domain = source.get("fallback_domain", "")
            cited_url = source.get("url") or source.get("resolved_url") or source.get("uri") or ""
            if not citation_kind:
                citation_kind = "resolved_url" if cited_url else ("domain_fallback" if fallback_domain else "unresolved")
            original_url = source.get("original_url") or source.get("uri") or cited_url
            resolved_url = source.get("resolved_url") or (cited_url if citation_kind == "resolved_url" else "")
            cited_domain = extract_domain(cited_url) or fallback_domain
            citations.append(
                Citation(
                    query=result.query,
                    run_index=result.run_index,
                    cited_url=cited_url,
                    cited_domain=cited_domain,
                    citation_position=index,
                    title=source.get("title", ""),
                    target_domain=active_target_domain,
                    original_url=original_url,
                    resolved_url=resolved_url,
                    resolution_status=source.get("resolution_status", "direct"),
                    resolution_method=source.get("resolution_method", "direct"),
                    final_status_code=source.get("final_status_code"),
                    final_domain=source.get("final_domain") or cited_domain,
                    resolution_error=source.get("resolution_error", ""),
                    source_label=source.get("source_label", "") or source.get("title", ""),
                    fallback_domain=fallback_domain,
                    citation_kind=citation_kind,
                    source_type=source.get("source_type", ""),
                    is_target_domain=is_same_or_subdomain(cited_domain, active_target_domain),
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
                fanout_queries=list(result.fanout_queries or []),
                grounding_metadata=dict(result.grounding_metadata or {}),
                error=result.error,
            )
        )
    return runs


def build_tables(
    targets: list[QueryTarget],
    runs: list[QueryRun],
    target_domain: str = "",
    brand_aliases: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    all_citations_df = pd.DataFrame(_citation_rows(runs), columns=CITATION_COLUMNS)
    citations_df = _valid_citations_df(all_citations_df)
    unresolved_redirects_df = _unresolved_redirects_df(all_citations_df)
    brand_mentions_df, run_mentions, brand_run_mentions_df = build_brand_mentions_table(
        targets,
        runs,
        citations_df,
        brand_aliases=brand_aliases,
    )
    runs_df = _runs_df(runs, targets, run_mentions)
    fanouts_df = _fanouts_df(runs, targets)
    ai_answers_df = _ai_answers_df(runs_df)
    query_metrics_df = _query_metrics(targets, runs, citations_df, run_mentions, target_domain)
    domain_metrics_df = _domain_metrics(citations_df, len(targets))
    url_metrics_df = _url_metrics(citations_df, len(targets))
    overview_df = _overview(
        targets,
        runs_df,
        citations_df,
        fanouts_df,
        query_metrics_df,
        domain_metrics_df,
        brand_mentions_df,
        unresolved_redirects_df,
        target_domain,
    )
    return {
        "overview": overview_df,
        "run_summary": overview_df,
        "runs": runs_df,
        "raw_runs": runs_df,
        "ai_answers": ai_answers_df,
        "fanouts": fanouts_df,
        "citations": citations_df,
        "all_citations": all_citations_df,
        "unresolved_redirects": unresolved_redirects_df,
        "query_metrics": query_metrics_df,
        "domain_metrics": domain_metrics_df,
        "url_metrics": url_metrics_df,
        "brand_mentions": brand_mentions_df,
        "brand_run_mentions": brand_run_mentions_df,
        "run_errors": runs_df.loc[runs_df["error"].astype(str) != "", ["query", "run_index", "provider", "model", "error"]],
    }


CITATION_COLUMNS = [
    "query",
    "run_index",
    "target_domain",
    "cited_url",
    "cited_domain",
    "title",
    "original_url",
    "resolved_url",
    "resolution_status",
    "resolution_method",
    "final_status_code",
    "final_domain",
    "resolution_error",
    "source_label",
    "fallback_domain",
    "citation_kind",
    "source_type",
    "citation_position",
    "is_target_domain",
    "is_exact_target_url",
    "is_acceptable_target_url",
]


def _citation_rows(runs: list[QueryRun]) -> list[dict]:
    rows: list[dict] = []
    for run in runs:
        for citation in run.citations:
            rows.append(
                {
                    "query": citation.query,
                    "run_index": citation.run_index,
                    "target_domain": citation.target_domain,
                    "cited_url": citation.cited_url,
                    "cited_domain": citation.cited_domain,
                    "title": citation.title,
                    "original_url": citation.original_url,
                    "resolved_url": citation.resolved_url,
                    "resolution_status": citation.resolution_status,
                    "resolution_method": citation.resolution_method,
                    "final_status_code": citation.final_status_code,
                    "final_domain": citation.final_domain,
                    "resolution_error": citation.resolution_error,
                    "source_label": citation.source_label,
                    "fallback_domain": citation.fallback_domain,
                    "citation_kind": citation.citation_kind,
                    "source_type": citation.source_type,
                    "citation_position": citation.citation_position,
                    "is_target_domain": citation.is_target_domain,
                    "is_exact_target_url": citation.is_exact_target_url,
                    "is_acceptable_target_url": citation.is_acceptable_target_url,
                }
            )
    return rows


def _valid_citations_df(citations_df: pd.DataFrame) -> pd.DataFrame:
    if citations_df.empty:
        return pd.DataFrame(columns=CITATION_COLUMNS)
    valid = citations_df[
        (
            (citations_df["cited_url"].astype(str) != "")
            | (citations_df["citation_kind"].astype(str) == "domain_fallback")
        )
        & (citations_df["citation_kind"].astype(str) != "unresolved")
        & (citations_df["cited_domain"].astype(str) != "vertexaisearch.cloud.google.com")
    ].copy()
    return valid.reset_index(drop=True)


def _unresolved_redirects_df(citations_df: pd.DataFrame) -> pd.DataFrame:
    if citations_df.empty:
        return pd.DataFrame(columns=CITATION_COLUMNS)
    unresolved = citations_df[
        (
            citations_df["resolution_status"].astype(str).isin(["unresolved_redirect", "domain_fallback"])
            | citations_df["citation_kind"].astype(str).isin(["unresolved", "domain_fallback"])
        )
        & (citations_df["original_url"].astype(str) != "")
    ].copy()
    return unresolved.reset_index(drop=True)


def _runs_df(
    runs: list[QueryRun],
    targets: list[QueryTarget],
    run_mentions: dict[tuple[str, int], list[str]],
) -> pd.DataFrame:
    target_by_query = {target.query: target for target in targets}
    rows = []
    for run in runs:
        target = target_by_query[run.query]
        rows.append(
            {
                "query": run.query,
                "query_cluster": target.query_cluster,
                "intent": target.intent,
                "target_domain": target.target_domain,
                "target_url": target.target_url,
                "acceptable_url_pattern": target.acceptable_url_pattern,
                "run_index": run.run_index,
                "provider": run.provider,
                "model": run.model,
                "response_text": run.response_text,
                "citation_count": len(_valid_run_citations(run)),
                "fanout_query_count": len(run.fanout_queries),
                "mentioned_brands": ", ".join(run_mentions.get((run.query, run.run_index), [])),
                "error": run.error,
                "grounding_metadata": _json_dump(run.grounding_metadata),
            }
        )
    return pd.DataFrame(rows, columns=RUN_COLUMNS)


def _valid_run_citations(run: QueryRun) -> list[Citation]:
    return [
        citation
        for citation in run.citations
        if (citation.cited_url or citation.citation_kind == "domain_fallback")
        and citation.citation_kind != "unresolved"
        and citation.cited_domain != "vertexaisearch.cloud.google.com"
    ]


RUN_COLUMNS = [
    "query",
    "query_cluster",
    "intent",
    "target_domain",
    "target_url",
    "acceptable_url_pattern",
    "run_index",
    "provider",
    "model",
    "response_text",
    "citation_count",
    "fanout_query_count",
    "mentioned_brands",
    "error",
    "grounding_metadata",
]


def _fanouts_df(runs: list[QueryRun], targets: list[QueryTarget]) -> pd.DataFrame:
    target_by_query = {target.query: target for target in targets}
    rows = []
    for run in runs:
        target = target_by_query[run.query]
        for index, fanout_query in enumerate(run.fanout_queries, start=1):
            rows.append(
                {
                    "query": run.query,
                    "query_cluster": target.query_cluster,
                    "target_domain": target.target_domain,
                    "run_index": run.run_index,
                    "provider": run.provider,
                    "model": run.model,
                    "fanout_position": index,
                    "fanout_query": fanout_query,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "query",
            "query_cluster",
            "target_domain",
            "run_index",
            "provider",
            "model",
            "fanout_position",
            "fanout_query",
        ],
    )


def _ai_answers_df(runs_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "query",
        "query_cluster",
        "run_index",
        "provider",
        "model",
        "target_domain",
        "citation_count",
        "fanout_query_count",
        "mentioned_brands",
        "response_text",
        "error",
    ]
    return runs_df[[column for column in columns if column in runs_df.columns]].copy()


def _query_metrics(
    targets: list[QueryTarget],
    runs: list[QueryRun],
    citations_df: pd.DataFrame,
    run_mentions: dict[tuple[str, int], list[str]],
    fallback_target_domain: str,
) -> pd.DataFrame:
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
        target_mention_flags = _target_mention_flags(query_runs, target, fallback_target_domain)
        competitor = _top_competitor(query_citations)
        citation_total = int(len(query_citations)) if not query_citations.empty else 0
        target_citation_total = int(query_citations["is_target_domain"].sum()) if not query_citations.empty else 0
        acceptable_seen = _presence_rate(acceptable_positions, run_count)
        domain_seen = _presence_rate(domain_positions, run_count)
        exact_seen = _presence_rate(exact_positions, run_count)
        inline_positions = _inline_citation_positions(query_runs)
        inline_seen = _presence_rate(inline_positions, run_count)
        mention_rate = _flag_rate(target_mention_flags)
        target_mismatch = 1 if domain_seen > 0 and target.target_url and acceptable_seen == 0 else 0
        competitor_pressure = min(35.0, competitor["citations"] * 4.0)
        visibility_gap = (1.0 - domain_seen / 100.0) * 45.0
        priority_weight = float(target.priority) * 4.0
        mismatch_penalty = 15.0 if target_mismatch else 0.0
        opportunity_score = round(min(100.0, visibility_gap + competitor_pressure + priority_weight + mismatch_penalty), 1)
        top_brands = _top_brands_for_query(target.query, query_runs, run_mentions)

        rows.append(
            {
                "query": target.query,
                "query_cluster": target.query_cluster,
                "intent": target.intent,
                "priority": target.priority,
                "target_domain": target.target_domain or fallback_target_domain,
                "target_url": target.target_url,
                "acceptable_url_pattern": target.acceptable_url_pattern,
                "target_brand_aliases": target.target_brand_aliases,
                "runs_count": run_count,
                "total_citations": citation_total,
                "unique_cited_urls": _nonempty_nunique(query_citations, "cited_url"),
                "fanout_query_count": int(sum(len(run.fanout_queries) for run in query_runs)),
                "target_domain_presence_rate": round(domain_seen, 1),
                "exact_target_url_presence_rate": round(exact_seen, 1),
                "acceptable_target_presence_rate": round(acceptable_seen, 1),
                "target_mention_rate": round(mention_rate, 1),
                "target_mention_answer_count": int(sum(target_mention_flags)),
                "domain_citation_rate": round(domain_seen, 1),
                "inline_citation_rate": round(inline_seen, 1),
                "exact_url_citation_rate": round(exact_seen, 1),
                "approved_page_citation_rate": round(acceptable_seen, 1),
                "best_exact_position": _best_position(exact_positions),
                "best_acceptable_position": _best_position(acceptable_positions),
                "best_domain_position": _best_position(domain_positions),
                "exact_rrf_score": _rrf_score(exact_positions, run_count),
                "acceptable_rrf_score": _rrf_score(acceptable_positions, run_count),
                "domain_rrf_score": _rrf_score(domain_positions, run_count),
                "exact_mrr": _rrf_score(exact_positions, run_count),
                "acceptable_mrr": _rrf_score(acceptable_positions, run_count),
                "domain_mrr": _rrf_score(domain_positions, run_count),
                "citation_share": round((target_citation_total / citation_total * 100.0) if citation_total else 0.0, 1),
                "top_competitor_domain": competitor["domain"],
                "top_competitor_url": competitor["url"],
                "top_competitor_citations": competitor["citations"],
                "top_brands": top_brands,
                "target_mismatch": bool(target_mismatch),
                "opportunity_score": opportunity_score,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("opportunity_score", ascending=False).reset_index(drop=True)


def _first_positions(runs: list[QueryRun], attr: str) -> list[int | None]:
    positions: list[int | None] = []
    for run in runs:
        matching = [
            citation.citation_position
            for citation in _valid_run_citations(run)
            if getattr(citation, attr)
        ]
        positions.append(min(matching) if matching else None)
    return positions


def _inline_citation_positions(runs: list[QueryRun]) -> list[int | None]:
    positions: list[int | None] = []
    for run in runs:
        matching = [
            citation.citation_position
            for citation in _valid_run_citations(run)
            if citation.is_target_domain and citation.source_type == "inline_cited"
        ]
        positions.append(min(matching) if matching else None)
    return positions


def _presence_rate(positions: list[int | None], run_count: int) -> float:
    if run_count == 0:
        return 0.0
    return sum(position is not None for position in positions) / run_count * 100.0


def _flag_rate(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return sum(bool(flag) for flag in flags) / len(flags) * 100.0


def _best_position(positions: list[int | None]) -> int | None:
    seen = [position for position in positions if position is not None]
    return min(seen) if seen else None


def _rrf_score(positions: list[int | None], run_count: int, k: int = 60) -> float:
    if run_count == 0:
        return 0.0
    raw = sum((1.0 / (k + position)) if position else 0.0 for position in positions)
    best_possible = run_count * (1.0 / (k + 1))
    return round((raw / best_possible) * 100.0 if best_possible else 0.0, 1)


def _target_mention_flags(runs: list[QueryRun], target: QueryTarget, fallback_target_domain: str) -> list[bool]:
    aliases = _target_mention_aliases(target, fallback_target_domain)
    return [_text_has_alias(run.response_text, aliases) for run in runs]


def _target_mention_aliases(target: QueryTarget, fallback_target_domain: str) -> set[str]:
    aliases = set(split_aliases(target.target_brand_aliases))
    active_domain = target.target_domain or fallback_target_domain
    for value in [active_domain, target.target_url]:
        brand = brand_from_domain(value)
        if brand:
            aliases.add(brand)
    return {canonical_brand(alias) for alias in aliases if canonical_brand(alias)}


def _text_has_alias(text: str, aliases: set[str]) -> bool:
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    if not aliases or not tokens:
        return False
    max_n = max(len(alias.split()) for alias in aliases)
    token_ngrams: set[str] = set()
    compact_ngrams: set[str] = set()
    for n in range(1, max_n + 1):
        for start in range(0, max(0, len(tokens) - n + 1)):
            chunk = tokens[start : start + n]
            token_ngrams.add(" ".join(chunk))
            compact_ngrams.add("".join(chunk))
    return any(alias in token_ngrams or alias.replace(" ", "") in compact_ngrams for alias in aliases)


def _top_brands_for_query(
    query: str,
    query_runs: list[QueryRun],
    run_mentions: dict[tuple[str, int], list[str]],
) -> str:
    counter: Counter[str] = Counter()
    for run in query_runs:
        counter.update(run_mentions.get((query, run.run_index), []))
    return ", ".join(f"{brand} ({count})" for brand, count in counter.most_common(5))


def _top_competitor(query_citations: pd.DataFrame) -> dict:
    if query_citations.empty:
        return {"domain": "-", "url": "-", "citations": 0}
    competitors = query_citations[~query_citations["is_target_domain"]]
    if competitors.empty:
        return {"domain": "-", "url": "-", "citations": 0}
    top_domain = competitors["cited_domain"].value_counts().index[0]
    top_domain_rows = competitors[competitors["cited_domain"] == top_domain]
    cited_urls = [url for url in top_domain_rows["cited_url"].astype(str).tolist() if url]
    top_url = Counter(cited_urls).most_common(1)[0][0] if cited_urls else "-"
    return {
        "domain": str(top_domain),
        "url": str(top_url),
        "citations": int(len(top_domain_rows)),
    }


def _domain_metrics(citations_df: pd.DataFrame, total_queries: int) -> pd.DataFrame:
    if citations_df.empty:
        return pd.DataFrame(columns=DOMAIN_COLUMNS)
    rows = []
    for domain, group in citations_df.groupby("cited_domain"):
        query_count = int(group["query"].nunique())
        rows.append(
            {
                "cited_domain": domain,
                "category": "Target" if bool(group["is_target_domain"].any()) else "Other",
                "total_citations": int(len(group)),
                "queries_count": query_count,
                "query_coverage": f"{query_count}/{total_queries}",
                "query_coverage_rate": round((query_count / total_queries * 100.0) if total_queries else 0.0, 1),
                "citation_kinds": " | ".join(sorted(group["citation_kind"].dropna().astype(str).unique())),
                "queries_list": " | ".join(sorted(group["query"].unique())),
            }
        )
    return pd.DataFrame(rows).sort_values(["total_citations", "queries_count"], ascending=False).reset_index(drop=True)


DOMAIN_COLUMNS = [
    "cited_domain",
    "category",
    "total_citations",
    "queries_count",
    "query_coverage",
    "query_coverage_rate",
    "citation_kinds",
    "queries_list",
]


def _url_metrics(citations_df: pd.DataFrame, total_queries: int) -> pd.DataFrame:
    if not citations_df.empty:
        citations_df = citations_df[citations_df["cited_url"].astype(str) != ""].copy()
    if citations_df.empty:
        return pd.DataFrame(columns=URL_COLUMNS)
    rows = []
    for url, group in citations_df.groupby("cited_url"):
        query_count = int(group["query"].nunique())
        rows.append(
            {
                "cited_url": url,
                "cited_domain": group.iloc[0]["cited_domain"],
                "title": group.iloc[0].get("title", ""),
                "category": "Target" if bool(group["is_target_domain"].any()) else "Other",
                "total_citations": int(len(group)),
                "queries_count": query_count,
                "query_coverage": f"{query_count}/{total_queries}",
                "query_coverage_rate": round((query_count / total_queries * 100.0) if total_queries else 0.0, 1),
                "citation_kinds": " | ".join(sorted(group["citation_kind"].dropna().astype(str).unique())),
                "queries_list": " | ".join(sorted(group["query"].unique())),
            }
        )
    return pd.DataFrame(rows).sort_values(["total_citations", "queries_count"], ascending=False).reset_index(drop=True)


URL_COLUMNS = [
    "cited_url",
    "cited_domain",
    "title",
    "category",
    "total_citations",
    "queries_count",
    "query_coverage",
    "query_coverage_rate",
    "citation_kinds",
    "queries_list",
]


def _overview(
    targets: list[QueryTarget],
    runs_df: pd.DataFrame,
    citations_df: pd.DataFrame,
    fanouts_df: pd.DataFrame,
    query_metrics_df: pd.DataFrame,
    domain_metrics_df: pd.DataFrame,
    brand_mentions_df: pd.DataFrame,
    unresolved_redirects_df: pd.DataFrame,
    fallback_target_domain: str,
) -> pd.DataFrame:
    total_queries = len(targets)
    total_answers = int(len(runs_df))
    successful_answers = int((runs_df["error"].astype(str) == "").sum()) if not runs_df.empty else 0
    top_competitor = "-"
    competitors = domain_metrics_df[domain_metrics_df["category"] != "Target"] if not domain_metrics_df.empty else pd.DataFrame()
    if not competitors.empty:
        top_competitor = str(competitors.iloc[0]["cited_domain"])
    top_brand = "-"
    if not brand_mentions_df.empty:
        top_brand = str(brand_mentions_df.iloc[0]["brand"])
    target_domains = sorted({target.target_domain or fallback_target_domain for target in targets if target.target_domain or fallback_target_domain})
    target_domain_label = target_domains[0] if len(target_domains) == 1 else "mixed"

    target_citation_answer_count = 0
    if not citations_df.empty:
        target_citation_answer_count = int(citations_df[citations_df["is_target_domain"]].drop_duplicates(["query", "run_index"]).shape[0])
    target_citation_query_count = int((query_metrics_df["domain_citation_rate"] > 0).sum()) if not query_metrics_df.empty else 0
    target_mention_answer_count = int(query_metrics_df["target_mention_answer_count"].sum()) if not query_metrics_df.empty else 0
    target_mention_query_count = int((query_metrics_df["target_mention_rate"] > 0).sum()) if not query_metrics_df.empty else 0

    values = {
        "target_domain": target_domain_label,
        "queries": total_queries,
        "total_queries": total_queries,
        "answers": total_answers,
        "total_answers": total_answers,
        "successful_answers": successful_answers,
        "run_errors": total_answers - successful_answers,
        "total_fanout_queries": int(len(fanouts_df)),
        "total_cited_urls": _nonempty_count(citations_df, "cited_url"),
        "unique_cited_urls": _nonempty_nunique(citations_df, "cited_url"),
        "unique_cited_domains": int(citations_df["cited_domain"].nunique()) if not citations_df.empty else 0,
        "unresolved_redirects": _unresolved_count(unresolved_redirects_df),
        "domain_fallback_citations": _domain_fallback_count(citations_df),
        "target_citation_query_coverage": f"{target_citation_query_count}/{total_queries}",
        "target_citation_query_coverage_rate": round((target_citation_query_count / total_queries * 100.0) if total_queries else 0.0, 1),
        "target_citation_answer_frequency": f"{target_citation_answer_count}/{total_answers}",
        "target_citation_answer_frequency_rate": round((target_citation_answer_count / total_answers * 100.0) if total_answers else 0.0, 1),
        "target_mention_query_coverage": f"{target_mention_query_count}/{total_queries}",
        "target_mention_query_coverage_rate": round((target_mention_query_count / total_queries * 100.0) if total_queries else 0.0, 1),
        "target_mention_answer_frequency": f"{target_mention_answer_count}/{total_answers}",
        "target_mention_answer_frequency_rate": round((target_mention_answer_count / total_answers * 100.0) if total_answers else 0.0, 1),
        "avg_opportunity_score": round(float(query_metrics_df["opportunity_score"].mean()), 1) if not query_metrics_df.empty else 0.0,
        "avg_acceptable_presence_rate": round(float(query_metrics_df["acceptable_target_presence_rate"].mean()), 1) if not query_metrics_df.empty else 0.0,
        "avg_domain_presence_rate": round(float(query_metrics_df["target_domain_presence_rate"].mean()), 1) if not query_metrics_df.empty else 0.0,
        "avg_acceptable_mrr": round(float(query_metrics_df["acceptable_rrf_score"].mean()), 1) if not query_metrics_df.empty else 0.0,
        "avg_acceptable_rrf_score": round(float(query_metrics_df["acceptable_rrf_score"].mean()), 1) if not query_metrics_df.empty else 0.0,
        "avg_domain_rrf_score": round(float(query_metrics_df["domain_rrf_score"].mean()), 1) if not query_metrics_df.empty else 0.0,
        "avg_target_mention_rate": round(float(query_metrics_df["target_mention_rate"].mean()), 1) if not query_metrics_df.empty else 0.0,
        "avg_approved_page_citation_rate": round(float(query_metrics_df["approved_page_citation_rate"].mean()), 1) if not query_metrics_df.empty else 0.0,
        "zero_visibility_queries": int((query_metrics_df["domain_citation_rate"] == 0).sum()) if not query_metrics_df.empty else 0,
        "target_mismatch_queries": int(query_metrics_df["target_mismatch"].sum()) if not query_metrics_df.empty else 0,
        "top_competitor_domain": top_competitor,
        "top_brand": top_brand,
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


def _json_dump(value: dict) -> str:
    if not value:
        return ""
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _nonempty_count(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int((df[column].astype(str) != "").sum())


def _nonempty_nunique(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(df.loc[df[column].astype(str) != "", column].nunique())


def _domain_fallback_count(df: pd.DataFrame) -> int:
    if df.empty or "citation_kind" not in df.columns:
        return 0
    return int((df["citation_kind"].astype(str) == "domain_fallback").sum())


def _unresolved_count(df: pd.DataFrame) -> int:
    if df.empty or "citation_kind" not in df.columns:
        return 0
    return int((df["citation_kind"].astype(str) == "unresolved").sum())
