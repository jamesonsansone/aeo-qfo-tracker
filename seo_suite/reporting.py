"""Markdown brief generation."""

from __future__ import annotations

import pandas as pd


def generate_brief(tables: dict[str, pd.DataFrame], target_domain: str) -> str:
    overview = tables["overview"].iloc[0].to_dict() if not tables["overview"].empty else {}
    query_metrics = tables["query_metrics"]
    domain_metrics = tables["domain_metrics"]
    url_metrics = tables["url_metrics"]
    weekly_queue = tables.get("weekly_queue", pd.DataFrame())

    top_opportunities = query_metrics.head(5)
    zero_visibility = query_metrics[query_metrics["acceptable_target_presence_rate"] == 0].head(8)
    competitors = domain_metrics[domain_metrics["category"] != "Target"].head(5) if not domain_metrics.empty else pd.DataFrame()
    competitor_urls = url_metrics[url_metrics["category"] != "Target"].head(5) if not url_metrics.empty else pd.DataFrame()

    lines = [
        f"# AI Search Opportunity Brief: {target_domain}",
        "",
        "## Executive Summary",
        (
            f"The query set contains {overview.get('queries', 0)} mapped queries. "
            f"The target domain appeared for {overview.get('avg_domain_presence_rate', 0)}% of runs on average, "
            f"while approved target pages appeared for {overview.get('avg_acceptable_presence_rate', 0)}%."
        ),
        (
            f"There are {overview.get('zero_visibility_queries', 0)} queries with no approved target page visibility "
            f"and {overview.get('target_mismatch_queries', 0)} queries where the domain appeared but the intended page did not."
        ),
        "",
        "## Highest Priority Opportunities",
    ]

    if top_opportunities.empty:
        lines.append("No query metrics were available.")
    else:
        for row in top_opportunities.to_dict("records"):
            lines.append(
                f"- {row['query']} ({row['query_cluster']}): opportunity {row['opportunity_score']}, "
                f"approved page visibility {row['acceptable_target_presence_rate']}%, "
                f"top competitor {row['top_competitor_domain']}."
            )

    lines.extend(["", "## Weekly Follow-Up Queue"])
    if weekly_queue.empty:
        lines.append("No weekly follow-up items were generated.")
    else:
        for row in weekly_queue.head(6).to_dict("records"):
            lines.append(
                f"- {row['recommended_follow_up']}: {row['query']} "
                f"(priority {row['weekly_priority_score']}, target {row['target_url']})."
            )

    lines.extend(["", "## Zero-Visibility Queries"])
    if zero_visibility.empty:
        lines.append("No zero-visibility queries were found for approved target pages.")
    else:
        for row in zero_visibility.to_dict("records"):
            lines.append(f"- {row['query']} -> target {row['target_url']}")

    lines.extend(["", "## Competitor Patterns"])
    if competitors.empty:
        lines.append("No recurring non-target competitor domains were found.")
    else:
        for row in competitors.to_dict("records"):
            lines.append(
                f"- {row['cited_domain']}: {row['total_citations']} citations across "
                f"{row['queries_count']} queries."
            )

    lines.extend(["", "## URLs To Review"])
    if competitor_urls.empty:
        lines.append("No non-target URLs were available for review.")
    else:
        for row in competitor_urls.to_dict("records"):
            lines.append(f"- {row['cited_url']} ({row['total_citations']} citations)")

    lines.extend(
        [
            "",
            "## Recommended Next Actions",
            "- Review the highest-opportunity target pages for answer clarity, product facts, and comparison language.",
            "- Compare the most-cited competitor pages against target pages before recommending content changes.",
            "- Treat these scores as probabilistic AI-search visibility signals, not traditional rank tracking.",
            "- In phase 2, add raw HTML vs rendered DOM checks and Product schema extraction for priority pages.",
        ]
    )
    return "\n".join(lines) + "\n"
