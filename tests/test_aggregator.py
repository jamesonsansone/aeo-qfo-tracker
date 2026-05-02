from seo_suite.aggregator import build_query_runs, build_tables
from seo_suite.models import ProviderResult, QueryTarget


def test_query_metrics_measure_exact_acceptable_domain_and_mismatch():
    targets = [
        QueryTarget(
            query="target exact",
            query_cluster="cluster",
            intent="commercial",
            target_url="https://example.com/products/a",
            acceptable_url_pattern="https://example.com/products/*",
            priority=5,
        ),
        QueryTarget(
            query="domain only",
            query_cluster="cluster",
            intent="commercial",
            target_url="https://example.com/products/b",
            acceptable_url_pattern="https://example.com/products/*",
            priority=5,
        ),
        QueryTarget(
            query="absent",
            query_cluster="cluster",
            intent="commercial",
            target_url="https://example.com/products/c",
            acceptable_url_pattern="https://example.com/products/*",
            priority=5,
        ),
    ]
    results = [
        ProviderResult(
            query="target exact",
            run_index=0,
            provider="fixture",
            model="fixture",
            response_text="",
            sources=[{"url": "https://example.com/products/a", "title": "A"}],
        ),
        ProviderResult(
            query="domain only",
            run_index=0,
            provider="fixture",
            model="fixture",
            response_text="",
            sources=[{"url": "https://example.com/blog/buying-guide", "title": "Guide"}],
        ),
        ProviderResult(
            query="absent",
            run_index=0,
            provider="fixture",
            model="fixture",
            response_text="",
            sources=[{"url": "https://competitor.com/best", "title": "Best"}],
        ),
    ]

    runs = build_query_runs(results, targets, "example.com")
    tables = build_tables(targets, runs, "example.com")
    metrics = tables["query_metrics"].set_index("query")

    assert metrics.loc["target exact", "exact_target_url_presence_rate"] == 100.0
    assert metrics.loc["target exact", "acceptable_target_presence_rate"] == 100.0
    assert metrics.loc["target exact", "approved_page_citation_rate"] == 100.0
    assert metrics.loc["domain only", "target_domain_presence_rate"] == 100.0
    assert metrics.loc["domain only", "domain_citation_rate"] == 100.0
    assert metrics.loc["domain only", "acceptable_target_presence_rate"] == 0.0
    assert bool(metrics.loc["domain only", "target_mismatch"]) is True
    assert metrics.loc["absent", "target_domain_presence_rate"] == 0.0
    assert metrics.loc["absent", "top_competitor_domain"] == "competitor.com"


def test_build_tables_preserves_citation_columns_when_no_sources():
    target = QueryTarget(
        query="missing citations",
        query_cluster="cluster",
        intent="test",
        target_url="https://example.com/products/a",
        acceptable_url_pattern="https://example.com/products/*",
        priority=3,
    )
    results = [
        ProviderResult(
            query=target.query,
            run_index=0,
            provider="fixture",
            model="fixture",
            response_text="No sources returned.",
            sources=[],
        )
    ]

    runs = build_query_runs(results, [target], "example.com")
    tables = build_tables([target], runs, "example.com")

    assert tables["citations"].empty
    assert "query" in tables["citations"].columns
    assert tables["query_metrics"].iloc[0]["approved_page_citation_rate"] == 0
