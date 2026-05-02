from seo_suite.aggregator import build_query_runs, build_tables
from seo_suite.models import ProviderResult, QueryTarget
from seo_suite.page_diagnostics import (
    analyze_page,
    build_page_diagnostics,
    build_weekly_queue,
    collect_diagnostic_urls,
    fetch_pagespeed_metrics,
)
import seo_suite.page_diagnostics as page_diagnostics


def test_raw_rendered_schema_detection_from_fixture_html():
    raw_html = """
    <html><head><title>PDP</title></head><body>
      <h1>Product Page</h1>
      <p>Short copy.</p>
    </body></html>
    """
    rendered_html = """
    <html><head><title>PDP</title>
      <script type="application/ld+json">
        {"@context":"https://schema.org","@type":"Product","offers":{"@type":"Offer","price":"99"}}
      </script>
    </head><body>
      <h1>Product Page</h1>
      <p>Short copy with expanded rendered product details, sizing, shipping, availability, and returns.</p>
    </body></html>
    """

    row = analyze_page("https://example.com/products/a", raw_html, rendered_html)

    assert row["product_schema_raw"] is False
    assert row["product_schema_rendered"] is True
    assert row["offer_schema_raw"] is False
    assert row["offer_schema_rendered"] is True
    assert "Product schema missing in raw HTML" in row["diagnostic_flags"]
    assert "Offer schema missing in raw HTML" in row["diagnostic_flags"]


def test_pagespeed_skips_without_api_key():
    assert fetch_pagespeed_metrics("https://example.com", api_key="") == {}


def test_page_diagnostics_join_targets_and_citations():
    targets = [
        QueryTarget(
            query="best product",
            query_cluster="cluster",
            intent="commercial",
            target_url="https://example.com/products/a",
            acceptable_url_pattern="https://example.com/products/*",
            priority=5,
        )
    ]
    results = [
        ProviderResult(
            query="best product",
            run_index=0,
            provider="fixture",
            model="fixture",
            response_text="",
            sources=[{"url": "https://competitor.com/best", "title": "Best"}],
        )
    ]

    runs = build_query_runs(results, targets, "example.com")
    tables = build_tables(targets, runs, "example.com")
    urls = collect_diagnostic_urls(targets, tables)
    tables["page_diagnostics"] = build_page_diagnostics(urls)
    tables["weekly_queue"] = build_weekly_queue(targets, tables)

    assert "https://example.com/products/a" in urls
    assert "https://competitor.com/best" in urls
    assert not tables["page_diagnostics"].empty
    assert tables["weekly_queue"].iloc[0]["recommended_follow_up"] == "Target page not cited"


def test_live_diagnostics_cache_reuses_rendered_result(tmp_path, monkeypatch):
    calls = {"raw": 0, "rendered": 0}

    def fake_raw(url):
        calls["raw"] += 1
        return "<html><body><h1>Raw</h1><p>Raw words here.</p></body></html>", "http_200"

    def fake_rendered(url):
        calls["rendered"] += 1
        return (
            '<html><head><script type="application/ld+json">'
            '{"@type":"Product","offers":{"@type":"Offer"}}'
            '</script></head><body><h1>Rendered</h1><p>Rendered words here.</p></body></html>',
            "rendered_with_playwright",
        )

    monkeypatch.setattr(page_diagnostics, "_fetch_raw_html", fake_raw)
    monkeypatch.setattr(page_diagnostics, "_render_html_playwright", fake_rendered)

    first = build_page_diagnostics(
        ["https://example.com/products/a"],
        fetch_live=True,
        render_live=True,
        cache_dir=tmp_path,
    )
    second = build_page_diagnostics(
        ["https://example.com/products/a"],
        fetch_live=True,
        render_live=True,
        cache_dir=tmp_path,
    )

    assert calls == {"raw": 1, "rendered": 1}
    assert bool(first.iloc[0]["product_schema_rendered"]) is True
    assert bool(second.iloc[0]["product_schema_rendered"]) is True
