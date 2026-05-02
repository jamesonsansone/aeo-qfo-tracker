from seo_suite.aggregator import build_query_runs, build_tables
from seo_suite.content_alignment import build_content_alignment, chunk_html_sections
from seo_suite.models import ProviderResult, QueryTarget
from seo_suite.page_diagnostics import build_page_diagnostics, build_weekly_queue, collect_diagnostic_urls


def test_chunk_html_sections_preserves_exact_text():
    html = "<html><body><h1>Product Name</h1><p>Exact PDP description with waterproof membrane.</p></body></html>"
    chunks = chunk_html_sections(html)
    combined = " ".join(chunk["text"] for chunk in chunks)
    assert "Exact PDP description with waterproof membrane." in combined


def test_content_alignment_scores_target_and_competitor_from_fixture_dom():
    targets = [
        QueryTarget(
            query="lightweight backpacking tent for two",
            query_cluster="backpacking tents",
            intent="commercial comparison",
            target_url="https://trailgear.example/products/cloudpeak-2p-tent",
            acceptable_url_pattern="https://trailgear.example/products/*tent*",
            priority=4,
        )
    ]
    results = [
        ProviderResult(
            query="lightweight backpacking tent for two",
            run_index=0,
            provider="fixture",
            model="fixture",
            response_text="Weight packed size vestibule storm protection and livability matter for two-person backpacking tents.",
            sources=[
                {
                    "url": "https://www.outdoorgearlab.com/topics/camping-and-hiking/best-backpacking-tent",
                    "title": "Best Backpacking Tents",
                }
            ],
        )
    ]
    runs = build_query_runs(results, targets, "trailgear.example")
    tables = build_tables(targets, runs, "trailgear.example")
    tables["page_diagnostics"] = build_page_diagnostics(
        collect_diagnostic_urls(targets, tables),
        fixture_path="data/fixture_pages.jsonl",
    )
    tables["weekly_queue"] = build_weekly_queue(targets, tables)
    alignment = build_content_alignment(targets, tables, fixture_path="data/fixture_pages.jsonl")

    assert not alignment.empty
    assert set(alignment["url_role"]) == {"Mapped Target URL", "Cited Competitor URL"}
    competitor = alignment[alignment["url_role"] == "Cited Competitor URL"].iloc[0]
    assert competitor["tfidf_score"] > 0
    assert competitor["source_text_mode"] == "rendered_dom_fixture"

