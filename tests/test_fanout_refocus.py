import pandas as pd
import pytest
from datetime import datetime
from urllib.error import HTTPError, URLError

from app import _runtime_diagnostics, _save_zip_to_desktop, _zip_outputs
from seo_suite import providers
from seo_suite.aggregator import build_query_runs, build_tables
from seo_suite.brand_mentions import brand_from_domain
from seo_suite.io import query_targets_from_dataframe
from seo_suite.models import ProviderResult, QueryTarget
from seo_suite.providers import (
    _grounded_prompt,
    _extract_gemini_fanout_queries,
    _extract_gemini_sources,
    _extract_openai_fanout_queries,
    _extract_openai_sources,
    _source_from_url,
    _resolve_source_url,
    OpenAIWebSearchProvider,
)


def test_csv_targets_use_global_defaults_and_row_overrides():
    df = pd.DataFrame(
        [
            {"query": "best running shoes"},
            {
                "query": "best carbon plated shoes",
                "target_domain": "nike.com",
                "target_url_pattern": "https://nike.com/running/*",
                "target_brand_aliases": "Nike",
            },
        ]
    )

    targets = query_targets_from_dataframe(
        df,
        default_target_domain="puma.com",
        default_target_url="https://puma.com/running",
        default_acceptable_url_pattern="https://puma.com/running/*",
        default_brand_aliases="Puma",
    )

    assert targets[0].target_domain == "puma.com"
    assert targets[0].target_url == "https://puma.com/running"
    assert targets[0].acceptable_url_pattern == "https://puma.com/running/*"
    assert targets[0].target_brand_aliases == "Puma"
    assert targets[1].target_domain == "nike.com"
    assert targets[1].acceptable_url_pattern == "https://nike.com/running/*"
    assert targets[1].target_brand_aliases == "Nike"


def test_csv_targets_accept_spreadsheet_friendly_column_names():
    df = pd.DataFrame(
        [
            {
                "Query": "best soccer cleats for 2026",
                "Type": "Best",
                "Strategic Category": "Soccer / Football",
            },
            {
                "Query": "Nike Mercurial vs adidas F50 vs Puma Ultra",
                "Type": "Comparison",
                "Strategic Category": "Soccer / Football",
                "Domain": "us.puma.com",
                "Target URL": "https://us.puma.com/us/en/sports/soccer",
                "URL Pattern": "https://us.puma.com/us/en/sports/soccer/*",
                "Cluster": "Boots",
                "Priority": "5",
            },
        ]
    )

    targets = query_targets_from_dataframe(
        df,
        default_target_domain="puma.com",
        default_target_url="https://puma.com/soccer",
        default_acceptable_url_pattern="https://puma.com/soccer/*",
    )

    assert targets[0].query == "best soccer cleats for 2026"
    assert targets[0].intent == "Best"
    assert targets[0].query_cluster == "Soccer / Football"
    assert targets[0].target_domain == "puma.com"
    assert targets[1].intent == "Comparison"
    assert targets[1].query_cluster == "Soccer / Football"
    assert targets[1].target_domain == "us.puma.com"
    assert targets[1].target_url == "https://us.puma.com/us/en/sports/soccer"
    assert targets[1].acceptable_url_pattern == "https://us.puma.com/us/en/sports/soccer/*"
    assert targets[1].priority == 5


def test_csv_targets_accept_bom_and_query_synonyms():
    df = pd.DataFrame(
        [
            {
                "\ufeffSeed Query": "best firm ground soccer cleats",
                "Query Type": "Best",
                "Strategic Category": "Soccer / Football",
                "Domain": "us.puma.com",
            }
        ]
    )

    targets = query_targets_from_dataframe(df)

    assert len(targets) == 1
    assert targets[0].query == "best firm ground soccer cleats"
    assert targets[0].intent == "Best"
    assert targets[0].query_cluster == "Soccer / Football"
    assert targets[0].target_domain == "us.puma.com"


def test_batch_brand_mentions_report_query_coverage_and_answer_frequency():
    targets = [
        QueryTarget(query="road running shoes", target_domain="puma.com", target_brand_aliases="Puma"),
        QueryTarget(query="trail running shoes", target_domain="puma.com", target_brand_aliases="Puma"),
    ]
    results = [
        ProviderResult(
            query="road running shoes",
            run_index=0,
            provider="fixture",
            model="fixture",
            response_text="Nike and Puma both make strong road running shoes.",
            sources=[{"url": "https://nike.com/running", "title": "Nike"}, {"url": "https://puma.com/running", "title": "Puma"}],
        ),
        ProviderResult(
            query="road running shoes",
            run_index=1,
            provider="fixture",
            model="fixture",
            response_text="Nike is mentioned again for road running.",
            sources=[{"url": "https://nike.com/running", "title": "Nike"}],
        ),
        ProviderResult(
            query="trail running shoes",
            run_index=0,
            provider="fixture",
            model="fixture",
            response_text="Nike appears in this trail running overview.",
            sources=[{"url": "https://nike.com/trail", "title": "Nike"}],
        ),
        ProviderResult(
            query="trail running shoes",
            run_index=1,
            provider="fixture",
            model="fixture",
            response_text="The answer focuses on fit and outsole lugs.",
            sources=[{"url": "https://example.com/trail", "title": "Trail"}],
        ),
    ]

    tables = build_tables(targets, build_query_runs(results, targets, "puma.com"), "puma.com")
    brands = tables["brand_mentions"].set_index("brand")

    assert brands.loc["Nike", "query_coverage"] == "2/2 queries"
    assert brands.loc["Nike", "answer_frequency"] == "3/4 answers"
    assert brands.loc["Puma", "query_coverage"] == "1/2 queries"
    assert brands.loc["Puma", "answer_frequency"] == "1/4 answers"


def test_brand_extraction_detects_running_brands_once_per_answer():
    target = QueryTarget(query="best trail running shoes", target_domain="us.puma.com", target_brand_aliases="puma")
    results = [
        ProviderResult(
            query=target.query,
            run_index=0,
            provider="fixture",
            model="fixture",
            response_text=(
                "La Sportiva and La Sportiva Prodigio Pro are technical picks. "
                "Brooks Cascadia, Saucony Xodus Ultra, HOKA Speedgoat, and Puma also appear."
            ),
            sources=[],
        )
    ]

    tables = build_tables([target], build_query_runs(results, [target], "us.puma.com"), "us.puma.com")
    brands = tables["brand_mentions"].set_index("brand")
    evidence = tables["brand_run_mentions"]

    for brand in ["La Sportiva", "Brooks", "Saucony", "HOKA", "Puma"]:
        assert brands.loc[brand, "answer_frequency_count"] == 1
    assert len(evidence[evidence["brand"] == "La Sportiva"]) == 1


def test_brand_extraction_does_not_count_lowercase_on_as_on_running():
    target = QueryTarget(query="best trail running shoes", target_domain="us.puma.com", target_brand_aliases="puma")
    results = [
        ProviderResult(
            query=target.query,
            run_index=0,
            provider="fixture",
            model="fixture",
            response_text="The shoe works on rocky terrain and on long descents.",
            sources=[{"url": "https://on.com/running", "title": "On"}],
        )
    ]

    tables = build_tables([target], build_query_runs(results, [target], "us.puma.com"), "us.puma.com")
    brands = set(tables["brand_mentions"]["brand"])

    assert "On Running" not in brands


def test_brand_extraction_counts_clear_on_running_evidence():
    target = QueryTarget(query="best running shoes", target_domain="us.puma.com", target_brand_aliases="puma")
    results = [
        ProviderResult(
            query=target.query,
            run_index=0,
            provider="fixture",
            model="fixture",
            response_text="On Running and the On Cloudmonster are frequently mentioned for max cushioning.",
            sources=[],
        )
    ]

    tables = build_tables([target], build_query_runs(results, [target], "us.puma.com"), "us.puma.com")
    brands = tables["brand_mentions"].set_index("brand")

    assert brands.loc["On Running", "answer_frequency"] == "1/1 answers"


def test_uploaded_brand_aliases_extend_brand_extraction():
    target = QueryTarget(query="best trail racing shoes", target_domain="us.puma.com", target_brand_aliases="puma")
    aliases = pd.DataFrame([{"brand": "Norda", "aliases": "Norda, 001A", "category": "trail"}])
    results = [
        ProviderResult(
            query=target.query,
            run_index=0,
            provider="fixture",
            model="fixture",
            response_text="The Norda 001A is often named in premium trail racing roundups.",
            sources=[],
        )
    ]

    tables = build_tables(
        [target],
        build_query_runs(results, [target], "us.puma.com"),
        "us.puma.com",
        brand_aliases=aliases,
    )

    assert tables["brand_mentions"].set_index("brand").loc["Norda", "category"] == "trail"


def test_normalized_rrf_scores_target_domain_positions_across_runs():
    target = QueryTarget(query="best shoes", target_domain="puma.com")
    results = [
        ProviderResult(
            query=target.query,
            run_index=0,
            provider="fixture",
            model="fixture",
            response_text="Puma is cited first.",
            sources=[{"url": "https://puma.com/running", "title": "Puma"}],
        ),
        ProviderResult(
            query=target.query,
            run_index=1,
            provider="fixture",
            model="fixture",
            response_text="No target citation.",
            sources=[{"url": "https://nike.com/running", "title": "Nike"}],
        ),
    ]

    tables = build_tables([target], build_query_runs(results, [target], "puma.com"), "puma.com")
    metrics = tables["query_metrics"].iloc[0]

    assert metrics["domain_citation_rate"] == 50.0
    assert metrics["domain_rrf_score"] == 50.0


def test_domain_only_target_flags_matching_cited_subpath():
    target = QueryTarget(query="puma clothing", target_domain="us.puma.com")
    results = [
        ProviderResult(
            query=target.query,
            run_index=0,
            provider="fixture",
            model="fixture",
            response_text="Puma clothing was cited.",
            sources=[{"url": "https://us.puma.com/us/en/mens/clothing", "title": "Puma Clothing"}],
        )
    ]

    tables = build_tables([target], build_query_runs(results, [target], ""), "")
    citation = tables["citations"].iloc[0]

    assert bool(citation["is_target_domain"]) is True
    assert tables["query_metrics"].iloc[0]["domain_citation_rate"] == 100.0


def test_domain_brand_seed_uses_registrable_domain_not_geo_subdomain():
    assert brand_from_domain("us.puma.com") == "Puma"
    assert brand_from_domain("https://shop.nike.com/running") == "Nike"


def test_provider_prompt_does_not_include_target_domain_or_aliases():
    target = QueryTarget(
        query="which brands have the best running shoe technology?",
        target_domain="us.puma.com",
        target_brand_aliases="Puma",
    )

    prompt = _grounded_prompt(target)

    assert "us.puma.com" not in prompt
    assert "Puma" not in prompt
    assert target.query in prompt


def test_runtime_diagnostics_include_sdk_versions():
    diagnostics = _runtime_diagnostics()

    assert diagnostics["python_executable"]
    assert diagnostics["openai_version"]
    assert diagnostics["httpx_version"]
    assert diagnostics["streamlit_version"]


def test_openai_provider_explains_proxies_package_mismatch(monkeypatch):
    import openai

    class BrokenOpenAI:
        def __init__(self, *args, **kwargs):
            raise TypeError("__init__() got an unexpected keyword argument 'proxies'")

    monkeypatch.setattr(openai, "OpenAI", BrokenOpenAI)

    with pytest.raises(RuntimeError, match="incompatible openai/httpx package"):
        OpenAIWebSearchProvider(api_key="test", model="gpt-5-nano")


def test_openai_parser_extracts_search_queries_and_sources():
    response = {
        "output": [
            {
                "action": {
                    "queries": ["best running shoes 2026", "running shoe reviews"],
                    "sources": [{"url": "https://example.com/a", "title": "A"}],
                }
            },
            {
                "type": "message",
                "content": [
                    {
                        "annotations": [
                            {"type": "url_citation", "url": "https://example.com/b", "title": "B"}
                        ]
                    }
                ],
            },
        ]
    }

    assert _extract_openai_fanout_queries(response) == ["best running shoes 2026", "running shoe reviews"]
    sources = _extract_openai_sources(response)

    assert [(source["url"], source["title"], source["resolution_status"]) for source in sources] == [
        ("https://example.com/a", "A", "direct"),
        ("https://example.com/b", "B", "direct"),
    ]


def test_gemini_parser_extracts_grounding_queries_and_chunks():
    response = {
        "candidates": [
            {
                "groundingMetadata": {
                    "webSearchQueries": ["running shoe reviews"],
                    "groundingChunks": [
                        {"web": {"uri": "https://example.com/review", "title": "Review"}}
                    ],
                }
            }
        ]
    }

    assert _extract_gemini_fanout_queries(response) == ["running shoe reviews"]
    sources = _extract_gemini_sources(response)

    assert [(source["url"], source["title"], source["resolution_status"]) for source in sources] == [
        ("https://example.com/review", "Review", "direct")
    ]


def test_gemini_parser_resolves_encoded_grounding_redirect_urls():
    final_url = "https://us.puma.com/us/en/mens/clothing"
    redirect_url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect?url=https%3A%2F%2Fus.puma.com%2Fus%2Fen%2Fmens%2Fclothing"
    response = {
        "candidates": [
            {
                "grounding_metadata": {
                    "grounding_chunks": [
                        {"web": {"uri": redirect_url, "title": "Puma Clothing"}}
                    ],
                }
            }
        ]
    }

    source = _extract_gemini_sources(response)[0]

    assert source["url"] == final_url
    assert source["original_url"] == redirect_url
    assert source["resolution_method"] == "encoded_param"


def test_resolver_follows_mocked_http_redirect(monkeypatch):
    redirect_url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/http"
    final_url = "https://www.rei.com/product/trail-shoe"

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def geturl(self):
            return final_url

        def read(self, *_args):
            return b""

    monkeypatch.setattr(providers, "_open_url", lambda _request, timeout=4.0: FakeResponse())
    _resolve_source_url.cache_clear()

    resolution = _resolve_source_url(redirect_url)

    assert resolution.resolved_url == final_url
    assert resolution.resolution_method == "http_head"
    assert resolution.final_status_code == 200


def test_resolver_uses_certifi_backed_context(monkeypatch):
    fake_context = object()
    captured = {}

    monkeypatch.setattr(providers.certifi, "where", lambda: "/tmp/fake-certifi.pem")
    monkeypatch.setattr(
        providers.ssl,
        "create_default_context",
        lambda cafile=None: captured.setdefault("context", (cafile, fake_context))[1],
    )
    providers._certifi_ssl_context.cache_clear()

    context = providers._certifi_ssl_context()

    assert context is fake_context
    assert captured["context"][0] == "/tmp/fake-certifi.pem"


def test_resolver_head_failure_falls_back_to_get(monkeypatch):
    redirect_url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/get"
    final_url = "https://www.soccer.com/shop/details"
    calls = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def geturl(self):
            return final_url

        def read(self, *_args):
            return b""

    def fake_open_url(request, timeout=4.0):
        method = request.get_method()
        calls.append(method)
        if method == "HEAD":
            raise HTTPError(redirect_url, 405, "Method Not Allowed", hdrs=None, fp=None)
        return FakeResponse()

    monkeypatch.setattr(providers, "_open_url", fake_open_url)
    _resolve_source_url.cache_clear()

    resolution = _resolve_source_url(redirect_url)

    assert calls == ["HEAD", "GET"]
    assert resolution.resolved_url == final_url
    assert resolution.resolution_method == "http_get"


def test_resolver_extracts_mocked_html_canonical(monkeypatch):
    redirect_url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/html"
    final_url = "https://www.outdoorgearlab.com/topics/shoes-and-boots/best-trail-running-shoes"

    class FakeResponse:
        def __init__(self, method: str):
            self.method = method
            self.status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def geturl(self):
            return redirect_url

        def read(self, *_args):
            return f'<html><link rel="canonical" href="{final_url}"></html>'.encode()

    def fake_urlopen(request, timeout=4.0):
        return FakeResponse(request.get_method())

    monkeypatch.setattr(providers, "_open_url", fake_urlopen)
    _resolve_source_url.cache_clear()

    resolution = _resolve_source_url(redirect_url)

    assert resolution.resolved_url == final_url
    assert resolution.resolution_method == "html_extraction"


def test_ssl_failure_with_source_title_produces_domain_fallback(monkeypatch):
    redirect_url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/ssl"

    monkeypatch.setattr(providers, "_open_url", lambda _request, timeout=4.0: (_ for _ in ()).throw(URLError("ssl failed")))
    _resolve_source_url.cache_clear()

    source = _source_from_url(redirect_url, "soccer.com")

    assert source["url"] == ""
    assert source["fallback_domain"] == "soccer.com"
    assert source["citation_kind"] == "domain_fallback"
    assert source["resolution_status"] == "domain_fallback"


def test_unresolved_vertex_redirects_are_excluded_from_citation_rankings():
    target = QueryTarget(query="best trail shoes", target_domain="us.puma.com", target_brand_aliases="puma")
    results = [
        ProviderResult(
            query=target.query,
            run_index=0,
            provider="fixture",
            model="fixture",
            response_text="HOKA is mentioned.",
            sources=[
                {
                    "url": "",
                    "title": "Redirect",
                    "original_url": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/missing",
                    "resolved_url": "",
                    "resolution_status": "unresolved_redirect",
                    "resolution_method": "failed",
                    "resolution_error": "No final URL found",
                }
            ],
        )
    ]

    tables = build_tables([target], build_query_runs(results, [target], "us.puma.com"), "us.puma.com")

    assert tables["domain_metrics"].empty
    assert len(tables["unresolved_redirects"]) == 1


def test_domain_fallback_appears_in_domains_not_urls():
    target = QueryTarget(query="best soccer cleats", target_domain="us.puma.com", target_brand_aliases="puma")
    results = [
        ProviderResult(
            query=target.query,
            run_index=0,
            provider="fixture",
            model="fixture",
            response_text="Nike and Puma are mentioned.",
            sources=[
                {
                    "url": "",
                    "title": "soccer.com",
                    "original_url": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/missing",
                    "resolved_url": "",
                    "resolution_status": "domain_fallback",
                    "resolution_method": "failed",
                    "resolution_error": "ssl failed",
                    "source_label": "soccer.com",
                    "fallback_domain": "soccer.com",
                    "citation_kind": "domain_fallback",
                }
            ],
        )
    ]

    tables = build_tables([target], build_query_runs(results, [target], "us.puma.com"), "us.puma.com")

    assert tables["url_metrics"].empty
    assert tables["domain_metrics"].iloc[0]["cited_domain"] == "soccer.com"
    assert tables["domain_metrics"].iloc[0]["citation_kinds"] == "domain_fallback"
    assert "vertexaisearch.cloud.google.com" not in set(tables["domain_metrics"]["cited_domain"])


def test_target_domain_fallback_counts_domain_visibility_not_url_visibility():
    target = QueryTarget(
        query="best puma soccer cleats",
        target_domain="us.puma.com",
        target_url="https://us.puma.com/us/en/sports/soccer",
        acceptable_url_pattern="https://us.puma.com/us/en/sports/soccer/*",
    )
    results = [
        ProviderResult(
            query=target.query,
            run_index=0,
            provider="fixture",
            model="fixture",
            response_text="Puma is mentioned.",
            sources=[
                {
                    "url": "",
                    "title": "us.puma.com",
                    "original_url": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/missing",
                    "resolved_url": "",
                    "resolution_status": "domain_fallback",
                    "resolution_method": "failed",
                    "source_label": "us.puma.com",
                    "fallback_domain": "us.puma.com",
                    "citation_kind": "domain_fallback",
                }
            ],
        )
    ]

    tables = build_tables([target], build_query_runs(results, [target], ""), "")
    metrics = tables["query_metrics"].iloc[0]

    assert metrics["domain_citation_rate"] == 100.0
    assert metrics["approved_page_citation_rate"] == 0.0
    assert tables["url_metrics"].empty
    assert bool(tables["domain_metrics"].iloc[0]["category"] == "Target")


def test_save_zip_to_desktop_uses_timestamped_file(tmp_path):
    tables = {"example": pd.DataFrame([{"a": 1}])}
    zip_bytes = _zip_outputs(tables)
    saved = _save_zip_to_desktop(
        zip_bytes,
        now=datetime(2026, 5, 15, 9, 30, 1),
        desktop=tmp_path,
    )

    assert saved.name == "seo_suite_fanout_20260515_093001.zip"
    assert saved.read_bytes() == zip_bytes


def test_save_zip_to_desktop_raises_on_unwritable_desktop_path(tmp_path):
    desktop_file = tmp_path / "Desktop"
    desktop_file.write_text("not a directory")

    with pytest.raises(OSError):
        _save_zip_to_desktop(b"zip", desktop=desktop_file)
