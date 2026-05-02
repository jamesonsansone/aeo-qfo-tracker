from seo_suite.matching import (
    is_exact_url_match,
    is_pattern_match,
    is_same_or_subdomain,
    normalize_host,
    normalize_url,
)


def test_normalize_host_removes_scheme_www_and_path():
    assert normalize_host("https://www.example.com/path") == "example.com"


def test_normalize_url_ignores_trailing_slash_and_case():
    assert normalize_url("HTTPS://www.Example.com/Products/A/") == "https://example.com/products/a"


def test_same_or_subdomain_matches_subdomains():
    assert is_same_or_subdomain("shop.example.com", "example.com")
    assert is_same_or_subdomain("example.com", "https://www.example.com")
    assert not is_same_or_subdomain("notexample.com", "example.com")


def test_exact_url_match_normalizes_urls():
    assert is_exact_url_match("https://www.example.com/products/a/", "https://example.com/products/a")


def test_pattern_match_supports_globs():
    assert is_pattern_match("https://example.com/products/stability-trail-runner", "https://example.com/products/*trail*")
    assert not is_pattern_match("https://example.com/blog/trail-running", "https://example.com/products/*trail*")


def test_pattern_match_supports_regex_when_regex_markers_present():
    assert is_pattern_match("https://example.com/collections/womens-hiking", r"/collections/.*hiking")

