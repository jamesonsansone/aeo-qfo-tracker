"""Page-level diagnostics for cited and target URLs."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
import hashlib
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

from seo_suite.matching import extract_domain, normalize_url
from seo_suite.models import QueryTarget


DIAGNOSTIC_COLUMNS = [
    "url",
    "domain",
    "fetch_status",
    "raw_html_available",
    "rendered_html_available",
    "product_schema_raw",
    "product_schema_rendered",
    "offer_schema_raw",
    "offer_schema_rendered",
    "raw_render_delta_type",
    "title_changed",
    "h1_changed",
    "canonical_changed",
    "internal_links_count",
    "external_outlinks_count",
    "word_count_raw",
    "word_count_rendered",
    "lcp",
    "inp",
    "cls",
    "pagespeed_performance_score",
    "field_data_available",
    "diagnostic_flags",
    "diagnostic_severity",
]


class _PageParser(HTMLParser):
    def __init__(self, page_url: str):
        super().__init__()
        self.page_url = page_url
        self.title = ""
        self.h1 = ""
        self.canonical = ""
        self.links: list[str] = []
        self.text_parts: list[str] = []
        self.schema_blocks: list[str] = []
        self._in_title = False
        self._in_h1 = False
        self._in_script = False
        self._script_type = ""
        self._script_parts: list[str] = []
        self._skip_text = False

    def handle_starttag(self, tag: str, attrs):
        attrs_dict = {key.lower(): value or "" for key, value in attrs}
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
        elif tag == "h1":
            self._in_h1 = True
        elif tag == "a" and attrs_dict.get("href"):
            self.links.append(urllib.parse.urljoin(self.page_url, attrs_dict["href"]))
        elif tag == "link" and attrs_dict.get("rel", "").lower() == "canonical":
            self.canonical = attrs_dict.get("href", "")
        elif tag == "script":
            self._in_script = True
            self._script_type = attrs_dict.get("type", "").lower()
            self._script_parts = []
            self._skip_text = True
        elif tag in {"style", "noscript"}:
            self._skip_text = True

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        elif tag == "h1":
            self._in_h1 = False
        elif tag == "script":
            if "ld+json" in self._script_type:
                self.schema_blocks.append("".join(self._script_parts))
            self._in_script = False
            self._script_type = ""
            self._script_parts = []
            self._skip_text = False
        elif tag in {"style", "noscript"}:
            self._skip_text = False

    def handle_data(self, data: str):
        text = data.strip()
        if not text:
            return
        if self._in_script:
            self._script_parts.append(data)
            return
        if self._in_title:
            self.title += (" " + text) if self.title else text
        elif self._in_h1 and not self.h1:
            self.h1 = text
        elif not self._skip_text:
            self.text_parts.append(text)


def build_page_diagnostics(
    urls: list[str],
    fixture_path: str | Path | None = None,
    fetch_live: bool = False,
    pagespeed_api_key: str | None = None,
    render_live: bool = False,
    cache_dir: str | Path = ".cache/page_diagnostics",
    cache_ttl_hours: int = 168,
) -> pd.DataFrame:
    fixture_pages = _load_fixture_pages(fixture_path) if fixture_path else {}
    rows = []
    for url in sorted({normalize_url(value) for value in urls if value}):
        fixture = fixture_pages.get(url)
        if fixture:
            raw_html = fixture.get("raw_html", "")
            rendered_html = fixture.get("rendered_html", "")
            status = fixture.get("fetch_status", "fixture")
            pagespeed = fixture.get("pagespeed", {})
        elif fetch_live:
            cached = _read_cache(url, cache_dir, cache_ttl_hours)
            if cached:
                rows.append(cached)
                continue
            raw_html, status = _fetch_raw_html(url)
            rendered_html, render_status = _render_html_playwright(url) if render_live else ("", "render_not_requested")
            if rendered_html:
                status = f"{status};{render_status}"
            elif render_live:
                status = f"{status};{render_status}"
            pagespeed = fetch_pagespeed_metrics(url, pagespeed_api_key)
        else:
            raw_html = ""
            rendered_html = ""
            status = "not_fetched"
            pagespeed = {}
        row = analyze_page(url, raw_html, rendered_html, status, pagespeed)
        if fetch_live and not fixture:
            _write_cache(url, row, cache_dir, raw_html=raw_html, rendered_html=rendered_html)
        rows.append(row)
    return pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS)


def collect_diagnostic_urls(targets: list[QueryTarget], tables: dict[str, pd.DataFrame]) -> list[str]:
    urls = {target.target_url for target in targets if target.target_url}
    citations = tables.get("citations", pd.DataFrame())
    metrics = tables.get("query_metrics", pd.DataFrame())
    if not citations.empty and "cited_url" in citations.columns:
        urls.update(citations["cited_url"].dropna().astype(str).tolist())
    if not metrics.empty and "top_competitor_url" in metrics.columns:
        urls.update(
            value
            for value in metrics["top_competitor_url"].dropna().astype(str).tolist()
            if value and value != "-"
        )
    return sorted(urls)


def analyze_page(
    url: str,
    raw_html: str,
    rendered_html: str = "",
    fetch_status: str = "fixture",
    pagespeed: dict | None = None,
) -> dict:
    raw = _extract_page_facts(url, raw_html)
    rendered = _extract_page_facts(url, rendered_html)
    pagespeed = pagespeed or {}
    raw_available = bool(raw_html)
    rendered_available = bool(rendered_html)
    product_raw = _has_schema(raw_html, "Product")
    product_rendered = _has_schema(rendered_html, "Product")
    offer_raw = _has_schema(raw_html, "Offer")
    offer_rendered = _has_schema(rendered_html, "Offer")
    title_changed = bool(raw["title"] and rendered["title"] and raw["title"] != rendered["title"])
    h1_changed = bool(raw["h1"] and rendered["h1"] and raw["h1"] != rendered["h1"])
    canonical_changed = bool(raw["canonical"] and rendered["canonical"] and raw["canonical"] != rendered["canonical"])
    delta_type = _delta_type(raw, rendered, product_raw, product_rendered, offer_raw, offer_rendered)
    flags = _diagnostic_flags(
        product_raw,
        product_rendered,
        offer_raw,
        offer_rendered,
        delta_type,
        pagespeed,
    )
    return {
        "url": normalize_url(url),
        "domain": extract_domain(url),
        "fetch_status": fetch_status,
        "raw_html_available": raw_available,
        "rendered_html_available": rendered_available,
        "product_schema_raw": product_raw,
        "product_schema_rendered": product_rendered,
        "offer_schema_raw": offer_raw,
        "offer_schema_rendered": offer_rendered,
        "raw_render_delta_type": delta_type,
        "title_changed": title_changed,
        "h1_changed": h1_changed,
        "canonical_changed": canonical_changed,
        "internal_links_count": raw["internal_links_count"],
        "external_outlinks_count": raw["external_outlinks_count"],
        "word_count_raw": raw["word_count"],
        "word_count_rendered": rendered["word_count"],
        "lcp": pagespeed.get("lcp"),
        "inp": pagespeed.get("inp"),
        "cls": pagespeed.get("cls"),
        "pagespeed_performance_score": pagespeed.get("pagespeed_performance_score"),
        "field_data_available": bool(pagespeed.get("field_data_available", False)),
        "diagnostic_flags": " | ".join(flags) if flags else "No issue detected",
        "diagnostic_severity": _severity(flags),
    }


def build_weekly_queue(
    targets: list[QueryTarget],
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    metrics = tables["query_metrics"]
    diagnostics = tables.get("page_diagnostics", pd.DataFrame())
    diagnostic_by_url = (
        diagnostics.set_index("url").to_dict("index") if not diagnostics.empty and "url" in diagnostics.columns else {}
    )
    target_by_query = {target.query: target for target in targets}
    rows = []
    for row in metrics.to_dict("records"):
        target = target_by_query[row["query"]]
        target_diag = diagnostic_by_url.get(normalize_url(target.target_url), {})
        issue = _queue_issue(row, target_diag)
        severity = max(float(row["opportunity_score"]), float(target_diag.get("diagnostic_severity", 0)) * 12.5)
        rows.append(
            {
                "query": row["query"],
                "query_cluster": row["query_cluster"],
                "recommended_follow_up": issue,
                "target_url": target.target_url,
                "top_competitor_url": row["top_competitor_url"],
                "approved_page_visibility": row["acceptable_target_presence_rate"],
                "approved_page_position_score": row["acceptable_mrr"],
                "wrong_client_page_cited": row["target_mismatch"],
                "opportunity_score": row["opportunity_score"],
                "diagnostic_severity": target_diag.get("diagnostic_severity", 0),
                "diagnostic_flags": target_diag.get("diagnostic_flags", "Not checked"),
                "weekly_priority_score": round(min(100.0, severity), 1),
            }
        )
    return pd.DataFrame(rows).sort_values("weekly_priority_score", ascending=False).reset_index(drop=True)


def fetch_pagespeed_metrics(url: str, api_key: str | None = None) -> dict:
    key = api_key or os.environ.get("PAGESPEED_API_KEY", "")
    if not key:
        return {}
    endpoint = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
    params = urllib.parse.urlencode({"url": url, "strategy": "mobile", "key": key})
    try:
        with urllib.request.urlopen(f"{endpoint}?{params}", timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}
    lighthouse = payload.get("lighthouseResult", {})
    audits = lighthouse.get("audits", {})
    categories = lighthouse.get("categories", {})
    loading = payload.get("loadingExperience", {}).get("metrics", {})
    return {
        "lcp": _psi_metric(loading, audits, "LARGEST_CONTENTFUL_PAINT_MS", "largest-contentful-paint"),
        "inp": _psi_metric(loading, audits, "INTERACTION_TO_NEXT_PAINT", "interactive"),
        "cls": _psi_metric(loading, audits, "CUMULATIVE_LAYOUT_SHIFT_SCORE", "cumulative-layout-shift"),
        "pagespeed_performance_score": round((categories.get("performance", {}).get("score") or 0) * 100, 1),
        "field_data_available": bool(loading),
    }


def _load_fixture_pages(path: str | Path) -> dict[str, dict]:
    fixture_pages: dict[str, dict] = {}
    fixture_path = Path(path)
    if not fixture_path.exists():
        return fixture_pages
    with fixture_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            fixture_pages[normalize_url(row["url"])] = row
    return fixture_pages


def _fetch_raw_html(url: str) -> tuple[str, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "seo-suite/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                return "", f"unsupported_content_type:{content_type}"
            return response.read().decode("utf-8", errors="replace"), f"http_{response.status}"
    except Exception as exc:
        return "", f"fetch_error:{exc.__class__.__name__}"


def _render_html_playwright(url: str, timeout_ms: int = 20000) -> tuple[str, str]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return "", "render_error:playwright_not_installed"

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(user_agent="seo-suite/0.1")
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            html = page.content()
            browser.close()
            return html, "rendered_with_playwright"
    except Exception as exc:
        return "", f"render_error:{exc.__class__.__name__}"


def _cache_path(url: str, cache_dir: str | Path) -> Path:
    digest = hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{digest}.json"


def _read_cache(url: str, cache_dir: str | Path, cache_ttl_hours: int) -> dict | None:
    path = _cache_path(url, cache_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    fetched_at = float(payload.get("cached_at", 0))
    if cache_ttl_hours > 0 and time.time() - fetched_at > cache_ttl_hours * 3600:
        return None
    row = payload.get("row")
    return row if isinstance(row, dict) else None


def _write_cache(url: str, row: dict, cache_dir: str | Path, raw_html: str = "", rendered_html: str = "") -> None:
    path = _cache_path(url, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "url": normalize_url(url),
        "cached_at": time.time(),
        "row": row,
        "raw_html": raw_html,
        "rendered_html": rendered_html,
    }
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def _extract_page_facts(url: str, html: str) -> dict:
    if not html:
        return {
            "title": "",
            "h1": "",
            "canonical": "",
            "internal_links_count": 0,
            "external_outlinks_count": 0,
            "word_count": 0,
        }
    parser = _PageParser(url)
    try:
        parser.feed(html)
    except Exception:
        pass
    page_domain = extract_domain(url)
    internal = 0
    external = 0
    for link in parser.links:
        link_domain = extract_domain(link)
        if not link_domain:
            continue
        if link_domain == page_domain or link_domain.endswith("." + page_domain):
            internal += 1
        else:
            external += 1
    text = " ".join(parser.text_parts)
    return {
        "title": _clean(parser.title),
        "h1": _clean(parser.h1),
        "canonical": normalize_url(parser.canonical) if parser.canonical else "",
        "internal_links_count": internal,
        "external_outlinks_count": external,
        "word_count": len(re.findall(r"\b\w+\b", text)),
    }


def _has_schema(html: str, schema_type: str) -> bool:
    if not html:
        return False
    schema_re = re.compile(r'"@type"\s*:\s*(?:"[^"]*%s[^"]*"|\[[^\]]*"%s"[^\]]*\])' % (schema_type, schema_type), re.I)
    itemtype_re = re.compile(r"itemtype=['\"]https?://schema\.org/%s['\"]" % schema_type, re.I)
    return bool(schema_re.search(html) or itemtype_re.search(html))


def _delta_type(raw: dict, rendered: dict, product_raw: bool, product_rendered: bool, offer_raw: bool, offer_rendered: bool) -> str:
    if not rendered["word_count"] and not product_rendered and not offer_rendered:
        return "Rendered version not checked"
    if (product_rendered and not product_raw) or (offer_rendered and not offer_raw):
        return "Structured data only after rendering"
    if rendered["word_count"] > raw["word_count"] * 1.5 and rendered["word_count"] - raw["word_count"] > 50:
        return "Important content only after rendering"
    if raw["word_count"] and abs(rendered["word_count"] - raw["word_count"]) <= max(25, raw["word_count"] * 0.1):
        return "Raw and rendered content broadly aligned"
    return "Moderate raw/render difference"


def _diagnostic_flags(product_raw: bool, product_rendered: bool, offer_raw: bool, offer_rendered: bool, delta_type: str, pagespeed: dict) -> list[str]:
    flags = []
    if product_rendered and not product_raw:
        flags.append("Product schema missing in raw HTML")
    if offer_rendered and not offer_raw:
        flags.append("Offer schema missing in raw HTML")
    if delta_type in {"Important content only after rendering", "Structured data only after rendering"}:
        flags.append("Important content only appears after rendering")
    if _has_cwv_risk(pagespeed):
        flags.append("Core Web Vitals risk")
    return flags


def _has_cwv_risk(pagespeed: dict) -> bool:
    lcp = pagespeed.get("lcp")
    inp = pagespeed.get("inp")
    cls = pagespeed.get("cls")
    score = pagespeed.get("pagespeed_performance_score")
    return bool(
        (isinstance(lcp, (int, float)) and lcp > 2500)
        or (isinstance(inp, (int, float)) and inp > 200)
        or (isinstance(cls, (int, float)) and cls > 0.1)
        or (isinstance(score, (int, float)) and score < 70)
    )


def _severity(flags: list[str]) -> int:
    weights = {
        "Product schema missing in raw HTML": 6,
        "Offer schema missing in raw HTML": 6,
        "Important content only appears after rendering": 7,
        "Core Web Vitals risk": 5,
    }
    return max([weights.get(flag, 1) for flag in flags], default=0)


def _queue_issue(metric_row: dict, target_diag: dict) -> str:
    if metric_row["acceptable_target_presence_rate"] == 0 and not metric_row["target_mismatch"]:
        return "Target page not cited"
    if metric_row["target_mismatch"]:
        return "Wrong client page cited"
    flags = target_diag.get("diagnostic_flags", "")
    if "Product schema missing in raw HTML" in flags:
        return "Product schema missing in raw HTML"
    if "Offer schema missing in raw HTML" in flags:
        return "Offer schema missing in raw HTML"
    if "Important content only appears after rendering" in flags:
        return "Important content only appears after rendering"
    if metric_row["top_competitor_citations"] >= 3:
        return "Competitor has stronger comparison content"
    return "No issue detected"


def _psi_metric(loading: dict, audits: dict, field_key: str, audit_key: str):
    field = loading.get(field_key, {})
    if "percentile" in field:
        return field["percentile"]
    audit = audits.get(audit_key, {})
    return audit.get("numericValue")


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()
