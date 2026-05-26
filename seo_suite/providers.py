"""Search providers for live and fixture-backed query fan-out runs."""

from __future__ import annotations

import os
import html
import re
import ssl
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urljoin, urlparse
from urllib.request import Request, urlopen

import certifi

from seo_suite.io import read_jsonl
from seo_suite.models import ProviderResult, QueryTarget


@dataclass(frozen=True)
class URLResolution:
    original_url: str
    resolved_url: str
    resolution_status: str
    resolution_method: str
    final_status_code: int | None = None
    final_domain: str = ""
    resolution_error: str = ""


class SearchProvider(Protocol):
    name: str
    model: str

    def run_query(self, target: QueryTarget, run_index: int) -> ProviderResult:
        ...


class FixtureProvider:
    name = "fixture"
    model = "fixture-grounded-search"

    def __init__(self, fixture_path: str):
        self.fixture_path = fixture_path
        self._by_query: dict[str, list[dict]] = defaultdict(list)
        for row in read_jsonl(fixture_path):
            self._by_query[row["query"]].append(row)

    def run_query(self, target: QueryTarget, run_index: int) -> ProviderResult:
        rows = self._by_query.get(target.query, [])
        if not rows:
            return ProviderResult(
                query=target.query,
                run_index=run_index,
                provider=self.name,
                model=self.model,
                response_text="No fixture data was available for this query.",
                sources=[],
                fanout_queries=[],
                error="",
            )
        row = rows[run_index % len(rows)]
        return ProviderResult(
            query=target.query,
            run_index=run_index,
            provider=row.get("provider", self.name),
            model=row.get("model", self.model),
            response_text=row.get("response_text", ""),
            sources=row.get("sources", []),
            fanout_queries=row.get("fanout_queries", []),
            grounding_metadata=row.get("grounding_metadata", {}),
            error=row.get("error", ""),
        )


class GeminiGroundedSearchProvider:
    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.1-flash-lite",
        delay_seconds: float = 0.0,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        self.model = model
        self.delay_seconds = delay_seconds
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for live Gemini runs.")

    def run_query(self, target: QueryTarget, run_index: int) -> ProviderResult:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        prompt = _grounded_prompt(target)
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
            ),
        )
        if self.delay_seconds:
            time.sleep(self.delay_seconds)

        sources = _extract_gemini_sources(response)
        fanout_queries = _extract_gemini_fanout_queries(response)
        return ProviderResult(
            query=target.query,
            run_index=run_index,
            provider=self.name,
            model=self.model,
            response_text=getattr(response, "text", "") or "",
            sources=sources,
            fanout_queries=fanout_queries,
            grounding_metadata=_extract_gemini_grounding_metadata(response),
        )


class OpenAIWebSearchProvider:
    name = "openai"

    def __init__(self, api_key: str | None = None, model: str = "gpt-5", delay_seconds: float = 0.0):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.delay_seconds = delay_seconds
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for live OpenAI runs.")
        from openai import OpenAI
        try:
            self._client = OpenAI(api_key=self.api_key)
        except TypeError as exc:
            if "proxies" in str(exc):
                import sys
                import openai as _oai
                import httpx as _httpx
                raise RuntimeError(
                    f"OpenAI client failed to initialize due to an incompatible openai/httpx package "
                    f"pair (openai {_oai.__version__} + "
                    f"httpx {_httpx.__version__} are incompatible). "
                    f"Active Python: {sys.executable}. "
                    "Run the app with: python -m streamlit run app.py"
                ) from exc
            raise

    def run_query(self, target: QueryTarget, run_index: int) -> ProviderResult:
        response = self._client.responses.create(
            model=self.model,
            tools=[{"type": "web_search"}],
            include=["web_search_call.action.sources"],
            input=_grounded_prompt(target),
        )
        if self.delay_seconds:
            time.sleep(self.delay_seconds)

        sources = _extract_openai_sources(response)
        fanout_queries = _extract_openai_fanout_queries(response)
        return ProviderResult(
            query=target.query,
            run_index=run_index,
            provider=self.name,
            model=self.model,
            response_text=getattr(response, "output_text", "") or "",
            sources=sources,
            fanout_queries=fanout_queries,
            grounding_metadata={"web_search_actions": _extract_openai_actions(response)},
        )


def _grounded_prompt(target: QueryTarget) -> str:
    return "\n".join(
        [
            "Answer the ecommerce shopping or research query using current web sources.",
            "Be neutral: do not favor any brand, store, or domain unless the user's query names it.",
            "Prefer concise, source-grounded recommendations and cite the sources you used.",
            f"Query: {target.query}",
        ]
    )


def _extract_openai_sources(response) -> list[dict]:
    sources: list[dict] = []
    seen: dict[str, int] = {}  # raw url → index in sources

    for item in _iter_output_items(response):
        action = _get_value(item, "action")
        for source in _as_list(_get_value(action, "sources")):
            url = _get_value(source, "url") or _get_value(source, "uri") or ""
            title = _get_value(source, "title") or ""
            if url and url not in seen:
                entry = _source_from_url(url, title)
                entry["source_type"] = "grounded"
                seen[url] = len(sources)
                sources.append(entry)

        if _get_value(item, "type") == "message":
            for content in _as_list(_get_value(item, "content")):
                for annotation in _as_list(_get_value(content, "annotations")):
                    if _get_value(annotation, "type") == "url_citation":
                        url = _get_value(annotation, "url") or ""
                        title = _get_value(annotation, "title") or ""
                        if not url:
                            continue
                        if url in seen:
                            sources[seen[url]]["source_type"] = "inline_cited"
                        else:
                            entry = _source_from_url(url, title)
                            entry["source_type"] = "inline_cited"
                            seen[url] = len(sources)
                            sources.append(entry)
    return sources


def _extract_openai_fanout_queries(response) -> list[str]:
    queries: list[str] = []
    seen = set()
    for action in _extract_openai_actions(response):
        for key in ("queries", "query", "search_query", "searchQuery"):
            for value in _as_list(action.get(key)):
                if isinstance(value, str) and value and value not in seen:
                    queries.append(value)
                    seen.add(value)
    return queries


def _extract_openai_actions(response) -> list[dict]:
    actions = []
    for item in _iter_output_items(response):
        action = _get_value(item, "action")
        if action:
            actions.append(_plain(action))
    return actions


def _extract_gemini_sources(response) -> list[dict]:
    sources: list[dict] = []
    seen = set()
    for metadata in _iter_gemini_grounding_metadata(response):
        for chunk in _as_list(_get_any_value(metadata, "grounding_chunks", "groundingChunks")):
            web = _get_value(chunk, "web")
            if not web:
                continue
            url = _get_value(web, "uri") or _get_value(web, "url") or ""
            title = _get_value(web, "title") or ""
            if url:
                source = _source_from_url(url, title)
                dedupe_key = source.get("url") or source.get("original_url")
                if dedupe_key and dedupe_key not in seen:
                    sources.append(source)
                    seen.add(dedupe_key)
    return sources


def _extract_gemini_fanout_queries(response) -> list[str]:
    queries: list[str] = []
    seen = set()
    for metadata in _iter_gemini_grounding_metadata(response):
        for key in ("web_search_queries", "webSearchQueries", "retrieval_queries", "retrievalQueries"):
            for value in _as_list(_get_value(metadata, key)):
                if isinstance(value, str) and value and value not in seen:
                    queries.append(value)
                    seen.add(value)
    return queries


def _extract_gemini_grounding_metadata(response) -> dict:
    items = [_plain(metadata) for metadata in _iter_gemini_grounding_metadata(response)]
    if not items:
        return {}
    return {"candidates": items}


def _iter_gemini_grounding_metadata(response):
    for candidate in _as_list(_get_value(response, "candidates")):
        metadata = _get_any_value(candidate, "grounding_metadata", "groundingMetadata")
        if metadata:
            yield metadata


def _iter_output_items(response):
    return _as_list(_get_value(response, "output"))


def _source_from_url(url: str, title: str = "") -> dict:
    resolution = _resolve_source_url(url)
    resolved_url = resolution.resolved_url if resolution.resolution_status != "unresolved_redirect" else ""
    source_label = str(title or "").strip()
    fallback_domain = ""
    citation_kind = "resolved_url"
    if not resolved_url:
        fallback_domain = _source_label_to_domain(source_label)
        citation_kind = "domain_fallback" if fallback_domain else "unresolved"
    resolution_status = "domain_fallback" if citation_kind == "domain_fallback" else resolution.resolution_status
    return {
        "url": resolved_url,
        "title": title,
        "original_url": resolution.original_url,
        "resolved_url": resolution.resolved_url,
        "resolution_status": resolution_status,
        "resolution_method": resolution.resolution_method,
        "final_status_code": resolution.final_status_code,
        "final_domain": resolution.final_domain,
        "resolution_error": resolution.resolution_error,
        "source_label": source_label,
        "fallback_domain": fallback_domain,
        "citation_kind": citation_kind,
    }


@lru_cache(maxsize=512)
def _resolve_source_url(url: str, timeout: float = 4.0) -> URLResolution:
    """Resolve Google/Vertex grounding wrappers while preserving audit metadata."""

    original_url = str(url or "").strip()
    if not original_url:
        return URLResolution("", "", "unresolved_redirect", "empty", resolution_error="Empty URL")
    if not _looks_like_grounding_redirect(original_url):
        return _resolution(original_url, original_url, "direct", "direct")

    encoded_destination = _extract_encoded_destination(original_url)
    if encoded_destination:
        return _resolution(original_url, encoded_destination, "resolved", "encoded_param")

    last_error = ""
    for method in ("HEAD", "GET"):
        try:
            request = Request(
                original_url,
                headers={"User-Agent": "Mozilla/5.0 SEO Suite URL resolver"},
                method=method,
            )
            with _open_url(request, timeout=timeout) as response:
                status = _response_status(response)
                final_url = response.geturl() or original_url
                if _usable_destination(original_url, final_url):
                    return _resolution(original_url, final_url, "resolved", f"http_{method.lower()}", status)
                if method == "GET":
                    body = _read_response_text(response)
                    html_destination = _extract_html_destination(body, final_url or original_url)
                    if html_destination and _usable_destination(original_url, html_destination):
                        return _resolution(original_url, html_destination, "resolved", "html_extraction", status)
                if method == "HEAD" and status in {403, 405}:
                    continue
                last_error = f"HTTP {status} did not expose a final URL"
        except HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            final_url = exc.geturl() or ""
            if _usable_destination(original_url, final_url):
                return _resolution(original_url, final_url, "resolved", f"http_{method.lower()}", exc.code)
            if method == "HEAD" and exc.code in {403, 405}:
                continue
        except (OSError, URLError, ValueError) as exc:
            last_error = str(exc)
            if method == "HEAD":
                continue

    return URLResolution(
        original_url=original_url,
        resolved_url="",
        resolution_status="unresolved_redirect",
        resolution_method="failed",
        final_status_code=None,
        final_domain="",
        resolution_error=last_error or "No final URL found",
    )


def _resolve_grounding_redirect(url: str, timeout: float = 4.0) -> str:
    """Backward-compatible URL-only redirect resolver."""

    resolution = _resolve_source_url(url, timeout=timeout)
    return resolution.resolved_url or url


def _open_url(request: Request, timeout: float):
    return urlopen(request, timeout=timeout, context=_certifi_ssl_context())


@lru_cache(maxsize=1)
def _certifi_ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


def _looks_like_grounding_redirect(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    path = urlparse(url).path.lower()
    return (
        "vertexaisearch" in host
        or "grounding" in host
        or "googleusercontent" in host
        or "grounding-api-redirect" in path
        or host.endswith("google.com")
        and path.startswith("/url")
    )


def _resolution(
    original_url: str,
    resolved_url: str,
    status: str,
    method: str,
    final_status_code: int | None = None,
    error: str = "",
) -> URLResolution:
    return URLResolution(
        original_url=original_url,
        resolved_url=resolved_url,
        resolution_status=status,
        resolution_method=method,
        final_status_code=final_status_code,
        final_domain=urlparse(resolved_url).netloc.lower(),
        resolution_error=error,
    )


def _extract_encoded_destination(original_url: str) -> str:
    parsed = urlparse(original_url)
    query = parse_qs(parsed.query)
    candidate_keys = {
        "url",
        "u",
        "q",
        "adurl",
        "target",
        "dest",
        "destination",
        "redirect",
        "redirect_url",
    }
    for key, values in query.items():
        if key.lower() not in candidate_keys:
            continue
        for value in values:
            candidate = _clean_url_candidate(unquote(value))
            if _usable_destination(original_url, candidate):
                return candidate

    decoded_path = unquote(parsed.path)
    for candidate in _embedded_url_candidates(decoded_path):
        if _usable_destination(original_url, candidate):
            return candidate

    decoded_url = unquote(original_url)
    for candidate in _embedded_url_candidates(decoded_url):
        if _usable_destination(original_url, candidate):
            return candidate
    return ""


def _embedded_url_candidates(text: str) -> list[str]:
    candidates = []
    for match in re.finditer(r"https?://", text or "", flags=re.I):
        candidates.append(_clean_url_candidate(text[match.start() :]))
    return candidates


def _clean_url_candidate(value: str) -> str:
    candidate = html.unescape(str(value or "").strip())
    if not candidate:
        return ""
    candidate = re.split(r"[\s\"'<>`]", candidate, maxsplit=1)[0]
    return candidate.rstrip(").,;]")


def _usable_destination(original_url: str, candidate_url: str) -> bool:
    candidate = _clean_url_candidate(candidate_url)
    if not candidate or candidate == original_url:
        return False
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    return not _looks_like_grounding_redirect(candidate)


def _source_label_to_domain(source_label: str) -> str:
    label = str(source_label or "").strip()
    if not label:
        return ""
    candidate = _clean_url_candidate(label)
    if not candidate:
        return ""
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", candidate):
        host = urlparse(candidate).netloc.lower()
    else:
        host = candidate.split("/", 1)[0].lower()
    if host.startswith("www."):
        host = host[4:]
    host = host.split(":", 1)[0].strip(".")
    if not re.match(r"^(?:[a-z0-9-]+\.)+[a-z]{2,}$", host):
        return ""
    if _looks_like_grounding_redirect(f"https://{host}/"):
        return ""
    return host


def _response_status(response) -> int:
    return int(getattr(response, "status", getattr(response, "code", 200)) or 200)


def _read_response_text(response, limit: int = 120_000) -> str:
    try:
        raw = response.read(limit)
    except TypeError:
        raw = response.read()
    if isinstance(raw, str):
        return raw
    return raw.decode("utf-8", errors="ignore")


def _extract_html_destination(body: str, base_url: str) -> str:
    if not body:
        return ""
    patterns = [
        r"<link\b[^>]*rel=[\"'][^\"']*canonical[^\"']*[\"'][^>]*href=[\"']([^\"']+)[\"']",
        r"<link\b[^>]*href=[\"']([^\"']+)[\"'][^>]*rel=[\"'][^\"']*canonical[^\"']*[\"']",
        r"<meta\b[^>]*http-equiv=[\"']refresh[\"'][^>]*content=[\"'][^\"']*url=([^\"'>]+)[\"']",
        r"<a\b[^>]*href=[\"']([^\"']+)[\"']",
    ]
    for pattern in patterns:
        match = re.search(pattern, body, flags=re.I | re.S)
        if match:
            return _clean_url_candidate(urljoin(base_url, match.group(1)))
    return ""


def _get_value(value: Any, key: str):
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _get_any_value(value: Any, *keys: str):
    for key in keys:
        found = _get_value(value, key)
        if found is not None:
            return found
    return None


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value) if isinstance(value, tuple) else [value]


def _plain(value):
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
