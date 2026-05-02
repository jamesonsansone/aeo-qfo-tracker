"""Search providers for live and fixture-backed query runs."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Protocol

from seo_suite.io import read_jsonl
from seo_suite.models import ProviderResult, QueryTarget


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
            )
        row = rows[run_index % len(rows)]
        return ProviderResult(
            query=target.query,
            run_index=run_index,
            provider=row.get("provider", self.name),
            model=row.get("model", self.model),
            response_text=row.get("response_text", ""),
            sources=row.get("sources", []),
        )


class OpenAIWebSearchProvider:
    name = "openai"

    def __init__(self, api_key: str | None = None, model: str = "gpt-5", delay_seconds: float = 1.0):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.delay_seconds = delay_seconds
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for live OpenAI runs.")

    def run_query(self, target: QueryTarget, run_index: int) -> ProviderResult:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        prompt = (
            "Answer the shopping or research query using current web sources. "
            "Prefer concise, source-grounded recommendations. Query: "
            f"{target.query}"
        )
        response = client.responses.create(
            model=self.model,
            tools=[{"type": "web_search"}],
            include=["web_search_call.action.sources"],
            input=prompt,
        )
        if self.delay_seconds:
            time.sleep(self.delay_seconds)

        sources = _extract_openai_sources(response)
        return ProviderResult(
            query=target.query,
            run_index=run_index,
            provider=self.name,
            model=self.model,
            response_text=getattr(response, "output_text", "") or "",
            sources=sources,
        )


def _extract_openai_sources(response) -> list[dict]:
    sources: list[dict] = []
    seen = set()

    for item in getattr(response, "output", []) or []:
        action = getattr(item, "action", None)
        for source in getattr(action, "sources", []) or []:
            url = getattr(source, "url", "") or ""
            title = getattr(source, "title", "") or ""
            if url and url not in seen:
                sources.append({"url": url, "title": title})
                seen.add(url)

        if getattr(item, "type", "") == "message":
            for content in getattr(item, "content", []) or []:
                for annotation in getattr(content, "annotations", []) or []:
                    if getattr(annotation, "type", "") == "url_citation":
                        url = getattr(annotation, "url", "") or ""
                        title = getattr(annotation, "title", "") or ""
                        if url and url not in seen:
                            sources.append({"url": url, "title": title})
                            seen.add(url)

    return sources
