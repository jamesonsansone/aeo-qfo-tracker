"""Shared dataclasses for the AI search opportunity matrix."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class QueryTarget:
    query: str
    query_cluster: str
    intent: str
    target_url: str
    acceptable_url_pattern: str
    priority: int = 3


@dataclass(frozen=True)
class Citation:
    query: str
    run_index: int
    cited_url: str
    cited_domain: str
    citation_position: int
    title: str = ""
    is_target_domain: bool = False
    is_exact_target_url: bool = False
    is_acceptable_target_url: bool = False


@dataclass(frozen=True)
class QueryRun:
    query: str
    run_index: int
    provider: str
    model: str
    response_text: str
    citations: list[Citation] = field(default_factory=list)


@dataclass(frozen=True)
class ProviderResult:
    query: str
    run_index: int
    provider: str
    model: str
    response_text: str
    sources: list[dict]

