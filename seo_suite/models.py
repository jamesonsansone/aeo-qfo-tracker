"""Shared dataclasses for AI search fan-out analysis."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class QueryTarget:
    query: str
    query_cluster: str = ""
    intent: str = ""
    target_url: str = ""
    acceptable_url_pattern: str = ""
    priority: int = 3
    target_domain: str = ""
    target_brand_aliases: str = ""


@dataclass(frozen=True)
class Citation:
    query: str
    run_index: int
    cited_url: str
    cited_domain: str
    citation_position: int
    title: str = ""
    target_domain: str = ""
    original_url: str = ""
    resolved_url: str = ""
    resolution_status: str = "direct"
    resolution_method: str = "direct"
    final_status_code: int | None = None
    final_domain: str = ""
    resolution_error: str = ""
    source_label: str = ""
    fallback_domain: str = ""
    citation_kind: str = "resolved_url"
    source_type: str = ""
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
    fanout_queries: list[str] = field(default_factory=list)
    grounding_metadata: dict = field(default_factory=dict)
    error: str = ""


@dataclass(frozen=True)
class ProviderResult:
    query: str
    run_index: int
    provider: str
    model: str
    response_text: str
    sources: list[dict]
    fanout_queries: list[str] = field(default_factory=list)
    grounding_metadata: dict = field(default_factory=dict)
    error: str = ""
