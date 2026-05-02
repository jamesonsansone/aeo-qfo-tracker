"""Exact-text content alignment scoring for AI answers and page content."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

from seo_suite.matching import normalize_url
from seo_suite.models import QueryTarget


class _SectionParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sections: list[dict] = []
        self._heading = "Document Body"
        self._current_tag = ""
        self._parts: list[str] = []
        self._skip = False
        self._title = ""

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        self._current_tag = tag
        if tag in {"script", "style", "noscript"}:
            self._skip = True
        if tag in {"title", "h1", "h2", "h3", "p", "li", "dt", "dd"}:
            self._parts = []

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        text = _clean(" ".join(self._parts))
        if tag == "title" and text:
            self._title = text
            self.sections.append({"heading": "Title", "text": text, "section_type": "title"})
        elif tag in {"h1", "h2", "h3"} and text:
            self._heading = text
            self.sections.append({"heading": text, "text": text, "section_type": tag})
        elif tag in {"p", "li", "dt", "dd"} and text:
            self.sections.append({"heading": self._heading, "text": text, "section_type": tag})
        if tag in {"script", "style", "noscript"}:
            self._skip = False
        if tag == self._current_tag:
            self._current_tag = ""
        self._parts = []

    def handle_data(self, data: str):
        if not self._skip and self._current_tag in {"title", "h1", "h2", "h3", "p", "li", "dt", "dd"}:
            self._parts.append(data)


def build_content_alignment(
    targets: list[QueryTarget],
    tables: dict[str, pd.DataFrame],
    fixture_path: str | Path | None = None,
    cache_dir: str | Path = ".cache/page_diagnostics",
) -> pd.DataFrame:
    html_by_url = _load_html_sources(fixture_path, cache_dir)
    diagnostics = tables.get("page_diagnostics", pd.DataFrame())
    runs = tables.get("runs", pd.DataFrame())
    citations = tables.get("citations", pd.DataFrame())
    metrics = tables.get("query_metrics", pd.DataFrame())
    if diagnostics.empty or runs.empty or metrics.empty:
        return _empty_alignment()

    metrics_by_query = metrics.set_index("query").to_dict("index")
    rows: list[dict] = []
    for target in targets:
        answer_text = " ".join(
            runs.loc[runs["query"] == target.query, "response_text"].dropna().astype(str).tolist()
        )
        urls = _urls_for_query(target, citations, metrics_by_query.get(target.query, {}))
        scored_rows = []
        for url, role in urls:
            html_source = html_by_url.get(normalize_url(url), {})
            html = html_source.get("rendered_html") or html_source.get("raw_html") or ""
            source_text_mode = html_source.get("mode", "not_available")
            chunks = chunk_html_sections(html)
            best = best_tfidf_match(answer_text, chunks)
            scored_rows.append(
                {
                    "query": target.query,
                    "url": normalize_url(url),
                    "url_role": role,
                    "source_text_mode": source_text_mode,
                    "best_matching_heading": best["heading"],
                    "best_chunk_text": best["text"],
                    "tfidf_score": best["score"],
                    "embedding_score": None,
                    "answer_capsule_present": best["score"] >= 0.18 and len(best["text"].split()) <= 120,
                }
            )
        top_competitor = max(
            (row["tfidf_score"] for row in scored_rows if row["url_role"] == "Cited Competitor URL"),
            default=0.0,
        )
        target_score = max(
            (row["tfidf_score"] for row in scored_rows if row["url_role"] == "Mapped Target URL"),
            default=0.0,
        )
        gap = round(max(0.0, top_competitor - target_score), 3)
        for row in scored_rows:
            row["alignment_gap_vs_top_competitor"] = gap
            rows.append(row)
    return pd.DataFrame(rows)


def chunk_html_sections(html: str, min_words: int = 4) -> list[dict]:
    if not html:
        return []
    parser = _SectionParser()
    try:
        parser.feed(html)
    except Exception:
        pass
    chunks = []
    buffer_by_heading: dict[str, list[str]] = {}
    section_type_by_heading: dict[str, str] = {}
    for section in parser.sections:
        text = _clean(section["text"])
        if len(text.split()) < min_words and section["section_type"] not in {"title", "h1", "h2", "h3"}:
            continue
        heading = section["heading"] or "Document Body"
        buffer_by_heading.setdefault(heading, []).append(text)
        section_type_by_heading.setdefault(heading, section["section_type"])
    for heading, parts in buffer_by_heading.items():
        text = _clean(" ".join(parts))
        if text:
            chunks.append(
                {
                    "heading": heading,
                    "text": text,
                    "word_count": len(text.split()),
                    "section_type": section_type_by_heading.get(heading, "body"),
                }
            )
    return chunks


def best_tfidf_match(reference_text: str, chunks: list[dict]) -> dict:
    if not reference_text or not chunks:
        return {"heading": "", "text": "", "score": 0.0}
    docs = [reference_text] + [chunk["text"] for chunk in chunks]
    vectors = _tfidf_vectors(docs)
    reference = vectors[0]
    best_index = -1
    best_score = 0.0
    for index, vector in enumerate(vectors[1:]):
        score = _cosine(reference, vector)
        if score > best_score:
            best_score = score
            best_index = index
    if best_index == -1:
        return {"heading": "", "text": "", "score": 0.0}
    chunk = chunks[best_index]
    return {
        "heading": chunk["heading"],
        "text": chunk["text"][:700],
        "score": round(best_score, 3),
    }


def _tfidf_vectors(docs: list[str]) -> list[dict[str, float]]:
    tokenized = [_tokens(doc) for doc in docs]
    doc_count = len(tokenized)
    df: Counter[str] = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    vectors = []
    for tokens in tokenized:
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        vector: dict[str, float] = {}
        for token, count in counts.items():
            tf = count / total
            idf = math.log((1 + doc_count) / (1 + df[token])) + 1
            vector[token] = tf * idf
        vectors.append(vector)
    return vectors


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    ]


def _urls_for_query(target: QueryTarget, citations: pd.DataFrame, metric: dict) -> list[tuple[str, str]]:
    urls = [(target.target_url, "Mapped Target URL")]
    if not citations.empty:
        query_citations = citations[citations["query"] == target.query]
        competitor_urls = (
            query_citations.loc[~query_citations["is_target_domain"], "cited_url"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(3)
            .tolist()
        )
        urls.extend((url, "Cited Competitor URL") for url in competitor_urls)
    top_competitor = metric.get("top_competitor_url")
    if top_competitor and top_competitor != "-":
        urls.append((top_competitor, "Cited Competitor URL"))
    deduped = []
    seen = set()
    for url, role in urls:
        normalized = normalize_url(url)
        if normalized and normalized not in seen:
            deduped.append((normalized, role))
            seen.add(normalized)
    return deduped


def _load_html_sources(fixture_path: str | Path | None, cache_dir: str | Path) -> dict[str, dict]:
    html_by_url: dict[str, dict] = {}
    if fixture_path and Path(fixture_path).exists():
        with Path(fixture_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                html_by_url[normalize_url(row["url"])] = {
                    "raw_html": row.get("raw_html", ""),
                    "rendered_html": row.get("rendered_html", ""),
                    "mode": "rendered_dom_fixture" if row.get("rendered_html") else "raw_dom_fixture",
                }
    cache_path = Path(cache_dir)
    if cache_path.exists():
        for path in cache_path.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            url = normalize_url(payload.get("url", ""))
            if not url:
                continue
            html_by_url[url] = {
                "raw_html": payload.get("raw_html", ""),
                "rendered_html": payload.get("rendered_html", ""),
                "mode": "rendered_dom" if payload.get("rendered_html") else "raw_dom",
            }
    return html_by_url


def _empty_alignment() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "query",
            "url",
            "url_role",
            "source_text_mode",
            "best_matching_heading",
            "best_chunk_text",
            "tfidf_score",
            "embedding_score",
            "answer_capsule_present",
            "alignment_gap_vs_top_competitor",
        ]
    )


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "you",
    "your",
    "into",
    "when",
    "where",
    "what",
    "which",
    "should",
    "have",
    "has",
    "had",
    "not",
    "but",
    "can",
    "use",
    "using",
    "best",
    "top",
    "page",
    "pages",
}

