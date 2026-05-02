# SEO Suite: Ecommerce Query Comparison Matrix

SEO Suite is a Streamlit application for bulk ecommerce query comparison across grounded AI-search results. It helps teams upload a mapped set of category, product, and comparison queries, run repeated grounded checks, and identify where a target domain is visible, missing, cited in the wrong place, or losing citations to recurring competitors.

The app is designed for repeatable category-level review: start with a query-to-target map, run several checks per query, compare cited URLs, and turn the results into a prioritized follow-up queue.

## Features

- Bulk query-set testing with repeated runs per query
- Exact target URL citation tracking
- Approved page pattern matching for PDP, PLP, and category URL variants
- Target-domain mention and citation rates
- Reciprocal Rank Fusion score for cited target positions
- Competitor domain and URL aggregation
- Query-level opportunity scoring
- Raw HTML and rendered HTML page diagnostics
- Product and Offer schema checks in raw and rendered HTML
- Title, H1, canonical, link-count, and word-count comparison
- Optional PageSpeed Insights metrics for LCP, INP, CLS, and performance score
- Exact-text content alignment between AI answers and cited page text
- SQLite snapshots for daily tracking and 8-week trend views
- Markdown opportunity brief generation

## App Views

- **Overview**: query bucket score, visibility rates, zero-visibility queries, competitors, and top opportunities
- **Query Matrix**: one row per mapped query with target visibility, approved page visibility, RRF score, competitors, and opportunity score
- **Citation Detail**: run-by-run cited URLs and answer text for a selected query
- **URL Diagnostics**: raw/rendered page checks for mapped target URLs and cited competitor URLs
- **Content Alignment**: exact-text TF-IDF comparison between answer text and page sections
- **Daily Tracker**: saved run batches and daily visibility metrics
- **8-Week Trends**: weekly rollups for visibility, citations, competitors, and recurring issues
- **Historical Weekly Queue**: current and prior follow-up items
- **Competitor Patterns**: recurring domains and URLs across the query set
- **Opportunity Brief**: Markdown summary for review and follow-up planning

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens with fixture data, so it can be evaluated without API credentials.

## Streamlit Community Cloud

Recommended app entry point:

```text
app.py
```

Add secrets in Streamlit Cloud instead of committing local `.env` values:

```toml
OPENAI_API_KEY = "your_openai_key"
PAGESPEED_API_KEY = "optional_pagespeed_key"
SEO_SUITE_DB_PATH = "data/seo_suite.db"
```

Fixture mode works without secrets. Live OpenAI mode requires `OPENAI_API_KEY`.

SQLite is useful for local runs and single-server deployments. On Streamlit Community Cloud, local files may reset between app restarts, so durable shared history should eventually use an external database or exported run files.

## Query Map Schema

Upload a CSV with these required columns:

```text
query
query_cluster
intent
target_url
acceptable_url_pattern
```

Optional:

```text
priority
```

`acceptable_url_pattern` defines approved URL variants for a query. It supports glob-style matching:

```text
https://example.com/products/*
https://example.com/collections/*hiking*
```

This is useful when a query can reasonably cite more than one valid PDP, PLP, or category URL.

## CLI Usage

Run the included fixture query set:

```bash
python cli.py run-query-set \
  --queries data/sample_queries.csv \
  --target-domain trailgear.example \
  --provider fixture \
  --runs 4 \
  --output outputs/demo_run
```

Save a run into SQLite for daily tracking:

```bash
python cli.py run-query-set \
  --queries data/sample_queries.csv \
  --target-domain trailgear.example \
  --provider fixture \
  --runs 4 \
  --save-to-sqlite \
  --db-path data/seo_suite.db \
  --query-set-name "Sample Ecommerce Query Set" \
  --output outputs/demo_run
```

Run live grounded checks with OpenAI:

```bash
export OPENAI_API_KEY=your_key_here
python cli.py run-query-set \
  --queries data/sample_queries.csv \
  --target-domain example.com \
  --provider openai \
  --runs 4 \
  --output outputs/live_run
```

## Page Diagnostics

Live page diagnostics are optional and cached under `.cache/page_diagnostics` by default. The app intentionally runs search fanout first, then lets you select which target or cited URLs should receive slower page-level checks.

Raw fetch diagnostics use standard HTTP requests. Rendered diagnostics use Playwright when enabled. Playwright is open source and has no per-page API cost, but it requires local browser binaries:

```bash
python -m playwright install chromium
```

Run live page fetching and rendering from the CLI:

```bash
python cli.py run-query-set \
  --queries data/sample_queries.csv \
  --target-domain example.com \
  --provider openai \
  --fetch-live-pages \
  --render-live-pages \
  --output outputs/live_run
```

Diagnostics include:

- raw HTML fetch status
- rendered HTML fetch status
- Product schema in raw and rendered HTML
- Offer schema in raw and rendered HTML
- title, H1, and canonical changes
- internal link count
- external outlink count
- raw and rendered word counts
- optional LCP, INP, CLS, and PageSpeed performance score

In the Streamlit app, use **Run Search Fanout** first. Then open **URL Diagnostics**, select the URLs to check, choose raw fetch, Playwright rendering, and PageSpeed options, and run diagnostics separately.

## Content Alignment

Content alignment uses exact text extracted from raw or rendered DOM HTML. It does not use summaries as the canonical text source.

Canonical text sources:

1. fixture HTML in `data/fixture_pages.jsonl`
2. cached raw/rendered HTML from live page diagnostics

The app chunks page text by headings and page sections, then compares AI answer text against those chunks with deterministic TF-IDF cosine similarity. This helps identify whether the cited answer appears better aligned with the target URL or with competing URLs.

Fallback content providers can be added later, but their output should be clearly labeled so transformed text is not mixed with exact DOM-based scores.

## Persistence

SQLite is the default persistence layer for local tracking. Saved runs include:

- query sets
- query targets
- run batches
- individual runs
- citations
- query metric snapshots
- page diagnostic snapshots
- content alignment snapshots
- weekly queue items

Generated SQLite files are ignored by git through `data/*.db`, `data/*.sqlite`, and `data/*.sqlite3`.

## Environment Variables

Copy `.env.example` to `.env` for local use:

```bash
cp .env.example .env
```

Available variables:

```text
OPENAI_API_KEY
PAGESPEED_API_KEY
SEO_SUITE_DB_PATH
```

Never commit `.env`, Streamlit secrets, generated SQLite databases, cached diagnostics, or output exports.

## License

This project is licensed under the GNU General Public License v3.0. See `LICENSE` for details.

## Measurement Notes

This is AI-search visibility measurement, not traditional rank tracking. Repeated runs are useful because grounded AI-search results can vary over time. Citation order, mention rate, approved page citation rate, competitor recurrence, page diagnostics, and content alignment should be interpreted together as directional signals for ecommerce query visibility.
