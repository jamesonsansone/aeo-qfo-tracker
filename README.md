# AEO Query Fan-Out Tracker

Open-source Streamlit tool for AI search visibility analysis. Run one query or a CSV batch against Gemini or OpenAI, then inspect which brands, domains, URLs, and provider-native search fan-outs appear in generated answers.

Use it to answer questions like:

- Does a target brand appear in AI answers for important commercial queries?
- Is the target domain cited, or are competitors being cited instead?
- Are AI systems citing the intended page, another page on the same site, or no target URL at all?
- Which queries have the largest AI visibility opportunity?

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

Fixture demo mode works without API keys. Live mode requires `GEMINI_API_KEY` or `OPENAI_API_KEY`.

## Environment

```bash
cp .env.example .env
```

Add your keys to `.env`. The file is git-ignored.

## One-Off Query

Use this for a single ad hoc query to quickly check brand visibility for one topic.

1. Set **Workflow** to **One-off query**.
2. Enter the **Target domain** you want to track, such as `example.com`.
3. Enter **Target brand aliases** as comma-separated names, products, or variants.
4. Select **Fixture demo**, **Gemini live**, or **OpenAI live**.
5. If using a live provider, paste the API key into the key field.
6. Choose the model shown for that provider.
7. Set **Runs per query**. Four runs is a good default for a more stable signal.
8. Type the query.
9. Click **Run Fan-Out**.

## CSV Batch

Use this to run a full category, topic, or keyword set.

1. Set **Workflow** to **CSV batch**.
2. Enter the **Default target domain** for rows that do not specify their own domain.
3. Enter global **Target brand aliases**, unless rows provide their own.
4. Select the provider, API key, model, and runs per query.
5. Upload the query CSV.
6. Click **Run Fan-Out**.

### Required CSV Column

| Column | Notes |
| --- | --- |
| `Query` | The search query sent to the AI provider. Also accepts `query`, `Keyword`, `Search Query`, `Seed Query`, `Prompt`, or `Question`. |

### Optional CSV Columns

| Column | Notes |
| --- | --- |
| `Domain` | Target domain for that row. Overrides the sidebar default. |
| `Target URL` | Exact URL to track at the page level. |
| `URL Pattern` | Glob or regex for a site section, such as `https://example.com/category/*`. |
| `Strategic Category` | Groups queries into clusters in the output. |
| `Cluster` | Fallback grouping if Strategic Category is blank. |
| `Intent` | Freeform intent label, such as `commercial comparison` or `product research`. |
| `Priority` | Integer from 1 to 5. Higher priority increases the Opportunity Score. |
| `Type` | Freeform label mapped to the intent field. |
| `Target Brand Aliases` | Row-level brand aliases. Overrides the sidebar default. |

Minimum viable CSV:

```csv
Query
best running shoes for speed training
brand a vs brand b
lightweight running shoes for men
```

## Reading The Results

| Tab | What it shows |
| --- | --- |
| Summary | Total answers, cited URLs, target citation frequency, top competitor, and visibility-rate charts. |
| Brand Mentions | How often each brand appeared in answer text, across queries and runs. |
| Cited URLs | Every URL and domain cited by the provider, with target highlighting. |
| Query Matrix | Per-query citation rate, mention rate, RRF-style position score, top competitors, and opportunity score. |
| Fan-Outs | Search sub-queries exposed by the provider's API metadata. |
| AI Answers | Full answer text per query and run. |
| Raw Runs | Audit log of every run, including errors and grounding metadata. |

Download all generated tables as a ZIP from the Summary tab.

## Interpreting Fan-Outs

The tool uses provider APIs, not the consumer ChatGPT or Gemini web chat interfaces. It captures query fan-outs only when the API exposes search or grounding metadata.

- OpenAI live mode uses the Responses API with the `web_search` tool and parses `web_search_call.action` metadata.
- Gemini live mode uses Google Search grounding and parses grounding metadata such as `webSearchQueries`, `groundingChunks`, and citations.
- Fixture demo mode reads saved example runs and is useful for local demos and tests.

Provider APIs do not guarantee identical behavior to consumer chat products. Treat fan-outs as provider-exposed search metadata for the configured API/model, not as a perfect transcript of what a consumer chat UI would do.

## Opportunity Score

The Query Matrix includes an `opportunity_score` from 0 to 100. It currently combines:

- Target domain visibility gap
- Competitor citation pressure
- Query priority
- A mismatch penalty when the target domain appears but the intended target URL or URL pattern does not

Use the score as a prioritization signal, not a universal ranking metric.

## License

GPL-3.0
