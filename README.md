# AEO Query Fan-Out Tracker

Streamlit tool for AI search visibility analysis. Run one query or a CSV batch against Gemini or OpenAI, then inspect which domains, URLs, and brands appear in the generated answers.

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

## Input

Upload a CSV with a `Query` column. Optional columns: `Domain`, `Target URL`, `URL Pattern`, `Cluster`, `Intent`, `Priority`, `Strategic Category`.

## License

GPL-3.0
