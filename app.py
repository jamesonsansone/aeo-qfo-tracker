"""Streamlit app for fast AI search query fan-out analysis."""

from __future__ import annotations

import io
import importlib.metadata
import zipfile
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

from seo_suite.aggregator import build_query_runs, build_tables
from seo_suite.io import query_targets_from_dataframe, _canonical_column_key
from seo_suite.providers import FixtureProvider, GeminiGroundedSearchProvider, OpenAIWebSearchProvider
from seo_suite.runner import run_queries


FIXTURE_RUNS = "data/fixture_raw_runs.jsonl"


DISPLAY_LABELS = {
    "target_domain": "Target Domain",
    "target_url": "Target URL",
    "acceptable_url_pattern": "Target URL Pattern",
    "target_brand_aliases": "Target Brand Aliases",
    "query_cluster": "Query Cluster",
    "run_index": "Run",
    "provider": "Provider",
    "model": "Model",
    "response_text": "AI Answer",
    "citation_count": "URLs Cited",
    "fanout_query_count": "Fan-Out Queries",
    "mentioned_brands": "Mentioned Brands",
    "brand": "Brand",
    "matched_alias": "Matched Alias",
    "evidence": "Evidence",
    "position": "Position",
    "evidence_runs": "Evidence Runs",
    "seed_sources": "Seed Sources",
    "category": "Category",
    "fanout_position": "Fan-Out Position",
    "fanout_query": "Fan-Out Query",
    "cited_url": "Cited URL",
    "cited_domain": "Cited Domain",
    "original_url": "Original URL",
    "resolved_url": "Resolved URL",
    "resolution_status": "Resolution Status",
    "resolution_method": "Resolution Method",
    "final_status_code": "Final Status",
    "final_domain": "Final Domain",
    "resolution_error": "Resolution Error",
    "source_label": "Source Label",
    "fallback_domain": "Fallback Domain",
    "citation_kind": "Citation Kind",
    "source_type": "Source Type",
    "citation_kinds": "Citation Kinds",
    "citation_position": "Citation Position",
    "is_target_domain": "Target Domain",
    "is_exact_target_url": "Exact Target URL",
    "is_acceptable_target_url": "Matches URL Pattern",
    "total_citations": "Total Citations",
    "unique_cited_urls": "Unique Cited URLs",
    "unresolved_redirects": "Unresolved Redirects",
    "domain_fallback_citations": "Domain Fallback Citations",
    "queries_count": "Queries",
    "query_coverage": "Query Coverage",
    "query_coverage_rate": "Query Coverage %",
    "answer_frequency": "Answer Frequency",
    "answer_frequency_rate": "Answer Frequency %",
    "is_target_brand": "Target Brand",
    "domain_citation_rate": "Domain Citation Rate",
    "inline_citation_rate": "Inline Citation Rate",
    "target_mention_rate": "Mention Rate",
    "approved_page_citation_rate": "URL Pattern Citation Rate",
    "domain_rrf_score": "Domain RRF Score",
    "acceptable_rrf_score": "URL Pattern RRF Score",
    "top_competitor_domain": "Top Competitor Domain",
    "top_competitor_url": "Top Competitor URL",
    "top_brands": "Top Brands",
    "opportunity_score": "Opportunity Score",
}


st.set_page_config(
    page_title="SEO Suite: Query Fan-Out",
    page_icon="",
    layout="wide",
)


def _check_environment() -> None:
    """Warn immediately if the active Python has an openai version too old for this app."""
    try:
        raw = importlib.metadata.version("openai")
        parts = tuple(int(x) for x in raw.split(".")[:2] if x.isdigit())
        openai_ok = parts >= (1, 52)
    except Exception:
        raw = "not installed"
        openai_ok = False

    if not openai_ok:
        st.error(
            f"**Environment mismatch.** openai {raw} is installed under `{sys.executable}` "
            f"(Python {sys.version.split()[0]}), but this app requires openai ≥ 1.52.0.\n\n"
            "The openai/httpx incompatibility will cause every OpenAI run to fail.\n\n"
            "**Fix:** launch the app with the project venv:\n"
            "```\npython -m streamlit run app.py\n```"
        )
        st.stop()


def main() -> None:
    _check_environment()
    with st.sidebar:
        st.title("SEO Suite")
        st.caption("Fast AI-search fan-out visibility for ecommerce queries.")

        workflow = st.radio("Workflow", ["One-off query", "CSV batch"], horizontal=True)
        domain_label = "Target domain" if workflow == "One-off query" else "Default target domain"
        default_target_domain = st.text_input(
            domain_label,
            value="trailgear.example",
            help="Domain-only matching is enough. Any cited URL on this domain or its subdomains is flagged as a target citation.",
            placeholder="us.puma.com",
        )
        default_brand_aliases = st.text_input(
            "Target brand aliases",
            value="TrailGear",
            help="Comma-separated names to count in answer text. Example: Puma, Puma Running.",
        )
        brand_alias_file = st.file_uploader(
            "Brand aliases CSV",
            type=["csv"],
            key="brand_aliases_csv",
            help="Optional extension list with columns `brand`, `aliases`, and optional `category`.",
        )
        with st.expander("Optional URL matching", expanded=False):
            default_target_url = st.text_input(
                "Exact target URL",
                value="",
                help="Optional. Use only if you care whether one exact URL is cited.",
            )
            default_url_pattern = st.text_input(
                "Target URL pattern",
                value="",
                help="Optional glob or regex. Example: https://us.puma.com/us/en/mens/*",
            )
            st.caption("Leave these blank for domain-only analysis. A cited URL like `https://us.puma.com/us/en/mens/clothing` will match target domain `us.puma.com`.")

        query_text = ""
        uploaded_file = None
        if workflow == "One-off query":
            query_text = st.text_area(
                "Query",
                value="best trail running shoes for rocky terrain",
                height=90,
            )
        else:
            uploaded_file = st.file_uploader("Upload query CSV", type=["csv"])
            st.caption("CSV requires `Query` or `query`; optional columns include `Type`, `Strategic Category`, `Domain`, `Target URL`, `URL Pattern`, `Cluster`, `Intent`, and `Priority`.")

        provider_name = st.selectbox("Provider", ["Fixture demo", "Gemini live", "OpenAI live"])
        if provider_name == "Gemini live":
            model = st.selectbox("Gemini model", ["gemini-3.1-flash-lite"])
            api_key = st.text_input("Gemini API key", value="", type="password")
        elif provider_name == "OpenAI live":
            model = st.selectbox("OpenAI model", ["gpt-5", "gpt-5-mini", "gpt-5-nano"])
            api_key = st.text_input("OpenAI API key", value="", type="password")
        else:
            model = "fixture-grounded-search"
            api_key = ""

        runs_count = st.number_input("Runs per query", min_value=1, max_value=8, value=4)
        concurrency = st.slider("Parallel API calls", min_value=1, max_value=8, value=4)
        retries = st.number_input("Retries per failed run", min_value=0, max_value=3, value=1)
        run_button = st.button("Run Fan-Out", type="primary", use_container_width=True)
        with st.expander("Runtime Diagnostics", expanded=False):
            diagnostics = _runtime_diagnostics()
            st.caption(f"Python: `{diagnostics['python_executable']}`")
            st.caption(f"Python version: `{diagnostics['python_version']}`")
            st.caption(f"OpenAI: `{diagnostics['openai_version']}`")
            st.caption(f"httpx: `{diagnostics['httpx_version']}`")
            st.caption(f"google-genai: `{diagnostics['google_genai_version']}`")
            st.caption(f"Streamlit: `{diagnostics['streamlit_version']}`")

    st.title("AI Search Query Fan-Out")
    st.caption("Run one query or a category batch, then inspect citations, fanout queries, answers, and brand mention coverage.")

    try:
        targets, input_df = _load_targets(
            workflow=workflow,
            query_text=query_text,
            uploaded_file=uploaded_file,
            default_target_domain=default_target_domain,
            default_target_url=default_target_url,
            default_url_pattern=default_url_pattern,
            default_brand_aliases=default_brand_aliases,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    _show_input_preview(input_df, targets)

    if run_button:
        missing_domain = [target.query for target in targets if not target.target_domain and not default_target_domain]
        if missing_domain:
            st.error("Every query needs a target domain. Add a default target domain or a `target_domain` column in the CSV.")
            st.stop()
        if not targets:
            st.error("Add at least one query to run.")
            st.stop()

        try:
            provider = _make_provider(provider_name, model, api_key)
            brand_aliases_df = _load_brand_aliases(brand_alias_file)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        progress = st.progress(0, text="Starting fan-out run...")
        status = st.empty()

        def on_progress(completed: int, total: int, result) -> None:
            progress.progress(completed / total, text=f"Completed {completed} of {total} runs")
            state = "failed" if result.error else "completed"
            status.info(f"{state.title()}: {result.query} (run {result.run_index + 1})")

        with st.spinner("Running grounded AI answers..."):
            provider_results = run_queries(
                targets,
                provider,
                int(runs_count),
                max_workers=int(concurrency),
                retries=int(retries),
                on_progress=on_progress,
            )
            query_runs = build_query_runs(provider_results, targets, default_target_domain)
            st.session_state.tables = build_tables(
                targets,
                query_runs,
                default_target_domain,
                brand_aliases=brand_aliases_df,
            )
            st.session_state.targets = targets
            st.session_state.brand_aliases = brand_aliases_df
            st.session_state.provider_name = provider_name
            st.session_state.model = model

        progress.empty()
        status.empty()
        st.success(f"Completed {len(targets)} queries x {int(runs_count)} runs.")

    if "tables" not in st.session_state:
        st.info("Configure a one-off query or upload a CSV batch, then run the fan-out analysis.")
        return

    tables = st.session_state.tables
    _render_summary_metrics(tables)
    _render_tabs(tables)


def _load_targets(
    workflow: str,
    query_text: str,
    uploaded_file,
    default_target_domain: str,
    default_target_url: str,
    default_url_pattern: str,
    default_brand_aliases: str,
):
    if workflow == "CSV batch" and uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding="utf-8-sig").fillna("")
    else:
        df = pd.DataFrame(
            [
                {
                    "query": query_text.strip(),
                    "target_domain": default_target_domain,
                    "target_url": default_target_url,
                    "acceptable_url_pattern": default_url_pattern,
                    "target_brand_aliases": default_brand_aliases,
                }
            ]
        )
    query_columns = [column for column in df.columns if _canonical_column_key(column) == "query"]
    if query_columns:
        df = df[df[query_columns[0]].astype(str).str.strip() != ""].reset_index(drop=True)
    targets = query_targets_from_dataframe(
        df,
        default_target_domain=default_target_domain,
        default_target_url=default_target_url,
        default_acceptable_url_pattern=default_url_pattern,
        default_brand_aliases=default_brand_aliases,
    )
    return targets, df


def _load_brand_aliases(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame(columns=["brand", "aliases", "category"])
    df = pd.read_csv(uploaded_file).fillna("")
    if "brand" not in df.columns:
        raise ValueError("Brand aliases CSV needs a `brand` column.")
    if "aliases" not in df.columns:
        df["aliases"] = ""
    if "category" not in df.columns:
        df["category"] = ""
    return df[["brand", "aliases", "category"]]


def _show_input_preview(input_df: pd.DataFrame, targets) -> None:
    with st.expander("Input Preview", expanded=False):
        if input_df.empty:
            st.caption("No queries loaded yet.")
            return
        preview = pd.DataFrame(
            [
                {
                    "query": target.query,
                    "target_domain": target.target_domain,
                    "target_url": target.target_url,
                    "acceptable_url_pattern": target.acceptable_url_pattern,
                    "target_brand_aliases": target.target_brand_aliases,
                }
                for target in targets
            ]
        )
        st.dataframe(_display(preview), hide_index=True, use_container_width=True)


def _make_provider(provider_name: str, model: str, api_key: str):
    if provider_name == "Fixture demo":
        return FixtureProvider(FIXTURE_RUNS)
    if provider_name == "Gemini live":
        return GeminiGroundedSearchProvider(api_key=api_key or None, model=model)
    return OpenAIWebSearchProvider(api_key=api_key or None, model=model)


def _runtime_diagnostics() -> dict[str, str]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "openai_version": _package_version("openai"),
        "httpx_version": _package_version("httpx"),
        "google_genai_version": _package_version("google-genai"),
        "streamlit_version": _package_version("streamlit"),
    }


def _package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _render_summary_metrics(tables: dict[str, pd.DataFrame]) -> None:
    overview = tables["overview"].iloc[0].to_dict()
    cols = st.columns(6)
    cols[0].metric("Queries", overview.get("total_queries", 0))
    cols[1].metric("Answers", overview.get("total_answers", 0))
    cols[2].metric("Fan-Outs", overview.get("total_fanout_queries", 0))
    cols[3].metric("Cited URLs", overview.get("total_cited_urls", 0))
    cols[4].metric("Target Citations", overview.get("target_citation_answer_frequency", "0/0"))
    cols[5].metric("Target Mentions", overview.get("target_mention_answer_frequency", "0/0"))
    unresolved = tables.get("unresolved_redirects", pd.DataFrame())
    if not unresolved.empty:
        unresolved_count = int((unresolved.get("citation_kind", pd.Series(dtype=str)).astype(str) == "unresolved").sum())
        fallback_count = int((unresolved.get("citation_kind", pd.Series(dtype=str)).astype(str) == "domain_fallback").sum())
        st.caption(
            f"{unresolved_count} unresolved redirect citation(s); "
            f"{fallback_count} domain fallback citation(s) were kept in domain rankings only."
        )


def _render_tabs(tables: dict[str, pd.DataFrame]) -> None:
    tab_summary, tab_brands, tab_urls, tab_matrix, tab_fanouts, tab_answers, tab_raw = st.tabs(
        ["Summary", "Brand Mentions", "Cited URLs", "Query Matrix", "Fan-Outs", "AI Answers", "Raw Runs"]
    )

    with tab_summary:
        st.subheader("Batch Summary")
        summary_cols = [
            "target_domain",
            "total_queries",
            "total_answers",
            "successful_answers",
            "run_errors",
            "total_fanout_queries",
            "total_cited_urls",
            "unique_cited_urls",
            "unique_cited_domains",
            "unresolved_redirects",
            "domain_fallback_citations",
            "target_citation_query_coverage",
            "target_citation_answer_frequency",
            "target_mention_query_coverage",
            "target_mention_answer_frequency",
            "avg_domain_rrf_score",
            "top_competitor_domain",
            "top_brand",
        ]
        st.dataframe(_display(tables["overview"][[col for col in summary_cols if col in tables["overview"].columns]]), hide_index=True, use_container_width=True)
        zip_bytes = _zip_outputs(tables)
        download_col, save_col = st.columns(2)
        with download_col:
            st.download_button(
                "Download tables ZIP",
                data=zip_bytes,
                file_name="seo_suite_fanout_tables.zip",
                mime="application/zip",
                use_container_width=True,
            )
        with save_col:
            if st.button("Save ZIP to Desktop", use_container_width=True):
                try:
                    saved_path = _save_zip_to_desktop(zip_bytes)
                    st.success(f"Saved to {saved_path}")
                except OSError as exc:
                    st.error(f"Could not save ZIP to Desktop: {exc}")

    with tab_brands:
        st.subheader("Detected Brands in AI Answer Text")
        brand_cols = [
            "brand",
            "is_target_brand",
            "query_coverage",
            "query_coverage_count",
            "query_coverage_rate",
            "answer_frequency",
            "answer_frequency_count",
            "answer_frequency_rate",
            "evidence_runs",
            "mentioned_queries",
            "seed_sources",
            "category",
        ]
        st.dataframe(
            _style_target(tables["brand_mentions"][[col for col in brand_cols if col in tables["brand_mentions"].columns]]),
            hide_index=True,
            use_container_width=True,
        )
        st.subheader("Brand Evidence by Run")
        evidence_cols = ["query", "run_index", "brand", "matched_alias", "evidence", "position"]
        evidence = tables.get("brand_run_mentions", pd.DataFrame())
        if evidence.empty:
            st.info("No brand mentions were detected in the answer text.")
        else:
            st.dataframe(
                _style_target(evidence[[col for col in evidence_cols if col in evidence.columns]]),
                hide_index=True,
                use_container_width=True,
            )

    with tab_urls:
        st.subheader("Cited URLs")
        url_cols = [
            "cited_url",
            "cited_domain",
            "category",
            "total_citations",
            "query_coverage",
            "query_coverage_rate",
            "citation_kinds",
            "title",
            "queries_list",
        ]
        st.dataframe(
            _style_target(tables["url_metrics"][[col for col in url_cols if col in tables["url_metrics"].columns]]),
            hide_index=True,
            use_container_width=True,
        )
        st.subheader("Cited Domains")
        domain_cols = [
            "cited_domain",
            "category",
            "total_citations",
            "query_coverage",
            "query_coverage_rate",
            "citation_kinds",
            "queries_list",
        ]
        st.dataframe(
            _style_target(tables["domain_metrics"][[col for col in domain_cols if col in tables["domain_metrics"].columns]]),
            hide_index=True,
            use_container_width=True,
        )
        st.subheader("Citation Detail")
        citation_cols = [
            "query",
            "run_index",
            "citation_position",
            "cited_domain",
            "cited_url",
            "original_url",
            "resolution_status",
            "resolution_method",
            "source_label",
            "fallback_domain",
            "citation_kind",
            "source_type",
            "final_status_code",
            "is_target_domain",
            "is_exact_target_url",
            "is_acceptable_target_url",
            "title",
        ]
        st.dataframe(
            _style_target(tables["citations"][[col for col in citation_cols if col in tables["citations"].columns]]),
            hide_index=True,
            use_container_width=True,
        )
        unresolved = tables.get("unresolved_redirects", pd.DataFrame())
        if not unresolved.empty:
            st.subheader("Redirect Diagnostics")
            unresolved_cols = [
                "query",
                "run_index",
                "citation_position",
                "original_url",
                "title",
                "source_label",
                "fallback_domain",
                "citation_kind",
                "resolution_status",
                "resolution_method",
                "resolution_error",
            ]
            st.dataframe(
                _display(unresolved[[col for col in unresolved_cols if col in unresolved.columns]]),
                hide_index=True,
                use_container_width=True,
            )

    with tab_matrix:
        st.subheader("Query Matrix")
        matrix_cols = [
            "query",
            "query_cluster",
            "target_domain",
            "runs_count",
            "target_mention_rate",
            "domain_citation_rate",
            "inline_citation_rate",
            "approved_page_citation_rate",
            "domain_rrf_score",
            "acceptable_rrf_score",
            "total_citations",
            "unique_cited_urls",
            "fanout_query_count",
            "top_competitor_domain",
            "top_competitor_url",
            "top_brands",
            "opportunity_score",
        ]
        st.dataframe(
            _style_opportunity(tables["query_metrics"][[col for col in matrix_cols if col in tables["query_metrics"].columns]]),
            hide_index=True,
            use_container_width=True,
        )

    with tab_fanouts:
        st.subheader("Provider-Native Fan-Out Queries")
        if tables["fanouts"].empty:
            st.info("No provider-native fanout queries were exposed for this run.")
        else:
            st.dataframe(_display(tables["fanouts"]), hide_index=True, use_container_width=True)

    with tab_answers:
        st.subheader("AI Answers")
        answer_cols = [
            "query",
            "run_index",
            "provider",
            "model",
            "citation_count",
            "fanout_query_count",
            "mentioned_brands",
            "response_text",
            "error",
        ]
        answers = tables["ai_answers"][[col for col in answer_cols if col in tables["ai_answers"].columns]]
        query_options = sorted(tables["ai_answers"]["query"].dropna().unique())
        if query_options:
            selected_query = st.selectbox("Read answers for query", query_options)
            for row in tables["ai_answers"][tables["ai_answers"]["query"] == selected_query].to_dict("records"):
                with st.expander(f"Run {int(row['run_index']) + 1}", expanded=False):
                    if row.get("error"):
                        st.error(row["error"])
                    st.write(row.get("response_text", ""))
        st.subheader("Answer Table")
        st.dataframe(_display(answers), hide_index=True, use_container_width=True)

    with tab_raw:
        st.subheader("Raw Runs")
        st.dataframe(_display(tables["raw_runs"]), hide_index=True, use_container_width=True)
        if not tables["run_errors"].empty:
            st.subheader("Run Errors")
            st.dataframe(_display(tables["run_errors"]), hide_index=True, use_container_width=True)


def _display(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy().rename(columns=DISPLAY_LABELS)


def _style_target(df: pd.DataFrame):
    display_df = _display(df)

    def highlight(row):
        values = {str(value).lower() for value in row.values}
        is_target = "target" in values or "true" in values
        return ["background-color: #e8f5e9; color: #124116" if is_target else "" for _ in row]

    return display_df.style.apply(highlight, axis=1)


def _style_opportunity(df: pd.DataFrame):
    display_df = _display(df)
    score_col = DISPLAY_LABELS.get("opportunity_score", "opportunity_score")
    if score_col not in display_df.columns:
        return display_df

    def color_score(value):
        try:
            score = float(value)
        except (TypeError, ValueError):
            return ""
        if score >= 75:
            return "background-color: #fde2e1; color: #6f1d1b"
        if score >= 50:
            return "background-color: #fff4ce; color: #5c4400"
        return "background-color: #e8f5e9; color: #124116"

    return display_df.style.map(color_score, subset=[score_col])


def _zip_outputs(tables: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, df in tables.items():
            zf.writestr(f"{name}.csv", df.to_csv(index=False))
    return buffer.getvalue()


def _save_zip_to_desktop(zip_bytes: bytes, now: datetime | None = None, desktop: Path | None = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    desktop_path = desktop or Path.home() / "Desktop"
    desktop_path.mkdir(parents=True, exist_ok=True)
    path = desktop_path / f"seo_suite_fanout_{timestamp}.zip"
    counter = 2
    while path.exists():
        path = desktop_path / f"seo_suite_fanout_{timestamp}_{counter}.zip"
        counter += 1
    path.write_bytes(zip_bytes)
    return path


if __name__ == "__main__":
    main()
