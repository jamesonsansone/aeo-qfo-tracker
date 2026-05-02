"""Streamlit app for the SEO Suite AI Search Query Opportunity Matrix."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from seo_suite.aggregator import build_query_runs, build_tables, make_citation_matrix
from seo_suite.content_alignment import build_content_alignment
from seo_suite.io import load_query_targets
from seo_suite.matching import normalize_url
from seo_suite.page_diagnostics import (
    build_page_diagnostics,
    build_weekly_queue,
    collect_diagnostic_urls,
)
from seo_suite.providers import FixtureProvider, OpenAIWebSearchProvider
from seo_suite.reporting import generate_brief
from seo_suite.storage import (
    build_daily_trends,
    build_historical_queue,
    build_weekly_rollups,
    save_run_batch,
)


SAMPLE_QUERIES = "data/sample_queries.csv"
FIXTURE_RUNS = "data/fixture_raw_runs.jsonl"
PAGE_FIXTURES = "data/fixture_pages.jsonl"


DISPLAY_LABELS = {
    "citation_count": "Source URLs Cited",
    "acceptable_url_pattern": "Approved Page Pattern",
    "acceptable_target_presence_rate": "Approved Page Visibility",
    "acceptable_mrr": "Approved Page RRF Score (Reciprocal Rank Fusion)",
    "avg_acceptable_presence_rate": "Avg Approved Page Visibility",
    "avg_acceptable_mrr": "Avg Approved Page RRF Score (Reciprocal Rank Fusion)",
    "target_mention_rate": "Mention Rate",
    "domain_citation_rate": "Domain Citation Rate",
    "exact_url_citation_rate": "Exact URL Citation Rate",
    "approved_page_citation_rate": "Approved Page Citation Rate",
    "target_mismatch": "Wrong Client Page Cited",
    "target_mismatch_queries": "Wrong-Page Queries",
    "exact_target_url_presence_rate": "Exact URL Visibility",
    "target_domain_presence_rate": "Domain Visibility",
    "best_acceptable_position": "Best Approved Page Position",
    "is_acceptable_target_url": "Approved Page Match",
    "is_exact_target_url": "Exact URL Match",
    "is_target_domain": "Target Domain Match",
    "raw_html_available": "Raw HTML Available",
    "rendered_html_available": "Rendered HTML Available",
    "product_schema_raw": "Product Schema in Raw HTML",
    "product_schema_rendered": "Product Schema After Rendering",
    "offer_schema_raw": "Offer Schema in Raw HTML",
    "offer_schema_rendered": "Offer Schema After Rendering",
    "raw_render_delta_type": "Raw vs Rendered Difference",
    "title_changed": "Title Changed After Rendering",
    "h1_changed": "H1 Changed After Rendering",
    "canonical_changed": "Canonical Changed After Rendering",
    "internal_links_count": "Internal Links",
    "external_outlinks_count": "External Outlinks",
    "word_count_raw": "Raw Word Count",
    "word_count_rendered": "Rendered Word Count",
    "pagespeed_performance_score": "PageSpeed Performance Score",
    "field_data_available": "Field Data Available",
    "diagnostic_flags": "Diagnostic Flags",
    "recommended_follow_up": "Recommended Follow-Up",
    "approved_page_visibility": "Approved Page Visibility",
    "approved_page_position_score": "Approved Page RRF Score (Reciprocal Rank Fusion)",
    "wrong_client_page_cited": "Wrong Client Page Cited",
    "weekly_priority_score": "Weekly Priority Score",
    "url_role": "URL Role",
    "tfidf_score": "TF-IDF Alignment Score",
    "embedding_score": "Embedding Score",
    "best_matching_heading": "Best Matching Section",
    "best_chunk_text": "Best Matching Text",
    "answer_capsule_present": "Answer Capsule Present",
    "alignment_gap_vs_top_competitor": "Alignment Gap vs Top Competitor",
    "source_text_mode": "Source Text Mode",
    "issue_status": "Issue Status",
    "run_date": "Run Date",
}


st.set_page_config(
    page_title="SEO Suite: AI Opportunity Matrix",
    page_icon="",
    layout="wide",
)


def _load_targets(uploaded_file) -> tuple[list, pd.DataFrame]:
    if uploaded_file is None:
        targets = load_query_targets(SAMPLE_QUERIES)
        df = pd.read_csv(SAMPLE_QUERIES)
        return targets, df
    df = pd.read_csv(uploaded_file).fillna("")
    temp = Path("outputs/_uploaded_queries.csv")
    temp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(temp, index=False)
    return load_query_targets(temp), df


@st.cache_data(show_spinner=False)
def _run_fixture(target_domain: str, runs_count: int) -> dict[str, pd.DataFrame]:
    targets = load_query_targets(SAMPLE_QUERIES)
    provider = FixtureProvider(FIXTURE_RUNS)
    provider_results = [
        provider.run_query(target, run_index)
        for target in targets
        for run_index in range(runs_count)
    ]
    runs = build_query_runs(provider_results, targets, target_domain)
    tables = build_tables(targets, runs, target_domain)
    tables["page_diagnostics"] = build_page_diagnostics(
        collect_diagnostic_urls(targets, tables),
        fixture_path=PAGE_FIXTURES,
        fetch_live=False,
    )
    tables["content_alignment"] = build_content_alignment(targets, tables, fixture_path=PAGE_FIXTURES)
    tables["weekly_queue"] = build_weekly_queue(targets, tables)
    return tables


def _run_analysis(
    targets,
    target_domain: str,
    provider_name: str,
    runs_count: int,
    model: str,
    fetch_live_pages: bool,
    render_live_pages: bool,
    pagespeed_api_key: str,
) -> dict[str, pd.DataFrame]:
    if provider_name == "Fixture demo":
        if len(targets) == len(load_query_targets(SAMPLE_QUERIES)) and all(
            target.query == sample.query for target, sample in zip(targets, load_query_targets(SAMPLE_QUERIES))
        ):
            return _run_fixture(target_domain, runs_count)
        provider = FixtureProvider(FIXTURE_RUNS)
    else:
        provider = OpenAIWebSearchProvider(model=model)

    provider_results = []
    progress = st.progress(0, text="Starting analysis...")
    total = len(targets) * runs_count
    completed = 0
    for target in targets:
        for run_index in range(runs_count):
            provider_results.append(provider.run_query(target, run_index))
            completed += 1
            progress.progress(completed / total, text=f"Completed {completed} of {total} runs")
    progress.empty()
    runs = build_query_runs(provider_results, targets, target_domain)
    tables = build_tables(targets, runs, target_domain)
    tables["page_diagnostics"] = build_page_diagnostics(
        collect_diagnostic_urls(targets, tables),
        fixture_path=PAGE_FIXTURES,
        fetch_live=fetch_live_pages,
        render_live=render_live_pages,
        pagespeed_api_key=pagespeed_api_key or None,
    )
    tables["content_alignment"] = build_content_alignment(
        targets,
        tables,
        fixture_path=PAGE_FIXTURES,
    )
    tables["weekly_queue"] = build_weekly_queue(targets, tables)
    return tables


def _zip_outputs(tables: dict[str, pd.DataFrame], brief: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, df in tables.items():
            zf.writestr(f"{name}.csv", df.to_csv(index=False))
        zf.writestr("client_brief.md", brief)
    return buffer.getvalue()


def _metric_value(tables: dict[str, pd.DataFrame], key: str, default="-"):
    overview = tables.get("overview", pd.DataFrame())
    if overview.empty or key not in overview.columns:
        return default
    return overview.iloc[0][key]


def _style_opportunity(df: pd.DataFrame, score_col: str = "opportunity_score"):
    def color_score(value):
        try:
            score = float(value)
        except (TypeError, ValueError):
            return ""
        if score >= 75:
            return "background-color: #f8d7da; color: #721c24"
        if score >= 50:
            return "background-color: #fff3cd; color: #6c4a00"
        return "background-color: #d4edda; color: #155724"

    return df.style.map(color_score, subset=[score_col])


def _display(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    for column in ["best_exact_position", "best_acceptable_position", "best_domain_position"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].fillna("-").astype(str)
    return display_df.rename(columns=DISPLAY_LABELS)


def _style_display_opportunity(df: pd.DataFrame, score_col: str = "opportunity_score"):
    display_df = _display(df)
    display_score = DISPLAY_LABELS.get(score_col, score_col)
    return _style_opportunity(display_df, display_score)


def _comparison_for_query(tables: dict[str, pd.DataFrame], targets, selected_query: str) -> pd.DataFrame:
    target_by_query = {target.query: target for target in targets}
    target = target_by_query[selected_query]
    metrics = tables["query_metrics"].set_index("query").loc[selected_query]
    urls = [target.target_url]
    competitor = metrics.get("top_competitor_url", "")
    if competitor and competitor != "-":
        urls.append(competitor)
    cited = tables["citations"][tables["citations"]["query"] == selected_query]
    competitor_urls = cited.loc[~cited["is_target_domain"], "cited_url"].drop_duplicates().head(3).tolist()
    urls.extend(url for url in competitor_urls if url not in urls)
    diagnostics = tables["page_diagnostics"].copy()
    diagnostics["_normalized"] = diagnostics["url"].apply(normalize_url)
    comparison = diagnostics[diagnostics["_normalized"].isin({normalize_url(url) for url in urls})].drop(columns=["_normalized"])
    comparison.insert(
        0,
        "url_role",
        comparison["url"].apply(lambda url: "Mapped Target URL" if normalize_url(url) == normalize_url(target.target_url) else "Cited Competitor URL"),
    )
    return comparison


def main() -> None:
    with st.sidebar:
        st.title("AI Opportunity Matrix")
        st.caption("Grounded AI-search visibility testing for mapped SEO query sets.")
        target_domain = st.text_input("Target domain", value="trailgear.example")
        provider_name = st.selectbox("Provider", ["Fixture demo", "OpenAI live"])
        runs_count = st.number_input("Runs per query", min_value=1, max_value=10, value=4)
        model = st.selectbox("OpenAI model", ["gpt-5", "gpt-5-nano"])
        fetch_live_pages = st.checkbox("Fetch live page diagnostics", value=False)
        render_live_pages = st.checkbox("Render pages with Playwright", value=False)
        pagespeed_api_key = st.text_input("PageSpeed API key", value="", type="password")
        save_to_sqlite = st.checkbox("Save run to SQLite", value=False)
        db_path = st.text_input("SQLite DB path", value="data/seo_suite.db")
        query_set_name = st.text_input("Query set name", value="Sample Ecommerce Query Set")
        uploaded_file = st.file_uploader("Upload query map CSV", type=["csv"])
        run_button = st.button("Run Analysis", type="primary", use_container_width=True)

        st.divider()
        st.caption("Required columns: query, query_cluster, intent, target_url, acceptable_url_pattern. Optional: priority.")

    st.title("SEO Suite: AI Search Query Opportunity Matrix")
    st.caption("Map queries to target URLs, run grounded tests, and see where the client wins, loses, or shows the wrong page.")

    targets, input_df = _load_targets(uploaded_file)

    if "tables" not in st.session_state or run_button:
        try:
            with st.spinner("Running query opportunity analysis..."):
                st.session_state.tables = _run_analysis(
                    targets,
                    target_domain,
                    provider_name,
                    int(runs_count),
                    model,
                    fetch_live_pages,
                    render_live_pages,
                    pagespeed_api_key,
                )
                st.session_state.target_domain = target_domain
                if save_to_sqlite:
                    st.session_state.last_batch_id = save_run_batch(
                        db_path,
                        query_set_name,
                        target_domain,
                        st.session_state.tables,
                        targets,
                        provider=provider_name,
                        model=model,
                        runs_per_query=int(runs_count),
                    )
                    st.session_state.db_path = db_path
                    st.session_state.query_set_name = query_set_name
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    tables = st.session_state.tables
    active_target = st.session_state.get("target_domain", target_domain)
    brief = generate_brief(tables, active_target)

    overview_cols = st.columns(5)
    overview_cols[0].metric("Queries", _metric_value(tables, "queries"))
    overview_cols[1].metric("Avg Opportunity", _metric_value(tables, "avg_opportunity_score"))
    overview_cols[2].metric("Approved Page Visibility", f"{_metric_value(tables, 'avg_acceptable_presence_rate')}%")
    overview_cols[3].metric("Approved RRF Score", _metric_value(tables, "avg_acceptable_mrr"))
    overview_cols[4].metric("Zero Visibility", _metric_value(tables, "zero_visibility_queries"))

    tab_overview, tab_matrix, tab_detail, tab_diagnostics, tab_alignment, tab_daily, tab_trends, tab_history, tab_queue, tab_competitors, tab_brief = st.tabs(
        [
            "Overview",
            "Query Matrix",
            "Citation Detail",
            "URL Diagnostics",
            "Content Alignment",
            "Daily Tracker",
            "8-Week Trends",
            "Historical Weekly Queue",
            "Weekly Queue",
            "Competitor Patterns",
            "Opportunity Brief",
        ]
    )

    with tab_overview:
        st.subheader("Query Bucket Summary")
        st.dataframe(_display(tables["overview"]), hide_index=True, use_container_width=True)
        st.subheader("Highest Opportunity Queries")
        columns = [
            "query",
            "query_cluster",
            "intent",
            "target_mention_rate",
            "approved_page_citation_rate",
            "acceptable_target_presence_rate",
            "acceptable_mrr",
            "top_competitor_domain",
            "opportunity_score",
        ]
        st.dataframe(
            _style_display_opportunity(tables["query_metrics"][columns].head(10)),
            hide_index=True,
            use_container_width=True,
        )
        st.subheader("Citation Matrix by Domain")
        st.dataframe(make_citation_matrix(tables["citations"], "cited_domain"), hide_index=True, use_container_width=True)

    with tab_matrix:
        st.subheader("Mapped Query Performance")
        matrix_cols = [
            "query",
            "query_cluster",
            "intent",
            "priority",
            "target_url",
            "acceptable_url_pattern",
            "target_domain_presence_rate",
            "exact_target_url_presence_rate",
            "acceptable_target_presence_rate",
            "target_mention_rate",
            "domain_citation_rate",
            "exact_url_citation_rate",
            "approved_page_citation_rate",
            "best_acceptable_position",
            "acceptable_mrr",
            "top_competitor_url",
            "target_mismatch",
            "opportunity_score",
        ]
        st.dataframe(_style_display_opportunity(tables["query_metrics"][matrix_cols]), hide_index=True, use_container_width=True)

    with tab_detail:
        query_options = tables["query_metrics"]["query"].tolist()
        selected_query = st.selectbox("Query", query_options)
        st.subheader("Runs")
        st.dataframe(_display(tables["runs"][tables["runs"]["query"] == selected_query]), hide_index=True, use_container_width=True)
        st.subheader("Citations")
        detail_cols = [
            "run_index",
            "citation_position",
            "cited_domain",
            "cited_url",
            "is_target_domain",
            "is_exact_target_url",
            "is_acceptable_target_url",
            "title",
        ]
        st.dataframe(
            _display(tables["citations"][tables["citations"]["query"] == selected_query][detail_cols]),
            hide_index=True,
            use_container_width=True,
        )
        st.subheader("URL Comparison")
        comparison_cols = [
            "url_role",
            "url",
            "domain",
            "diagnostic_flags",
            "raw_render_delta_type",
            "product_schema_raw",
            "product_schema_rendered",
            "offer_schema_raw",
            "offer_schema_rendered",
            "word_count_raw",
            "word_count_rendered",
            "internal_links_count",
            "external_outlinks_count",
            "lcp",
            "inp",
            "cls",
            "pagespeed_performance_score",
        ]
        comparison = _comparison_for_query(tables, targets, selected_query)
        st.dataframe(_display(comparison[[col for col in comparison_cols if col in comparison.columns]]), hide_index=True, use_container_width=True)
        with st.expander("Answer Text"):
            for row in tables["runs"][tables["runs"]["query"] == selected_query].to_dict("records"):
                st.markdown(f"**Run {row['run_index']}**")
                st.write(row["response_text"])

    with tab_diagnostics:
        st.subheader("URL Diagnostics")
        diagnostic_cols = [
            "url",
            "domain",
            "fetch_status",
            "diagnostic_flags",
            "raw_render_delta_type",
            "product_schema_raw",
            "product_schema_rendered",
            "offer_schema_raw",
            "offer_schema_rendered",
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
        ]
        diagnostics = tables["page_diagnostics"]
        st.dataframe(_display(diagnostics[[col for col in diagnostic_cols if col in diagnostics.columns]]), hide_index=True, use_container_width=True)

    with tab_alignment:
        st.subheader("Content Alignment")
        alignment = tables.get("content_alignment", pd.DataFrame())
        if alignment.empty:
            st.info("No content alignment rows are available yet.")
        else:
            selected_alignment_query = st.selectbox("Alignment Query", alignment["query"].drop_duplicates().tolist())
            alignment_cols = [
                "query",
                "url_role",
                "url",
                "source_text_mode",
                "tfidf_score",
                "embedding_score",
                "best_matching_heading",
                "best_chunk_text",
                "answer_capsule_present",
                "alignment_gap_vs_top_competitor",
            ]
            st.dataframe(
                _display(alignment[alignment["query"] == selected_alignment_query][alignment_cols]),
                hide_index=True,
                use_container_width=True,
            )

    with tab_daily:
        st.subheader("Daily Tracker")
        active_db_path = st.session_state.get("db_path", "data/seo_suite.db")
        active_query_set = st.session_state.get("query_set_name", "Sample Ecommerce Query Set")
        daily = build_daily_trends(active_db_path, active_query_set)
        if daily.empty:
            st.info("No saved SQLite batches yet. Enable 'Save run to SQLite' and run an analysis.")
        else:
            st.line_chart(daily.set_index("run_date")[["mention_rate", "approved_page_citation_rate", "approved_page_rrf_score"]])
            st.dataframe(_display(daily), hide_index=True, use_container_width=True)

    with tab_trends:
        st.subheader("8-Week Trends")
        active_db_path = st.session_state.get("db_path", "data/seo_suite.db")
        active_query_set = st.session_state.get("query_set_name", "Sample Ecommerce Query Set")
        weekly = build_weekly_rollups(active_db_path, active_query_set, weeks=8)
        if weekly.empty:
            st.info("No weekly rollups available yet.")
        else:
            st.line_chart(weekly.set_index("week_start")[["mention_rate", "approved_page_citation_rate", "approved_page_rrf_score"]])
            st.dataframe(_display(weekly), hide_index=True, use_container_width=True)

    with tab_history:
        st.subheader("Historical Weekly Queue")
        active_db_path = st.session_state.get("db_path", "data/seo_suite.db")
        active_query_set = st.session_state.get("query_set_name", "Sample Ecommerce Query Set")
        historical_queue = build_historical_queue(active_db_path, active_query_set)
        if historical_queue.empty:
            st.info("No saved historical queue items yet.")
        else:
            history_cols = [
                "query",
                "query_cluster",
                "recommended_follow_up",
                "issue_status",
                "weekly_priority_score",
                "diagnostic_flags",
            ]
            st.dataframe(
                _style_display_opportunity(historical_queue[[col for col in history_cols if col in historical_queue.columns]], "weekly_priority_score"),
                hide_index=True,
                use_container_width=True,
            )

    with tab_queue:
        st.subheader("Weekly Follow-Up Queue")
        queue_cols = [
            "query",
            "query_cluster",
            "recommended_follow_up",
            "target_url",
            "top_competitor_url",
            "approved_page_visibility",
            "approved_page_position_score",
            "wrong_client_page_cited",
            "opportunity_score",
            "diagnostic_flags",
            "weekly_priority_score",
        ]
        queue = tables["weekly_queue"]
        st.dataframe(_style_display_opportunity(queue[[col for col in queue_cols if col in queue.columns]], "weekly_priority_score"), hide_index=True, use_container_width=True)

    with tab_competitors:
        st.subheader("Recurring Competitor Domains")
        domain_df = tables["domain_metrics"]
        st.dataframe(domain_df[domain_df["category"] != "Target"], hide_index=True, use_container_width=True)
        st.subheader("Recurring Competitor URLs")
        url_df = tables["url_metrics"]
        st.dataframe(url_df[url_df["category"] != "Target"], hide_index=True, use_container_width=True)

    with tab_brief:
        st.subheader("Client-Ready Brief")
        st.markdown(brief)
        st.download_button(
            "Download outputs ZIP",
            data=_zip_outputs(tables, brief),
            file_name="seo_suite_opportunity_matrix.zip",
            mime="application/zip",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
