"""CLI for running the AI Search Query Opportunity Matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from seo_suite.aggregator import build_query_runs, build_tables
from seo_suite.content_alignment import build_content_alignment
from seo_suite.io import load_query_targets, write_jsonl, write_outputs
from seo_suite.page_diagnostics import build_page_diagnostics, build_weekly_queue, collect_diagnostic_urls
from seo_suite.providers import FixtureProvider, OpenAIWebSearchProvider
from seo_suite.reporting import generate_brief
from seo_suite.storage import save_run_batch


def run_query_set(args: argparse.Namespace) -> None:
    targets = load_query_targets(args.queries)
    provider = (
        OpenAIWebSearchProvider(model=args.model, delay_seconds=args.delay)
        if args.provider == "openai"
        else FixtureProvider(args.fixture)
    )

    provider_results = []
    for target in targets:
        for run_index in range(args.runs):
            provider_results.append(provider.run_query(target, run_index))

    runs = build_query_runs(provider_results, targets, args.target_domain)
    tables = build_tables(targets, runs, args.target_domain)
    diagnostic_urls = collect_diagnostic_urls(targets, tables)
    tables["page_diagnostics"] = build_page_diagnostics(
        diagnostic_urls,
        fixture_path=args.page_fixtures,
        fetch_live=args.fetch_live_pages,
        render_live=args.render_live_pages,
        cache_dir=args.page_cache_dir,
        cache_ttl_hours=args.page_cache_ttl_hours,
        pagespeed_api_key=args.pagespeed_api_key,
    )
    tables["content_alignment"] = build_content_alignment(
        targets,
        tables,
        fixture_path=args.page_fixtures,
        cache_dir=args.page_cache_dir,
    )
    tables["weekly_queue"] = build_weekly_queue(targets, tables)
    brief = generate_brief(tables, args.target_domain)
    if args.save_to_sqlite:
        batch_id = save_run_batch(
            args.db_path,
            args.query_set_name,
            args.target_domain,
            tables,
            targets,
            provider=args.provider,
            model=args.model,
            runs_per_query=args.runs,
            notes=args.notes,
        )
        print(f"Saved SQLite run batch {batch_id} to {args.db_path}")

    raw_rows = [
        {
            "query": result.query,
            "run_index": result.run_index,
            "provider": result.provider,
            "model": result.model,
            "response_text": result.response_text,
            "sources": result.sources,
        }
        for result in provider_results
    ]

    output_dir = Path(args.output)
    write_jsonl(output_dir / "raw_runs.jsonl", raw_rows)
    write_outputs(output_dir, tables, brief)
    print(f"Wrote query opportunity outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Search Query Opportunity Matrix")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-query-set", help="Run a mapped query set")
    run_parser.add_argument("--queries", default="data/sample_queries.csv")
    run_parser.add_argument("--target-domain", required=True)
    run_parser.add_argument("--provider", choices=["fixture", "openai"], default="fixture")
    run_parser.add_argument("--fixture", default="data/fixture_raw_runs.jsonl")
    run_parser.add_argument("--runs", type=int, default=4)
    run_parser.add_argument("--model", default="gpt-5")
    run_parser.add_argument("--delay", type=float, default=1.0)
    run_parser.add_argument("--page-fixtures", default="data/fixture_pages.jsonl")
    run_parser.add_argument("--fetch-live-pages", action="store_true")
    run_parser.add_argument("--render-live-pages", action="store_true")
    run_parser.add_argument("--page-cache-dir", default=".cache/page_diagnostics")
    run_parser.add_argument("--page-cache-ttl-hours", type=int, default=168)
    run_parser.add_argument("--pagespeed-api-key", default=None)
    run_parser.add_argument("--save-to-sqlite", action="store_true")
    run_parser.add_argument("--db-path", default="data/seo_suite.db")
    run_parser.add_argument("--query-set-name", default="Sample Ecommerce Query Set")
    run_parser.add_argument("--notes", default="")
    run_parser.add_argument("--output", default="outputs/demo_run")
    run_parser.set_defaults(func=run_query_set)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
