from seo_suite.aggregator import build_query_runs, build_tables
from seo_suite.content_alignment import build_content_alignment
from seo_suite.models import ProviderResult, QueryTarget
from seo_suite.page_diagnostics import build_page_diagnostics, build_weekly_queue, collect_diagnostic_urls
from seo_suite.storage import (
    build_daily_trends,
    build_historical_queue,
    build_weekly_rollups,
    init_db,
    load_recent_batches,
    save_run_batch,
)


def _fixture_tables():
    targets = [
        QueryTarget(
            query="best trail running shoes for flat feet",
            query_cluster="trail running shoes",
            intent="commercial comparison",
            target_url="https://trailgear.example/products/stability-trail-runner",
            acceptable_url_pattern="https://trailgear.example/products/*trail*",
            priority=5,
        )
    ]
    results = [
        ProviderResult(
            query="best trail running shoes for flat feet",
            run_index=0,
            provider="fixture",
            model="fixture",
            response_text="TrailGear Stability Trail Runner is relevant for flat feet stability trail running shoes.",
            sources=[
                {"url": "https://trailgear.example/products/stability-trail-runner", "title": "Stability Trail Runner"},
                {"url": "https://www.runnersworld.com/gear/a20865799/best-trail-running-shoes/", "title": "Best Trail Running Shoes"},
            ],
        )
    ]
    runs = build_query_runs(results, targets, "trailgear.example")
    tables = build_tables(targets, runs, "trailgear.example")
    tables["page_diagnostics"] = build_page_diagnostics(
        collect_diagnostic_urls(targets, tables),
        fixture_path="data/fixture_pages.jsonl",
    )
    tables["content_alignment"] = build_content_alignment(targets, tables, fixture_path="data/fixture_pages.jsonl")
    tables["weekly_queue"] = build_weekly_queue(targets, tables)
    return targets, tables


def test_sqlite_schema_idempotent(tmp_path):
    db_path = tmp_path / "seo_suite.db"
    init_db(str(db_path))
    init_db(str(db_path))
    assert db_path.exists()


def test_save_and_reload_fixture_batch(tmp_path):
    db_path = tmp_path / "seo_suite.db"
    targets, tables = _fixture_tables()
    batch_id = save_run_batch(
        str(db_path),
        "Fixture Set",
        "trailgear.example",
        tables,
        targets,
        provider="fixture",
        model="fixture",
        runs_per_query=1,
        run_date="2026-05-01",
    )
    assert batch_id > 0
    recent = load_recent_batches(str(db_path), "Fixture Set", days=56)
    assert len(recent["run_batches"]) == 1
    assert len(recent["query_metrics"]) == 1
    assert len(recent["content_alignment"]) >= 1
    assert len(recent["weekly_queue"]) == 1


def test_daily_weekly_and_historical_queue(tmp_path):
    db_path = tmp_path / "seo_suite.db"
    targets, tables = _fixture_tables()
    save_run_batch(str(db_path), "Fixture Set", "trailgear.example", tables, targets, run_date="2026-04-24")
    save_run_batch(str(db_path), "Fixture Set", "trailgear.example", tables, targets, run_date="2026-05-01")

    daily = build_daily_trends(str(db_path), "Fixture Set")
    weekly = build_weekly_rollups(str(db_path), "Fixture Set", weeks=8)
    historical = build_historical_queue(str(db_path), "Fixture Set")

    assert len(daily) == 2
    assert not weekly.empty
    assert not historical.empty
    assert "issue_status" in historical.columns

