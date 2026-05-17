"""Deterministic brand mention extraction for AI answer text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from seo_suite.matching import extract_domain, normalize_host
from seo_suite.models import QueryRun, QueryTarget


GENERIC_DOMAIN_LABELS = {
    "www",
    "m",
    "shop",
    "store",
    "blog",
    "learn",
    "support",
    "help",
}

COMPOUND_PUBLIC_SUFFIX_LABELS = {"co", "com", "net", "org", "ac", "edu", "gov"}
COMPOUND_PUBLIC_SUFFIX_TLDS = {"uk", "au", "nz", "jp", "br", "mx", "za", "kr", "in"}

BUILTIN_BRANDS = [
    {"brand": "Puma", "aliases": "Puma, Puma Running, Puma Nitro, Nitro Elite", "category": "running"},
    {"brand": "Nike", "aliases": "Nike, Nike Running, Vaporfly, Alphafly, Pegasus, ZoomX", "category": "running"},
    {"brand": "Adidas", "aliases": "Adidas, Adidas Running, Adizero, Lightstrike, Terrex", "category": "running"},
    {"brand": "ASICS", "aliases": "ASICS, Asics, Metaspeed, Gel-Nimbus, Gel Kayano", "category": "running"},
    {"brand": "Brooks", "aliases": "Brooks, Brooks Running, Cascadia, Hyperion, Glycerin", "category": "running"},
    {"brand": "Saucony", "aliases": "Saucony, Endorphin, Peregrine, Xodus", "category": "running"},
    {"brand": "HOKA", "aliases": "HOKA, Hoka, Hoka One One, Speedgoat, Mafate, Tecton X", "category": "running"},
    {"brand": "La Sportiva", "aliases": "La Sportiva, Sportiva, Prodigio, Bushido, Akasha", "category": "outdoor"},
    {"brand": "Salomon", "aliases": "Salomon, S/Lab, Sense Ride, Speedcross, Pulsar", "category": "outdoor"},
    {"brand": "Altra", "aliases": "Altra, Lone Peak, Olympus, Mont Blanc", "category": "running"},
    {"brand": "On Running", "aliases": "On Running, On Cloud, Cloudmonster, Cloudsurfer, Cloudvista, Cloudboom", "category": "running"},
    {"brand": "New Balance", "aliases": "New Balance, NB, FuelCell, Fresh Foam", "category": "running"},
    {"brand": "Mizuno", "aliases": "Mizuno, Wave Rider, Wave Rebellion", "category": "running"},
    {"brand": "Topo Athletic", "aliases": "Topo Athletic, Topo, Ultraventure, MTN Racer", "category": "running"},
    {"brand": "Merrell", "aliases": "Merrell, Agility Peak, MTL", "category": "outdoor"},
    {"brand": "Inov-8", "aliases": "Inov-8, Inov8, Trailfly, Mudtalon", "category": "outdoor"},
    {"brand": "The North Face", "aliases": "The North Face, TNF, Vectiv, Summit Vectiv", "category": "outdoor"},
    {"brand": "Arc'teryx", "aliases": "Arc'teryx, Arcteryx, Norvan, Sylan", "category": "outdoor"},
    {"brand": "Under Armour", "aliases": "Under Armour, UA, Flow Velociti", "category": "running"},
    {"brand": "Reebok", "aliases": "Reebok, Floatride", "category": "running"},
    {"brand": "Craft", "aliases": "Craft, Craft Sportswear, Nordlite, Xplor", "category": "running"},
    {"brand": "NNormal", "aliases": "NNormal, Kjerag, Tomir", "category": "outdoor"},
    {"brand": "Scarpa", "aliases": "Scarpa, Spin Planet, Ribelle Run", "category": "outdoor"},
    {"brand": "Dynafit", "aliases": "Dynafit, Alpine Pro, Ultra Pro", "category": "outdoor"},
    {"brand": "REI", "aliases": "REI, REI Co-op", "category": "retailer"},
    {"brand": "Road Runner Sports", "aliases": "Road Runner Sports", "category": "retailer"},
    {"brand": "Fleet Feet", "aliases": "Fleet Feet", "category": "retailer"},
    {"brand": "Runner's World", "aliases": "Runner's World, Runners World", "category": "publisher"},
    {"brand": "OutdoorGearLab", "aliases": "OutdoorGearLab, Outdoor Gear Lab", "category": "publisher"},
    {"brand": "Switchback Travel", "aliases": "Switchback Travel", "category": "publisher"},
]


@dataclass
class BrandSeed:
    canonical: str
    display: str
    aliases: set[str] = field(default_factory=set)
    case_sensitive_aliases: set[str] = field(default_factory=set)
    is_target_brand: bool = False
    seed_sources: set[str] = field(default_factory=set)
    category: str = ""


@dataclass(frozen=True)
class BrandMatch:
    brand: str
    matched_alias: str
    evidence: str
    position: int


def build_brand_mentions_table(
    targets: list[QueryTarget],
    runs: list[QueryRun],
    citations_df: pd.DataFrame,
    brand_aliases: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, int], list[str]], pd.DataFrame]:
    """Return brand-level and run-level answer-text mention rows."""

    seeds = build_brand_seeds(targets, citations_df, brand_aliases)
    run_mentions: dict[tuple[str, int], list[str]] = {}
    run_rows = []
    for run in runs:
        matches = extract_brand_matches(run.response_text, seeds.values())
        run_mentions[(run.query, run.run_index)] = [match.brand for match in matches]
        for match in matches:
            run_rows.append(
                {
                    "query": run.query,
                    "run_index": run.run_index,
                    "brand": match.brand,
                    "matched_alias": match.matched_alias,
                    "evidence": match.evidence,
                    "position": match.position,
                }
            )

    run_mentions_df = pd.DataFrame(run_rows, columns=BRAND_RUN_COLUMNS)
    if not seeds:
        return _empty_brand_mentions(), run_mentions, run_mentions_df

    total_queries = len({target.query for target in targets})
    total_answers = len(runs)
    output_rows = []
    for seed in seeds.values():
        brand_rows = run_mentions_df[run_mentions_df["brand"] == seed.display] if not run_mentions_df.empty else pd.DataFrame()
        query_count = int(brand_rows["query"].nunique()) if not brand_rows.empty else 0
        answer_count = int(len(brand_rows.drop_duplicates(["query", "run_index"]))) if not brand_rows.empty else 0
        if query_count == 0 and not seed.is_target_brand:
            continue
        output_rows.append(
            {
                "brand": seed.display,
                "is_target_brand": seed.is_target_brand,
                "query_coverage": f"{query_count}/{total_queries} queries",
                "query_coverage_count": query_count,
                "total_queries": total_queries,
                "query_coverage_rate": round((query_count / total_queries * 100.0) if total_queries else 0.0, 1),
                "answer_frequency": f"{answer_count}/{total_answers} answers",
                "answer_frequency_count": answer_count,
                "total_answers": total_answers,
                "answer_frequency_rate": round((answer_count / total_answers * 100.0) if total_answers else 0.0, 1),
                "evidence_runs": _evidence_runs(brand_rows),
                "mentioned_queries": " | ".join(sorted(brand_rows["query"].unique())) if not brand_rows.empty else "",
                "seed_sources": " | ".join(sorted(seed.seed_sources)),
                "category": seed.category,
            }
        )

    if not output_rows:
        return _empty_brand_mentions(), run_mentions, run_mentions_df
    table = pd.DataFrame(output_rows)
    return table.sort_values(
        ["query_coverage_count", "answer_frequency_count", "is_target_brand", "brand"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True), run_mentions, run_mentions_df


BRAND_RUN_COLUMNS = ["query", "run_index", "brand", "matched_alias", "evidence", "position"]


def build_brand_seeds(
    targets: list[QueryTarget],
    citations_df: pd.DataFrame,
    brand_aliases: pd.DataFrame | None = None,
) -> dict[str, BrandSeed]:
    seeds: dict[str, BrandSeed] = {}
    for row in BUILTIN_BRANDS:
        _upsert_seed(
            seeds,
            row["brand"],
            aliases=split_aliases(row["aliases"]),
            is_target=False,
            source="built_in",
            category=row.get("category", ""),
        )

    if brand_aliases is not None and not brand_aliases.empty:
        for row in brand_aliases.fillna("").to_dict("records"):
            brand = str(row.get("brand", "")).strip()
            aliases = split_aliases(row.get("aliases", ""))
            category = str(row.get("category", "")).strip()
            if brand:
                _upsert_seed(seeds, brand, aliases=aliases, is_target=False, source="uploaded_alias", category=category)

    for target in targets:
        target_aliases = split_aliases(target.target_brand_aliases)
        if target_aliases:
            _upsert_seed(seeds, target_aliases[0], aliases=target_aliases, is_target=True, source="target_alias")
        domain_brand = brand_from_domain(target.target_domain)
        if domain_brand:
            _upsert_seed(seeds, domain_brand, is_target=True, source="target_domain")
        url_brand = brand_from_domain(target.target_url)
        if url_brand:
            _upsert_seed(seeds, url_brand, is_target=True, source="target_url")

    if not citations_df.empty and "cited_domain" in citations_df.columns:
        for domain in citations_df["cited_domain"].dropna().astype(str).unique():
            brand = brand_from_domain(domain)
            if brand:
                _upsert_seed(seeds, brand, is_target=False, source="cited_domain")
    return seeds


def extract_brand_matches(text: str, seeds: Iterable[BrandSeed]) -> list[BrandMatch]:
    matches: dict[str, BrandMatch] = {}
    for seed in seeds:
        found = _match_seed(text, seed)
        if found and (seed.display not in matches or found.position < matches[seed.display].position):
            matches[seed.display] = found
    return sorted(matches.values(), key=lambda match: (match.position, match.brand.lower()))


def mentioned_brands_for_text(text: str, seeds: Iterable[BrandSeed]) -> list[str]:
    return [match.brand for match in extract_brand_matches(text, seeds)]


def split_aliases(value) -> list[str]:
    aliases = []
    for alias in re.split(r"[,;|]", str(value or "")):
        clean = alias.strip()
        if clean:
            aliases.append(clean)
    return aliases


def brand_from_domain(value: str) -> str:
    host = normalize_host(value)
    if not host:
        host = extract_domain(value)
    labels = [label for label in host.split(".") if label]
    label = _registrable_domain_label(labels)
    if not label:
        return ""
    special = {
        "on": "On Running",
        "hoka": "HOKA",
        "asics": "ASICS",
        "rei": "REI",
        "outdoorgearlab": "OutdoorGearLab",
    }
    return special.get(label, _display_brand(label.replace("-", " ")))


def canonical_brand(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(value or "").lower())
    return " ".join(tokens)


def _upsert_seed(
    seeds: dict[str, BrandSeed],
    value: str,
    aliases: list[str] | None = None,
    is_target: bool = False,
    source: str = "",
    category: str = "",
) -> None:
    display = _display_brand(value)
    canonical = canonical_brand(display)
    if not canonical:
        return
    seed = seeds.get(canonical)
    if seed is None:
        seed = BrandSeed(canonical=canonical, display=display)
        seeds[canonical] = seed
    seed.is_target_brand = seed.is_target_brand or is_target
    if source:
        seed.seed_sources.add(source)
    if category and not seed.category:
        seed.category = category

    for alias in [value] + list(aliases or []):
        alias_clean = " ".join(str(alias or "").strip().split())
        if not alias_clean:
            continue
        alias_canonical = canonical_brand(alias_clean)
        if not alias_canonical:
            continue
        if alias_canonical == "on":
            seed.case_sensitive_aliases.add("On")
            continue
        seed.aliases.add(alias_canonical)
        if _needs_case_sensitive_match(alias_clean):
            seed.case_sensitive_aliases.add(alias_clean)


def _match_seed(text: str, seed: BrandSeed) -> BrandMatch | None:
    candidates = []
    for alias in sorted(seed.aliases, key=len, reverse=True):
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", flags=re.I)
        match = pattern.search(text)
        if match:
            candidates.append((match.start(), alias, match.group(0)))
    for alias in sorted(seed.case_sensitive_aliases, key=len, reverse=True):
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])")
        match = pattern.search(text)
        if match:
            candidates.append((match.start(), alias, match.group(0)))
    if not candidates:
        return None
    position, alias, matched_text = sorted(candidates, key=lambda item: item[0])[0]
    return BrandMatch(
        brand=seed.display,
        matched_alias=matched_text,
        evidence=_evidence_snippet(text, position, len(matched_text)),
        position=position + 1,
    )


def _registrable_domain_label(labels: list[str]) -> str:
    if not labels:
        return ""
    if len(labels) == 1:
        return "" if labels[0] in GENERIC_DOMAIN_LABELS else labels[0]
    domain_index = -2
    if (
        len(labels) >= 3
        and labels[-1] in COMPOUND_PUBLIC_SUFFIX_TLDS
        and labels[-2] in COMPOUND_PUBLIC_SUFFIX_LABELS
    ):
        domain_index = -3
    label = labels[domain_index]
    if label in GENERIC_DOMAIN_LABELS:
        prefix = labels[:domain_index] if domain_index != 0 else []
        for fallback in reversed(prefix):
            if fallback not in GENERIC_DOMAIN_LABELS:
                return fallback
        return ""
    return label


def _display_brand(value: str) -> str:
    clean = " ".join(re.findall(r"[A-Za-z0-9']+", str(value or "")))
    if not clean:
        return ""
    known = {
        "asics": "ASICS",
        "hoka": "HOKA",
        "rei": "REI",
        "nb": "NB",
        "tnf": "TNF",
        "ua": "UA",
        "nnormal": "NNormal",
        "inov 8": "Inov-8",
        "arcteryx": "Arc'teryx",
        "outdoorgearlab": "OutdoorGearLab",
    }
    canonical = canonical_brand(clean)
    if canonical in known:
        return known[canonical]
    if clean.isupper():
        return clean
    if any(ch.isupper() for ch in clean[1:]) and any(ch.islower() for ch in clean):
        return clean
    return clean.title()


def _needs_case_sensitive_match(alias: str) -> bool:
    canonical = canonical_brand(alias)
    return canonical in {"on"} or len(canonical) <= 2


def _evidence_snippet(text: str, start: int, length: int, window: int = 70) -> str:
    left = max(0, start - window)
    right = min(len(text), start + length + window)
    snippet = text[left:right].strip()
    if left > 0:
        snippet = "..." + snippet
    if right < len(text):
        snippet += "..."
    return snippet


def _evidence_runs(brand_rows: pd.DataFrame) -> str:
    if brand_rows.empty:
        return ""
    pairs = []
    for row in brand_rows.sort_values(["query", "run_index"]).to_dict("records"):
        pairs.append(f"{row['query']} run {int(row['run_index']) + 1}")
    return " | ".join(pairs)


def _empty_brand_mentions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "brand",
            "is_target_brand",
            "query_coverage",
            "query_coverage_count",
            "total_queries",
            "query_coverage_rate",
            "answer_frequency",
            "answer_frequency_count",
            "total_answers",
            "answer_frequency_rate",
            "evidence_runs",
            "mentioned_queries",
            "seed_sources",
            "category",
        ]
    )
