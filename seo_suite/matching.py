"""URL normalization and target matching helpers."""

from __future__ import annotations

import fnmatch
import re
from urllib.parse import urlparse, urlunparse


def ensure_scheme(url: str) -> str:
    value = str(url or "").strip()
    if value and not re.match(r"^https?://", value, flags=re.I):
        return "https://" + value
    return value


def normalize_host(value: str) -> str:
    if not value:
        return ""
    parsed = urlparse(ensure_scheme(value.lower()))
    host = parsed.netloc or parsed.path.split("/")[0]
    if "@" in host:
        host = host.rsplit("@", 1)[1]
    host = host.split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def normalize_url(value: str) -> str:
    if not value:
        return ""
    parsed = urlparse(ensure_scheme(value.strip()))
    scheme = parsed.scheme.lower() or "https"
    host = normalize_host(parsed.netloc or parsed.path)
    path = parsed.path if parsed.netloc else "/" + "/".join(parsed.path.split("/")[1:])
    path = re.sub(r"/{2,}", "/", path or "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return urlunparse((scheme, host, path, "", parsed.query, "")).lower()


def is_same_or_subdomain(host: str, target_domain: str) -> bool:
    clean_host = normalize_host(host)
    clean_target = normalize_host(target_domain)
    if not clean_host or not clean_target:
        return False
    return clean_host == clean_target or clean_host.endswith("." + clean_target)


def is_exact_url_match(url: str, target_url: str) -> bool:
    return bool(url and target_url and normalize_url(url) == normalize_url(target_url))


def is_pattern_match(url: str, pattern: str) -> bool:
    if not url or not pattern:
        return False
    normalized = normalize_url(url)
    pattern_value = str(pattern).strip().lower()
    if not pattern_value:
        return False

    if ".*" in pattern_value or re.search(r"[\^\$\(\)\[\]\{\}\+]", pattern_value):
        try:
            return re.search(pattern_value, normalized) is not None
        except re.error:
            return False

    pattern_normalized = normalize_url(pattern_value) if "://" in pattern_value or "." in pattern_value.split("/")[0] else pattern_value
    return fnmatch.fnmatch(normalized, pattern_normalized)


def extract_domain(url: str) -> str:
    return normalize_host(url)
