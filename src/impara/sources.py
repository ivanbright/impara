"""Signal sources.

Every source must return *candidates*. Candidates are not evidence until
``extract`` has locally confirmed that a complaint pattern is literally
present in the fetched text.

Sources are deliberately restricted to public, no-auth endpoints, and every
source reports its own health so the CLI can say plainly what it could not
reach instead of silently pretending.
"""
import http.client
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import Signal, SourceHealth, SourceStatus

USER_AGENT = "impara/0.3 (+https://github.com/ivanbright/impara)"
TIMEOUT = 15

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".impara")
CACHE_TTL = 6 * 3600  # seconds


def _cache_path() -> str:
    return os.path.join(CACHE_DIR, "cache.json")


def _load_cache() -> Dict[str, Any]:
    try:
        with open(_cache_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    try:
        if not os.path.isdir(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_cache_path(), "w", encoding="utf-8") as fh:
            json.dump(cache, fh)
    except OSError:
        # A read-only or locked cache must never break discovery.
        pass


def fetch_json(url: str, cache_key: str = "") -> Tuple[Optional[Any], str]:
    """GET a JSON document.

    Returns ``(data, error)``. Exactly one of the two is set. Network and
    HTTP failures are surfaced as strings rather than raised, so a single
    unreachable source degrades the run instead of aborting it.
    """
    now = time.time()
    if cache_key:
        cached = _load_cache().get(cache_key)
        if cached and isinstance(cached, dict) and now - cached.get("t", 0) < CACHE_TTL:
            return cached.get("d"), ""

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    ctx = ssl.create_default_context()

    # Transient transport failures (chunked-stream truncation, resets) are
    # retried once, then reported as unavailable. A flaky source must never
    # crash a discovery run.
    last_err = ""
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx) as resp:
                body = resp.read().decode("utf-8", "replace")
            break
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                return None, "rate_limited"
            return None, "http_%d" % exc.code
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", None)
            last_err = "unreachable (%s)" % reason
        except (http.client.HTTPException, ssl.SSLError, OSError, ValueError) as exc:
            last_err = "truncated response (%s)" % type(exc).__name__
        if attempt == 0:
            time.sleep(0.6)
    else:
        return None, last_err or "unreachable"

    try:
        data = json.loads(body)
    except ValueError:
        return None, "invalid_json"

    if cache_key:
        cache = _load_cache()
        cache[cache_key] = {"t": now, "d": data}
        _save_cache(cache)
    return data, ""


# --------------------------------------------------------------------------- #
# GitHub issues
# --------------------------------------------------------------------------- #
def github_search(query: str, per_page: int = 30) -> Tuple[List[Dict[str, Any]], str]:
    url = "https://api.github.com/search/issues?%s" % urllib.parse.urlencode(
        {"q": query, "sort": "created", "order": "desc", "per_page": per_page}
    )
    data, err = fetch_json(url, cache_key="gh:" + query)
    if err:
        return [], err
    if not isinstance(data, dict):
        return [], "invalid_json"
    return data.get("items") or [], ""


def github_candidates(pattern_query: str, per_page: int = 30) -> Tuple[List[Dict[str, Any]], str]:
    """Unquoted search: GitHub's quoted-phrase matching is unreliable, so we
    over-fetch and let ``extract`` verify each hit."""
    return github_search("%s type:issue" % pattern_query, per_page=per_page)


# --------------------------------------------------------------------------- #
# Hacker News (Algolia)
# --------------------------------------------------------------------------- #
def hn_search(query: str, tags: str, hits: int = 30) -> Tuple[List[Dict[str, Any]], str]:
    url = "https://hn.algolia.com/api/v1/search?%s" % urllib.parse.urlencode(
        {"query": query, "tags": tags, "hitsPerPage": hits}
    )
    data, err = fetch_json(url, cache_key="hn:%s:%s" % (tags, query))
    if err:
        return [], err
    if not isinstance(data, dict):
        return [], "invalid_json"
    return data.get("hits") or [], ""


def hn_item(item_id: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Full comment tree for one HN item."""
    url = "https://hn.algolia.com/api/v1/items/%s" % item_id
    data, err = fetch_json(url, cache_key="hnitem:%s" % item_id)
    if err:
        return None, err
    if not isinstance(data, dict):
        return None, "invalid_json"
    return data, ""


# --------------------------------------------------------------------------- #
# Stack Exchange
# --------------------------------------------------------------------------- #
def stackexchange_search(query: str, pagesize: int = 30) -> Tuple[List[Dict[str, Any]], str]:
    url = "https://api.stackexchange.com/2.3/search/advanced?%s" % urllib.parse.urlencode(
        {
            "order": "desc",
            "sort": "votes",
            "q": query,
            "site": "stackoverflow",
            "pagesize": pagesize,
            "filter": "default",
        }
    )
    data, err = fetch_json(url, cache_key="se:" + query)
    if err:
        return [], err
    if not isinstance(data, dict):
        return [], "invalid_json"
    return data.get("items") or [], ""


def stackexchange_candidates(pattern_query: str, pagesize: int = 30) -> Tuple[List[Dict[str, Any]], str]:
    url = "https://api.stackexchange.com/2.3/search/advanced?%s" % urllib.parse.urlencode(
        {
            "order": "desc",
            "sort": "activity",
            "q": pattern_query,
            "site": "stackoverflow",
            "pagesize": pagesize,
            "filter": "withbody",
        }
    )
    data, err = fetch_json(url, cache_key="seb:" + pattern_query)
    if err:
        return [], err
    if not isinstance(data, dict):
        return [], "invalid_json"
    return data.get("items") or [], ""


# --------------------------------------------------------------------------- #
# Availability probe
# --------------------------------------------------------------------------- #
def probe_sources() -> List[SourceHealth]:
    """Cheap round-trip against each source so ``impara sources`` reports the
    truth about what is reachable right now."""
    health: List[SourceHealth] = []
    probes = [
        (
            "github",
            "GitHub issues",
            "https://api.github.com/search/issues?q=%s&per_page=1"
            % urllib.parse.quote("i wish there was"),
            "probe:github",
            "issue tracker discussions and feature requests",
        ),
        (
            "hackernews",
            "Hacker News (Algolia)",
            "https://hn.algolia.com/api/v1/search?query=%s&tags=story&hitsPerPage=1"
            % urllib.parse.quote("i wish there was"),
            "probe:hn",
            "public discussion threads and comments",
        ),
        (
            "stackexchange",
            "Stack Overflow (Stack Exchange)",
            "https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=votes"
            "&q=%s&site=stackoverflow&pagesize=1" % urllib.parse.quote("wish there was"),
            "probe:stackexchange",
            "questions and accepted-workaround discussions",
        ),
    ]
    for key, label, url, cache_key, ok_note in probes:
        _, err = fetch_json(url, cache_key=cache_key)
        health.append(
            SourceHealth(
                key=key,
                label=label,
                status=_status(err),
                note=_note(err, ok_note),
            )
        )

    # Reddit blocks anonymous API access. We do not pretend otherwise.
    health.append(
        SourceHealth(
            key="reddit",
            label="Reddit",
            status=SourceStatus.UNAVAILABLE,
            note="anonymous API access blocked; not queried. No signals are attributed to it.",
        )
    )
    return health


def _status(err: str) -> SourceStatus:
    if not err:
        return SourceStatus.OK
    if err == "rate_limited":
        return SourceStatus.RATE_LIMITED
    return SourceStatus.UNAVAILABLE


def _note(err: str, ok_note: str) -> str:
    if not err:
        return ok_note
    return "unavailable: %s" % err
