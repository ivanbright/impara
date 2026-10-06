"""Convert raw source candidates into verified signals.

This is the evidence gate. A candidate becomes a ``Signal`` only when a
complaint pattern is **literally present in the text we fetched**. Search
rankings, popularity and recency never promote a candidate on their own.

If nothing survives verification for a source, that source contributes zero
signals and says so.
"""
import hashlib
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import sources as src
from .models import Signal, SourceHealth, SourceStatus
from .patterns import PATTERNS, find_patterns
from .terms import extract_terms

# Representative query text per pattern. Search APIs get a short phrase; the
# regex gate decides whether the hit really contains the pattern.
SEARCH_QUERY: Dict[str, str] = {
    "wish": "i wish there was",
    "use_because": "i use because",
    "frustrating": "this is frustrating",
    "manual": "i have to do this manually",
    "why_no_support": "why doesnt support",
    "tried_tools": "i tried several tools but",
}

MAX_EXCERPT = 260

Converter = Callable[[Dict[str, Any]], Optional[Signal]]
Fetcher = Callable[[str, int], Tuple[List[Dict[str, Any]], str]]


def _stable_id(*parts: str) -> str:
    raw = "|".join(p for p in parts if p)
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:8]


def _excerpt(text: str, span_start: int, span_end: int) -> str:
    """Return the sentence containing the match, clamped to a readable size."""
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return ""
    if 0 <= span_start < len(clean):
        window = clean[max(0, span_start - 60): span_end + 140].strip()
    else:
        window = clean[:MAX_EXCERPT]
    if len(window) > MAX_EXCERPT:
        window = window[:MAX_EXCERPT].rsplit(" ", 1)[0] + "..."
    return window


def _make_signal(
    source: str,
    title: str,
    body: str,
    url: str,
    author: str = "",
    engagement: int = 0,
    observed_at: str = "",
    thread: str = "",
    verify: str = "",
) -> Optional[Signal]:
    """``verify`` is the text that must exhibit a complaint pattern.

    When provided it is used exclusively for pattern matching and excerpting,
    so a signal obtained from a thread cannot inherit the pattern from the
    thread's title. Every signal must show the pattern in its own text.
    """
    haystack = verify if verify else "%s\n%s" % (title or "", body or "")
    matches = find_patterns(haystack)
    if not matches:
        return None

    start, end = 0, 0
    for spec in PATTERNS:
        m = re.compile(spec.regex, re.IGNORECASE).search(haystack)
        if m:
            start, end = m.start(), m.end()
            break

    terms = extract_terms("%s %s" % (title, body))
    return Signal(
        id=_stable_id(source, url, title),
        source=source,
        title=re.sub(r"\s+", " ", (title or "").strip())[:200],
        excerpt=_excerpt(haystack, start, end),
        url=url,
        patterns=[p.key for p in matches],
        terms=terms,
        author=author,
        engagement=max(0, int(engagement or 0)),
        observed_at=str(observed_at or ""),
        thread=thread,
    )


# --------------------------------------------------------------------------- #
# Per-source converters
# --------------------------------------------------------------------------- #
def from_github(item: Dict[str, Any]) -> Optional[Signal]:
    repo = (item.get("repository_url") or "").split("repos/")[-1]
    url = item.get("html_url") or ("https://github.com/%s" % repo)
    return _make_signal(
        source="github",
        title=item.get("title") or "",
        body=item.get("body") or "",
        url=url,
        author=((item.get("user") or {}).get("login") or ""),
        engagement=int(item.get("comments") or 0),
        observed_at=str(item.get("created_at") or "")[:10],
        # Each GitHub hit is its own issue, so there is no thread that implies
        # one shared topic; leave blank and let term similarity decide.
        thread="",
    )


def from_hn(item: Dict[str, Any]) -> Optional[Signal]:
    title = item.get("title") or item.get("story_title") or ""
    body = item.get("comment_text") or item.get("story_text") or ""
    oid = item.get("objectID")
    url = item.get("url") or ("https://news.ycombinator.com/item?id=%s" % oid)
    story = item.get("story_id") or item.get("story_url") or ""
    # A comment must exhibit the pattern in its own text. Only a story hit is
    # verified against its title, because there the title IS the statement.
    verify = body if body else title
    return _make_signal(
        source="hackernews",
        title=title,
        body=body,
        url=url,
        author=item.get("author") or "",
        engagement=int(item.get("points") or 0) + int(item.get("num_comments") or 0),
        observed_at=str(item.get("created_at") or "")[:10],
        thread=("hn:%s" % story) if story else "",
        verify=verify,
    )


def from_stackexchange(item: Dict[str, Any]) -> Optional[Signal]:
    body = re.sub(r"<[^>]+>", " ", item.get("body") or "")
    owner = item.get("owner") or {}
    qid = item.get("question_id")
    return _make_signal(
        source="stackexchange",
        title=item.get("title") or "",
        body=body,
        url=item.get("link") or "",
        author=owner.get("display_name") or "",
        engagement=int(item.get("score") or 0),
        observed_at=str(item.get("creation_date") or ""),
        thread=("se:%s" % qid) if qid else "",
    )


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
def _collect(
    fetch: Fetcher,
    convert: Converter,
    per_page: int,
    health: SourceHealth,
) -> List[Signal]:
    signals: List[Signal] = []
    last_err = ""

    for query in SEARCH_QUERY.values():
        try:
            items, err = fetch(query, per_page)
        except Exception as exc:  # never let one source abort discovery
            items, err = [], "unreachable (%s)" % type(exc).__name__
        if err and not items and not last_err:
            last_err = err
        for raw in items:
            try:
                sig = convert(raw)
            except Exception:
                continue
            if sig is not None:
                signals.append(sig)

    if last_err and not signals:
        health.status = (
            SourceStatus.RATE_LIMITED if last_err == "rate_limited" else SourceStatus.UNAVAILABLE
        )
        health.note = "unavailable: %s" % last_err

    health.signals_found = len(signals)
    return signals


def _fetch_github(query: str, n: int):
    return src.github_candidates(query, per_page=n)


# Thread expansion: pull the rest of a discussion that already showed a
# complaint. Empirically ~0.5% of comments match a complaint pattern, so this
# costs many extra requests for little yield - it is opt-in via --deep.
DEEP_EXPAND_PER_QUERY = 4
HN_MAX_COMMENTS = 400
HN_MAX_COMMENTS_PER_THREAD = 60
_DEEP = False


def _flatten_comments(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    stack = list(tree.get("children") or [])
    while stack:
        node = stack.pop(0)
        if not isinstance(node, dict):
            continue
        if node.get("text"):
            out.append(node)
        kids = node.get("children")
        if kids:
            stack.extend(kids)
    return out


def _expand_thread(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    story_id = tree.get("id")
    story_title = tree.get("title") or ""
    items: List[Dict[str, Any]] = []
    for node in _flatten_comments(tree)[:HN_MAX_COMMENTS_PER_THREAD]:
        comment_id = node.get("id")
        text = re.sub(r"<[^>]+>", " ", node.get("text") or "")
        if not text.strip():
            continue
        items.append(
            {
                "comment_text": text,
                "story_title": story_title,
                "story_id": story_id,
                "objectID": comment_id,
                "author": node.get("author") or "",
                "created_at": (node.get("created_at") or "")[:10],
                "points": 0,
                "num_comments": 0,
                "url": "https://news.ycombinator.com/item?id=%s" % comment_id,
            }
        )
    return items


def _fetch_hn(query: str, n: int) -> Tuple[List[Dict[str, Any]], str]:
    out: List[Dict[str, Any]] = []
    last = ""

    stories, err = src.hn_search(query, tags="story", hits=n)
    if stories:
        out.extend(stories)
    elif err and not last:
        last = err

    comments, err = src.hn_search(query, tags="comment", hits=n)
    if comments:
        out.extend(comments)
    elif err and not last:
        last = err

    # Prefer threads that already showed repeated complaints, then threads
    # whose story title matched. Expansion only adds comments that themselves
    # contain a pattern, so nothing is inherited from the title.
    tally: Dict[Any, Dict[str, Any]] = {}
    for c in comments:
        sid = c.get("story_id")
        if not sid:
            continue
        tally.setdefault(sid, {"id": sid, "matched": 0, "title": ""})["matched"] += 1
    for s in stories:
        sid = s.get("objectID")
        if not sid:
            continue
        tally.setdefault(sid, {"id": sid, "matched": 0, "title": s.get("title") or ""})

    candidates = sorted(tally.values(), key=lambda v: -int(v.get("matched", 0)))
    if not _DEEP:
        return out, last

    expanded = 0
    for cand in candidates:
        if expanded >= DEEP_EXPAND_PER_QUERY:
            break
        tree, terr = src.hn_item(str(cand["id"]))
        if terr and not last:
            last = terr
        if not tree:
            continue
        # Skip megathreads (hiring posts and the like) that would waste time.
        if int(tree.get("descendants") or 0) > HN_MAX_COMMENTS:
            continue
        out.extend(_expand_thread(tree))
        expanded += 1

    return out, last


def _fetch_se(query: str, n: int):
    return src.stackexchange_candidates(query, pagesize=n)


_FETCHERS: Dict[str, Tuple[Fetcher, Converter]] = {
    "github": (_fetch_github, from_github),
    "hackernews": (_fetch_hn, from_hn),
    "stackexchange": (_fetch_se, from_stackexchange),
}


def collect(
    sources: Optional[List[str]] = None,
    market: str = "",
    country: str = "",
    per_source: int = 30,
    deep: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Signal], List[SourceHealth]]:
    """Fetch candidates from every healthy source and keep only verified ones.

    Returns ``(signals, health)``.
    """
    global _DEEP
    _DEEP = deep
    health = src.probe_sources()
    available = {h.key for h in health if h.available}
    wanted = set(sources) if sources else set(available)

    def note(msg: str) -> None:
        if progress is not None:
            progress(msg)

    signals: List[Signal] = []
    for h in health:
        if h.key not in wanted:
            continue
        if not h.available or h.key not in _FETCHERS:
            if h.available:
                note("skipping %s (no fetcher)\n" % h.key)
            else:
                note("%s unavailable, skipping\n" % h.label)
            continue
        note("querying %s ...\n" % h.label)
        fetch, convert = _FETCHERS[h.key]
        got = _collect(fetch, convert, per_source, h)
        note("  %s: %d verified signal(s)\n" % (h.label, len(got)))
        signals.extend(got)

    if market:
        signals = [s for s in signals if _matches_filter(s, market)]
    if country:
        signals = [s for s in signals if _matches_filter(s, country)]

    for h in health:
        if h.key in wanted:
            h.signals_found = len([s for s in signals if s.source == h.key])

    # De-duplicate by stable id, keeping the first observation.
    seen: set = set()
    unique: List[Signal] = []
    for s in signals:
        if s.id in seen:
            continue
        seen.add(s.id)
        unique.append(s)
    return unique, health


def _matches_filter(signal: Signal, term: str) -> bool:
    t = term.strip().lower()
    if not t:
        return True
    hay = "%s %s %s" % (signal.title, signal.excerpt, " ".join(signal.terms))
    return t in hay.lower()
