"""Term extraction and clustering helpers."""
import re
from typing import Dict, List, Sequence, Set

STOPWORDS: Set[str] = set(
    """
    a about above after again against all am an and any are as at be because
    been before being below between both but by can could did do does doing
    down during each few for from further had has have having he her here hers
    herself him himself his how i if in into is it its itself just me more
    most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why
    will with would you your yours yourself yourselves
    """.split()
)

# Terms that signal product/market relevance rather than boilerplate.
DOMAIN_HINTS: Set[str] = set(
    """
    tool tools app apps product products software service services platform
    workflow workflows automate automation manual export import integrate
    integration sync synchronize backup report reports dashboard tracking
    support feature features api data file files database schedule scheduled
    team teams project projects invoice invoices budget budgeting payment
    payments customer customers email calendar task tasks
    """.split()
)

# Words that describe the *shape* of a complaint rather than its subject.
# Keeping them makes clusters merge on shared phrasing instead of shared
# topic, and makes titles read like noise.
NOISE_WORDS: Set[str] = set(
    """
    wish wished wishing want wants wanted need needs needed use used using
    manually manual thing things one two three say says said really just
    still even ever never get gets got make makes made going gonna like
    much many more most some any every always often sometimes actually
    possible able can could would should may might shall
    wishthere was were there here this that these those
    """.split()
)

_TOKEN = re.compile(r"[a-z][a-z0-9]{2,}")
_HAS_DIGIT_RUN = re.compile(r"\d{3,}")


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [
        t
        for t in _TOKEN.findall(text.lower())
        if t not in STOPWORDS
        and t not in NOISE_WORDS
        and not _HAS_DIGIT_RUN.search(t)
    ]


def extract_terms(text: str, limit: int = 18) -> List[str]:
    """Content terms, ranked by frequency, domain vocabulary boosted."""
    counts: Dict[str, int] = {}
    for tok in tokenize(text):
        counts[tok] = counts.get(tok, 0) + 1

    def rank(item):
        word, n = item
        boost = 4 if word in DOMAIN_HINTS else 0
        return (-(n + boost), word)

    ranked = sorted(counts.items(), key=rank)
    return [w for w, _ in ranked[:limit]]


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    if inter == 0:
        return 0.0
    return inter / float(len(sa | sb))


def ngram_overlap(a: Sequence[str], b: Sequence[str]) -> float:
    """Overlap on shared multi-word phrases, which clusters better than raw
    single terms for problem statements."""
    ga = set(zip(a, a[1:])) if len(a) > 1 else set()
    gb = set(zip(b, b[1:])) if len(b) > 1 else set()
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / float(len(ga | gb))


def similarity(a: Sequence[str], b: Sequence[str]) -> float:
    return max(jaccard(a, b), ngram_overlap(a, b))
