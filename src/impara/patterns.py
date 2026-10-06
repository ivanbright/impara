"""Complaint-signal patterns.

These are the linguistic shapes of a real problem. A piece of text only
counts as an *observed signal* if one of these patterns is literally present
in it. Nothing here is inferred.
"""
import re
from typing import List, NamedTuple

from typing import Pattern


class PatternSpec(NamedTuple):
    key: str
    label: str
    regex: str
    # What the pattern tells us about the problem, used downstream.
    meaning: str


PATTERNS: List[PatternSpec] = [
    PatternSpec(
        key="wish",
        label='I wish there was...',
        regex=r"\bi\s+wish\s+(?:there\s+was|there\s+were|i\s+could|i\s+had)\b",
        meaning="a capability is absent",
    ),
    PatternSpec(
        key="use_because",
        label="I currently use X because...",
        regex=r"\bi\s+(?:currently\s+)?use\s+.+?\s+because\b",
        meaning="a workaround exists for a reason",
    ),
    PatternSpec(
        key="frustrating",
        label="This is frustrating...",
        regex=r"\b(?:this|it|that)\s+(?:is|'s|was)\s+(?:really\s+|so\s+)?(?:frustrating|annoying|painful|irritating)\b",
        meaning="the current process causes pain",
    ),
    PatternSpec(
        key="manual",
        label="I have to do this manually...",
        regex=r"\bi\s+have\s+to\s+(?:do|redo|re-do|run|copy|paste|check)\s+(?:this|it|all\s+of\s+that|everything)\s+manually\b",
        meaning="work that should be automated",
    ),
    PatternSpec(
        key="why_no_support",
        label="Why doesn't X support...",
        regex=r"\bwhy\s+(?:doesn'?t|can'?t|don'?t)\s+\S+\s+(?:support|allow|integrate|import|export|handle|work\s+with)\b",
        meaning="a missing integration or capability",
    ),
    PatternSpec(
        key="tried_tools",
        label="I've tried several tools but...",
        regex=r"\bi'?ve\s+tried\s+(?:several|various|many|a\s+few|all|multiple)\s+(?:tools|apps|solutions|alternatives|products|things)\s+but\b",
        meaning="existing solutions have been rejected",
    ),
]

_COMPILED = [(p, re.compile(p.regex, re.IGNORECASE)) for p in PATTERNS]


def find_patterns(text: str):
    """Return every pattern actually present in ``text``.

    This is the evidence gate: presence must be observed in the source text,
    never assumed from a search ranking.
    """
    if not text:
        return []
    return [spec for spec, rx in _COMPILED if rx.search(text)]


def pattern_keys() -> List[str]:
    return [p.key for p in PATTERNS]
