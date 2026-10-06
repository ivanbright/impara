"""Build the structured problem profile.

Answers the questions from the product definition while keeping observed
evidence in a separate bucket from inference.
"""
import re
from typing import List, Tuple

from .models import (
    Confidence,
    EvidenceKind,
    EvidenceRecord,
    Problem,
    ProblemProfile,
)
from .patterns import PATTERNS
from .scoring import evidence_strength

WORKAROUND_RE = re.compile(
    r"\bi\s+(?:currently\s+)?use\s+(.{2,60}?)\s+because\b", re.IGNORECASE
)
PAY_RE = re.compile(
    r"\b(i'?d pay|worth paying|shut up and take my money|paid (?:for|version|tool)|"
    r"subscription|pricing)\b",
    re.IGNORECASE,
)


def build_profile(problem: Problem) -> ProblemProfile:
    text = " ".join(s.excerpt for s in problem.signals)
    n = len(problem.signals)
    sources = problem.sources

    n_signals = n
    ev_score = min(100, n_signals * 6 + len(sources) * 10 + len(problem.authors) * 5)
    strength = evidence_strength(n_signals, len(sources), ev_score)

    evidence: List[EvidenceRecord] = [
        EvidenceRecord(
            kind=EvidenceKind.OBSERVED,
            claim="%d independent problem signals" % n,
            detail="Verified by literal pattern match in fetched text.",
            url=problem.signals[0].url if problem.signals else "",
        ),
        EvidenceRecord(
            kind=EvidenceKind.OBSERVED,
            claim="%d distinct complaint pattern(s)" % len(problem.pattern_types),
            detail=", ".join(problem.pattern_types),
        ),
        EvidenceRecord(
            kind=EvidenceKind.OBSERVED,
            claim="Sources: %s" % (", ".join(sources) or "none"),
            detail="%d distinct author(s)" % len(problem.authors),
        ),
        EvidenceRecord(
            kind=EvidenceKind.OBSERVED,
            claim="%d total engagement action(s)" % problem.engagements,
            detail="sum of comments/points/scores on the source items",
        ),
    ]

    workarounds = _workarounds(text)
    if workarounds:
        evidence.append(
            EvidenceRecord(
                kind=EvidenceKind.OBSERVED,
                claim="Stated workaround: %s" % workarounds[0],
                detail="quoted from signal text",
            )
        )

    paying = bool(PAY_RE.search(text))
    if paying:
        evidence.append(
            EvidenceRecord(
                kind=EvidenceKind.OBSERVED,
                claim="Willingness-to-pay language present",
                detail="one or more signals mention paying for a solution",
            )
        )

    insufficient = n < 3 or strength in (Confidence.NONE, Confidence.WEAK)

    return ProblemProfile(
        problem=problem,
        who=_who(problem),
        trying_to_accomplish=_goal(problem),
        why_difficult=_why_difficult(problem),
        frequency=_frequency(problem, text),
        workarounds=workarounds or ["not stated in any observed signal"],
        why_existing_insufficient=_insufficient(problem, text),
        evidence=evidence,
        evidence_strength=strength,
        severity=_severity(text),
        paying_now=("yes - willingness to pay observed" if paying else
                    "no payment behaviour observed"),
        product_type=_product_type(problem),
        insufficient_evidence=insufficient,
    )


def _who(problem: Problem) -> str:
    """Only claim what the sources support. Who posts is observed; who *has*
    the problem is an inference and is labelled as such."""
    authors = problem.authors
    srcs = ", ".join(problem.sources) or "unknown source"
    base = "Authors posting in %s (%d distinct)" % (srcs, len(authors))
    hints = [t for t in problem.terms[:6]]
    if hints:
        return "%s | domain vocabulary observed: %s" % (base, ", ".join(hints))
    return base


def _goal(problem: Problem) -> str:
    """Summarise the intent stated in the signals without inventing a task."""
    for signal in problem.signals:
        m = WORKAROUND_RE.search(signal.excerpt)
        if m:
            return "Complete work currently done with: %s" % m.group(1).strip()
    terms = problem.terms[:5]
    if terms:
        return "Observable intent keywords: %s" % ", ".join(terms)
    return "not stated in observed text"


def _why_difficult(problem: Problem) -> str:
    reasons: List[str] = []
    types = problem.pattern_types
    meanings = {p.key: p.meaning for p in PATTERNS}
    for t in types:
        if t in meanings:
            reasons.append(meanings[t])
    if not reasons:
        return "not stated in observed text"
    seen: List[str] = []
    for r in reasons:
        if r not in seen:
            seen.append(r)
    return "; ".join(seen)


def _frequency(problem: Problem, text: str) -> str:
    n = len(problem.signals)
    if n >= 8:
        band = "frequent"
    elif n >= 4:
        band = "recurring"
    elif n >= 2:
        band = "occasional"
    else:
        band = "isolated"
    return "%s - %d observed signal(s), %d source(s), %d author(s)" % (
        band,
        n,
        len(problem.signals and problem.sources),
        len(problem.authors),
    )


def _workarounds(text: str) -> List[str]:
    out: List[str] = []
    for m in WORKAROUND_RE.finditer(text):
        value = m.group(1).strip().strip(".,;")
        if value and value.lower() not in (w.lower() for w in out):
            out.append(value)
        if len(out) >= 3:
            break
    return out


def _insufficient(problem: Problem, text: str) -> str:
    if WORKAROUND_RE.search(text):
        return "existing tools require the manual step described in the signals"
    types = problem.pattern_types
    if "tried_tools" in types:
        return "signals explicitly report trying several tools and still failing"
    if "wish" in types:
        return "the capability is reported absent from current options"
    if "why_no_support" in types:
        return "a missing integration is reported"
    return "not stated in observed text"


def _severity(text: str) -> str:
    hits = sum(
        1
        for w in ("frustrat", "annoying", "painful", "broken", "struggle", "hours", "manually")
        if w in text.lower()
    )
    if hits >= 3:
        return "high - repeated severity language"
    if hits >= 1:
        return "moderate - severity language present"
    return "low - no severity language observed"


def _product_type(problem: Problem) -> str:
    """A cautious classification, explicitly an inference."""
    terms = set(problem.terms)
    if terms & {"automation", "automate", "manual", "integrate", "integration", "sync", "synchronize"}:
        return "workflow automation / integration layer (inferred)"
    if terms & {"export", "import", "report", "reports", "dashboard", "tracking", "data"}:
        return "data reporting / tracking tool (inferred)"
    if terms & {"team", "teams", "project", "projects", "customer", "customers"}:
        return "team or project coordination tool (inferred)"
    return "unclassified - insufficient signal vocabulary"
