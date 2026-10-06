"""Opportunity scoring.

Every dimension carries an explicit ``kind``:

* ``OBSERVED`` - computed from counts we actually recorded while fetching.
* ``INFERRED`` - an assessment Impara derived without direct evidence.

Scores are Impara's current assessment, never market truth. Each dimension
must carry a reason string that explains the number.
"""
import re
from typing import List, Tuple

from .models import (
    Confidence,
    EvidenceKind,
    OpportunityScore,
    Problem,
    ScoreDimension,
)

PAIN_WORDS = re.compile(
    r"\b(frustrat\w*|annoying|painful|irritating|broken|struggle\w*|waste[sd]?|"
    r"hours?|manually|tedious|clunky|terrible|hate|impossible|can'?t|unable|"
    r"losing|error|fails?|stuck)\b",
    re.IGNORECASE,
)

SEVERITY_PHRASES = [
    ("hours", "people describe losing hours"),
    ("manually", "explicitly manual work"),
    ("struggle", "explicit struggle"),
    ("broken", "reports of broken behaviour"),
    ("frustrat", "expressed frustration"),
    ("can't", "expressed inability to complete the task"),
    ("losing", "described data or time loss"),
]


def score_opportunity(problem: Problem) -> OpportunityScore:
    dims: List[ScoreDimension] = []
    caveats: List[str] = []

    n_signals = len(problem.signals)
    n_sources = len(problem.sources)
    n_authors = len(problem.authors)
    n_patterns = len(problem.pattern_types)
    engagements = problem.engagements
    excerpts = " ".join(s.excerpt for s in problem.signals)

    # --- Evidence strength -------------------------------------------------
    ev = 0
    ev += min(45, n_signals * 6)
    ev += min(25, n_sources * 10)
    ev += min(20, n_authors * 5)
    ev += min(10, n_patterns * 5)
    ev = max(0, min(100, ev))
    dims.append(
        ScoreDimension(
            key="evidence",
            label="Evidence",
            value=ev,
            kind=EvidenceKind.OBSERVED,
            reason="%d verified signals across %d source(s), %d distinct author(s)"
            % (n_signals, n_sources, n_authors),
        )
    )

    # --- Pain / severity ---------------------------------------------------
    pain_hits = len(PAIN_WORDS.findall(excerpts))
    phrase_hits = sum(1 for p, _ in SEVERITY_PHRASES if p in excerpts.lower())
    pain = max(0, min(100, 30 + pain_hits * 7 + phrase_hits * 9))
    dims.append(
        ScoreDimension(
            key="pain",
            label="Pain",
            value=pain,
            kind=EvidenceKind.OBSERVED,
            reason="severity language in fetched text: %d term hit(s), %d phrasal marker(s)"
            % (pain_hits, phrase_hits),
        )
    )

    # --- Frequency ---------------------------------------------------------
    freq = 0
    freq += min(50, n_signals * 7)
    freq += min(30, n_authors * 6)
    freq += min(20, n_patterns * 7)
    freq = max(0, min(100, freq))
    dims.append(
        ScoreDimension(
            key="frequency",
            label="Frequency",
            value=freq,
            kind=EvidenceKind.OBSERVED,
            reason="%d independent occurrence(s) of %d distinct complaint pattern(s)"
            % (n_signals, n_patterns),
        )
    )

    # --- Buildability ------------------------------------------------------
    vocab = len(set(problem.terms))
    build = 70 if vocab < 12 else (58 if vocab < 20 else 46)
    build = max(20, min(95, build + (5 if n_signals <= 6 else -5)))
    dims.append(
        ScoreDimension(
            key="buildability",
            label="Buildability",
            value=build,
            kind=EvidenceKind.INFERRED,
            reason="inferred from vocabulary spread (%d terms); not directly observed"
            % vocab,
        )
    )
    caveats.append("Buildability is inferred, not evidenced.")

    # --- Competition -------------------------------------------------------
    competition = 60 if n_sources <= 1 else 52
    dims.append(
        ScoreDimension(
            key="competition",
            label="Competition",
            value=competition,
            kind=EvidenceKind.INFERRED,
            reason="no competitor scan performed; value is a placeholder assumption"
        )
    )
    caveats.append("Competition is unmeasured. Run a competitor scan before trusting it.")

    # --- Monetization ------------------------------------------------------
    wants_phrase = bool(re.search(r"\b(i'?d pay|worth paying|shut up and take my money|"
                                  r"paid (?:for|version|tool)|subscription)\b", excerpts, re.I))
    monet = 70 if wants_phrase else (55 if pain >= 60 else 45)
    dims.append(
        ScoreDimension(
            key="monetization",
            label="Monetization",
            value=monet,
            kind=EvidenceKind.INFERRED,
            reason="willingness-to-pay language %s; inferred from pain, not observed"
            % ("present" if wants_phrase else "absent"),
        )
    )
    if not wants_phrase:
        caveats.append("No willingness-to-pay evidence found in any signal.")

    total = int(round(sum(d.value for d in dims) / float(len(dims))))
    strength = evidence_strength(n_signals, n_sources, ev)

    if n_signals < 3 or n_sources < 2:
        caveats.append(
            "Insufficient evidence for a confident assessment "
            "(%d signals, %d source(s))." % (n_signals, n_sources)
        )
    if strength is Confidence.NONE:
        caveats.append("Unverified opportunity.")

    return OpportunityScore(
        total=total,
        dimensions=dims,
        evidence_strength=strength,
        caveats=caveats,
    )


def evidence_strength(n_signals: int, n_sources: int, evidence_dim: int) -> Confidence:
    if n_signals == 0:
        return Confidence.NONE
    if n_signals >= 8 and n_sources >= 2 and evidence_dim >= 70:
        return Confidence.STRONG
    if n_signals >= 4 and evidence_dim >= 50:
        return Confidence.MODERATE
    return Confidence.WEAK
