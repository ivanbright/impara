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
    pain = max(0, min(100, pain_hits * 7 + phrase_hits * 9))
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

    # --- Dimensions we cannot measure yet ---------------------------------
    # Deliberately unscored. A number here would be decoration dressed as a
    # measurement, so these are reported as unknown until an evidence-
    # producing mechanism exists for them.
    for key, label, why in (
        (
            "buildability",
            "Buildability",
            "nothing has been measured about how hard this would be to build",
        ),
        (
            "competition",
            "Competition",
            "no competitor scan has been run",
        ),
        (
            "monetization",
            "Monetization",
            "willingness-to-pay has not been observed in any signal",
        ),
    ):
        dims.append(
            ScoreDimension(
                key=key,
                label=label,
                value=None,
                kind=EvidenceKind.UNKNOWN,
                reason="not measured: %s" % why,
            )
        )
        caveats.append("%s is unknown - %s." % (label, why))

    measured = [d.value for d in dims if d.value is not None]
    total = (
        int(round(sum(measured) / float(len(measured)))) if measured else 0
    )
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
