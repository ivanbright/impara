"""Data model for signals, problems, evidence and opportunity scores.

Two vocabulary rules are enforced here and must not be blurred:

* ``EvidenceKind.OBSERVED``  - something we literally read in a source.
* ``EvidenceKind.INFERRED``  - an assessment Impara computed.

Nothing that is ``INFERRED`` may be presented as though it were a fact
observed in the world.
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class EvidenceKind(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"


class Confidence(str, Enum):
    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class SourceStatus(str, Enum):
    OK = "ok"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"


@dataclass
class SourceHealth:
    """Honest report of what a source could actually do right now."""

    key: str
    label: str
    status: SourceStatus
    note: str = ""
    signals_found: int = 0

    @property
    def available(self) -> bool:
        return self.status == SourceStatus.OK


@dataclass
class Signal:
    """A single observed instance of someone describing a problem."""

    id: str
    source: str
    title: str
    excerpt: str
    url: str
    patterns: List[str]
    terms: List[str]
    author: str = ""
    engagement: int = 0
    observed_at: str = ""
    # Observed thread identity from the source (same HN story, same Stack
    # Exchange question). Empty when the source offers no thread structure.
    thread: str = ""

    @property
    def pattern_count(self) -> int:
        return len(self.patterns)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["patterns"] = list(self.patterns)
        d["terms"] = list(self.terms)
        return d


@dataclass
class EvidenceRecord:
    """One supporting item, with how it was obtained."""

    kind: EvidenceKind
    claim: str
    detail: str = ""
    url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "claim": self.claim,
            "detail": self.detail,
            "url": self.url,
        }


@dataclass
class Problem:
    """A cluster of related signals describing the same underlying problem."""

    id: str
    title: str
    signals: List[Signal] = field(default_factory=list)
    terms: List[str] = field(default_factory=list)

    def add_signal(self, signal: Signal) -> None:
        self.signals.append(signal)
        self.terms = sorted(set(self.terms) | set(signal.terms))

    @property
    def sources(self) -> List[str]:
        return sorted(set(s.source for s in self.signals))

    @property
    def authors(self) -> List[str]:
        return sorted(set(s.author for s in self.signals if s.author))

    @property
    def engagements(self) -> int:
        return sum(s.engagement for s in self.signals)

    @property
    def pattern_types(self) -> List[str]:
        seen: List[str] = []
        for s in self.signals:
            for p in s.patterns:
                if p not in seen:
                    seen.append(p)
        return seen

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "terms": list(self.terms),
            "signals": [s.to_dict() for s in self.signals],
            "sources": self.sources,
            "signal_count": len(self.signals),
        }


@dataclass
class ScoreDimension:
    """One axis of the opportunity score.

    ``kind`` is mandatory: an axis computed from observed counts is still an
    assessment, but an axis with no evidence behind it must be flagged so the
    user can see where Impara is guessing.
    """

    key: str
    label: str
    value: int  # 0-100
    kind: EvidenceKind
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "kind": self.kind.value,
            "reason": self.reason,
        }


@dataclass
class OpportunityScore:
    total: int
    dimensions: List[ScoreDimension]
    evidence_strength: Confidence
    caveats: List[str] = field(default_factory=list)

    @property
    def observed_ratio(self) -> float:
        if not self.dimensions:
            return 0.0
        obs = sum(1 for d in self.dimensions if d.kind is EvidenceKind.OBSERVED)
        return obs / float(len(self.dimensions))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "evidence_strength": self.evidence_strength.value,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "caveats": list(self.caveats),
        }


@dataclass
class ProblemProfile:
    """Everything we can say about one problem, evidence separated from inference."""

    problem: Problem
    who: str
    trying_to_accomplish: str
    why_difficult: str
    frequency: str
    workarounds: List[str]
    why_existing_insufficient: str
    evidence: List[EvidenceRecord]
    evidence_strength: Confidence
    severity: str
    paying_now: str
    product_type: str
    insufficient_evidence: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem.to_dict(),
            "who": self.who,
            "trying_to_accomplish": self.trying_to_accomplish,
            "why_difficult": self.why_difficult,
            "frequency": self.frequency,
            "workarounds": list(self.workarounds),
            "why_existing_insufficient": self.why_existing_insufficient,
            "evidence": [e.to_dict() for e in self.evidence],
            "evidence_strength": self.evidence_strength.value,
            "severity": self.severity,
            "paying_now": self.paying_now,
            "product_type": self.product_type,
            "insufficient_evidence": self.insufficient_evidence,
        }


@dataclass
class ValidationPlan:
    problem_id: str
    interviewees: List[str]
    questions: List[str]
    behaviors: List[str]
    alternatives_to_compare: List[str]
    assumptions_to_test: List[str]
    confirming_signals: List[str]
    invalidating_signals: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProductDefinition:
    problem_id: str
    problem_statement: str
    target_user: str
    job_to_be_done: str
    existing_alternatives: List[str]
    product_opportunity: str
    proposed_mvp: List[str]
    essential_features: List[str]
    excluded_features: List[str]
    business_model: str
    technical_complexity: str
    validation_risks: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Opportunity:
    """A discovered problem plus its assessment."""

    problem: Problem
    profile: ProblemProfile
    score: OpportunityScore

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.problem.id,
            "title": self.problem.title,
            "signal_count": len(self.problem.signals),
            "sources": self.problem.sources,
            "profile": self.profile.to_dict(),
            "score": self.score.to_dict(),
        }
