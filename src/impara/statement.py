"""Problem statements, and the provenance rule that keeps them honest.

The product's whole value rests on one invariant:

    CLAIM -> VERBATIM QUOTE -> SIGNAL ID -> URL

A statement that cannot be traced back to fetched source text is not
evidence, so it is labelled ``INFERRED`` or dropped. ``verify()`` is the
enforcement point: it re-checks every quote against the signal text it
claims to come from, using whitespace-normalised substring matching. That
tolerates the line wrapping an LLM may introduce, and nothing else -
changing a single word still fails.
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from .models import EvidenceKind, Signal


def normalize(text: str) -> str:
    """Collapse whitespace so layout differences do not read as edits."""
    return " ".join((text or "").split())


@dataclass
class Quote:
    """A verbatim span lifted from one signal's fetched text."""

    text: str
    signal_id: str
    url: str = ""
    source: str = ""
    author: str = ""

    @property
    def provenance(self) -> str:
        return "%s (%s) %s" % (self.signal_id, self.source or "?", self.url or "<no url>")

    def to_dict(self) -> Dict[str, str]:
        return {
            "text": self.text,
            "signal_id": self.signal_id,
            "url": self.url,
            "source": self.source,
            "author": self.author,
        }


@dataclass
class Claim:
    """One assertion the statement makes, carried by quotes that back it."""

    text: str
    quotes: List[Quote] = field(default_factory=list)
    kind: EvidenceKind = EvidenceKind.OBSERVED

    @property
    def urls(self) -> List[str]:
        return [q.url for q in self.quotes if q.url]

    def to_dict(self) -> Dict[str, object]:
        return {
            "text": self.text,
            "kind": self.kind.value,
            "quotes": [q.to_dict() for q in self.quotes],
        }


@dataclass
class ProblemStatement:
    """A pool entry: what the problem is, who has it, and how we know."""

    statement: str
    domain: str = ""
    who: List[str] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)
    signal_ids: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    unanswered: List[str] = field(default_factory=list)
    # "llm" when a model wrote it, "unavailable" when no model was configured.
    understanding: str = "unavailable"

    @property
    def quote_count(self) -> int:
        return sum(len(c.quotes) for c in self.claims)

    @property
    def urls(self) -> List[str]:
        seen: List[str] = []
        for claim in self.claims:
            for url in claim.urls:
                if url and url not in seen:
                    seen.append(url)
        return seen

    @property
    def evidence_signals(self) -> List[str]:
        seen: List[str] = []
        for claim in self.claims:
            for quote in claim.quotes:
                if quote.signal_id and quote.signal_id not in seen:
                    seen.append(quote.signal_id)
        return seen

    @property
    def inferred_claims(self) -> int:
        return sum(1 for c in self.claims if c.kind is not EvidenceKind.OBSERVED)

    def to_dict(self) -> Dict[str, object]:
        return {
            "statement": self.statement,
            "domain": self.domain,
            "who": list(self.who),
            "understanding": self.understanding,
            "claims": [c.to_dict() for c in self.claims],
            "signal_ids": list(self.signal_ids),
            "sources": list(self.sources),
            "unanswered": list(self.unanswered),
            "urls": self.urls,
            "quote_count": self.quote_count,
        }


@dataclass
class VerificationReport:
    ok: bool
    checked: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.ok:
            return "%d quote(s) verified against source text" % self.checked
        return "; ".join(self.errors[:5])

    def to_dict(self) -> Dict[str, object]:
        return {"ok": self.ok, "checked": self.checked, "errors": list(self.errors)}


def attach_provenance(statement: ProblemStatement, signals: Sequence[Signal]) -> int:
    """Fill url/source/author on every quote from the signal it cites.

    Provenance is copied from the fetched record rather than trusted from
    whoever wrote the claim, so it can never disagree with the corpus.
    """
    index: Dict[str, Signal] = {s.id: s for s in signals}
    filled = 0
    for claim in statement.claims:
        for quote in claim.quotes:
            source_signal = index.get(quote.signal_id)
            if source_signal is None:
                continue
            changed = (
                quote.url != source_signal.url
                or quote.source != source_signal.source
                or quote.author != source_signal.author
            )
            quote.url = source_signal.url
            quote.source = source_signal.source
            quote.author = source_signal.author
            if changed:
                filled += 1
    return filled


def verify(
    statement: ProblemStatement,
    signals: Sequence[Signal],
) -> VerificationReport:
    """Re-check every observed claim against the signal text it cites."""
    index: Dict[str, Signal] = {s.id: s for s in signals}
    errors: List[str] = []
    checked = 0
    tag = statement.statement[:60]

    if not statement.statement.strip():
        errors.append("empty statement")

    for signal_id in statement.signal_ids:
        if signal_id not in index:
            errors.append("statement cites unknown signal %s" % signal_id)

    observed_claims = 0
    for claim in statement.claims:
        if claim.kind is not EvidenceKind.OBSERVED:
            continue
        observed_claims += 1
        if not claim.quotes:
            errors.append("observed claim has no quote: %s [%s]" % (claim.text[:50], tag))
            continue
        if not claim.text.strip():
            errors.append("observed claim has no text [%s]" % tag)
        for quote in claim.quotes:
            checked += 1
            if not quote.text.strip():
                errors.append("empty quote on signal %s [%s]" % (quote.signal_id, tag))
                continue
            source_signal = index.get(quote.signal_id)
            if source_signal is None:
                errors.append(
                    "quote cites signal %s which is not in the corpus [%s]"
                    % (quote.signal_id, tag)
                )
                continue
            haystack = normalize(source_signal.excerpt)
            if normalize(quote.text) not in haystack:
                errors.append(
                    "quote is not verbatim in signal %s [%s]"
                    % (quote.signal_id, tag)
                )
                continue
            if quote.url and source_signal.url and quote.url != source_signal.url:
                errors.append(
                    "quote for signal %s carries the wrong URL [%s]"
                    % (quote.signal_id, tag)
                )
                continue
            if quote.source and source_signal.source and quote.source != source_signal.source:
                errors.append(
                    "quote for signal %s carries the wrong source [%s]"
                    % (quote.signal_id, tag)
                )

    if not observed_claims:
        errors.append("statement has no observed evidence [%s]" % tag)

    return VerificationReport(ok=not errors, checked=checked, errors=errors)


def verify_all(
    statements: Iterable[ProblemStatement],
    signals: Sequence[Signal],
) -> Dict[str, VerificationReport]:
    """Verify a pool. Unverifiable statements are dropped by the caller."""
    return {s.statement: verify(s, signals) for s in statements}
