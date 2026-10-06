"""The understanding stage: raw signal text -> verifiable problem statements.

This is the only stage that needs a model, and it is the only stage allowed
to invent *structure*. It may not invent *facts*: every observed claim it
produces must survive ``statement.verify()`` against the fetched signal
text, or the statement is discarded.

When no model is configured the stage does not guess. It reports
``reason`` and returns no statements, and the CLI falls back to the
deterministic discovery pipeline the user already has.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .llm import LLMConfig, LLMError, chat_json, unavailable_reason
from .models import EvidenceKind, Signal
from .statement import (
    Claim,
    ProblemStatement,
    Quote,
    attach_provenance,
    normalize,
    verify,
)

MAX_SIGNAL_CHARS = 900
MIN_EVIDENCE_CHARS = 40

# Signals that describe nothing: they are too short to carry evidence, are a
# bare link, are a removal note, or are a reaction rather than a report.
_REMOVAL = re.compile(
    r"(?:\[(?:removed|deleted)\]"
    r"|\b(?:comment|post|reply|submission)\s+(?:has\s+been|was|is)\s+(?:removed|deleted)\b"
    r"|\bremoved by (?:a|the) moderator\b"
    r"|\bmoderators?\s+(?:removed|deleted)\b)",
    re.I,
)
_REACTION_ONLY = re.compile(
    r"^(?:lol|lmao|haha|heh|nice|cool|same|agreed|this|thanks|thank you|underrated"
    r"|word|facts|oof|yikes)[!.?\s]*$",
    re.I,
)
_BARE_LINK = re.compile(r"^(?:https?://|www\.)\S+$", re.I)
SYSTEM_ROLE = (
    "You help developers find real problems to build against. You are given "
    "RAW SIGNALS - text actually fetched from public sources - and you turn "
    "them into problem statements. You never state anything the signals do "
    "not show, and you never quote text that is not in front of you."
)

INSTRUCTION = """Write PROBLEM STATEMENTS for problems the signals actually describe.

Rules:
1. Only cite signal ids from the list below.
2. Every quote must be a VERBATIM, CONTIGUOUS substring of that signal's
   excerpt. Copy it character for character: no paraphrasing, no tidy-up,
   no ellipses in the middle.
3. Skip signals where the text is a joke, an insult, moderation or support
   commentary, or merely a link/thread title - rather than a description of
   something not working in someone's life or workflow.
4. Do not invent problems no signal describes. Fewer statements, all
   evidenced, beat more statements, some invented.
5. What a person wants or how they would pay is not observed unless they
   wrote it. Put those in "unanswered".
6. "observed" claims need quotes. Anything you infer is not a claim - it
   belongs in "unanswered".

Return ONLY JSON, no prose:
{
  "problems": [
    {
      "statement": "one sentence: who is stuck, on what, and why it matters",
      "domain": "short market label",
      "who": ["role or context"],
      "claims": [
        {"text": "the claim", "kind": "observed",
         "quotes": [{"signal_id": "sig-001", "quote": "verbatim text"}]}
      ],
      "unanswered": ["open question"]
    }
  ]
}"""


@dataclass
class UnderstandingResult:
    statements: List[ProblemStatement] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)
    screened_out: List[str] = field(default_factory=list)
    reason: str = ""
    model: str = ""
    signals_seen: int = 0

    @property
    def available(self) -> bool:
        return not self.reason

    @property
    def dropped(self) -> int:
        return len(self.rejected)

    @property
    def screened(self) -> int:
        return len(self.screened_out)

    @property
    def usable_signals(self) -> int:
        return self.signals_seen - self.screened

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "model": self.model,
            "signals_seen": self.signals_seen,
            "usable_signals": self.usable_signals,
            "screened_out": list(self.screened_out),
            "understanding": "unavailable" if not self.available else "llm",
            "dropped": self.dropped,
            "rejections": list(self.rejected),
            "problems": [s.to_dict() for s in self.statements],
        }


def describe_signals(signals: Sequence[Signal], limit: int = MAX_SIGNAL_CHARS) -> str:
    """Render signals compactly and unambiguously for the prompt."""
    lines = []
    for s in signals:
        excerpt = " ".join((s.excerpt or s.title or "").split())
        if len(excerpt) > limit:
            excerpt = excerpt[:limit] + "..."
        lines.append(
            "[%s] source=%s author=%s url=%s\n    %s"
            % (s.id, s.source, s.author or "-", s.url or "-", excerpt)
        )
    return "\n".join(lines)


def screen_reason(signal: Signal) -> str:
    """Why this signal is not usable evidence, or '' when it is usable."""
    text = normalize(signal.excerpt or "")
    if not text:
        return "no fetched text"
    if len(text) < MIN_EVIDENCE_CHARS:
        return "too short to be evidence (%d chars)" % len(text)
    if _BARE_LINK.match(text.rstrip(".,;")):
        return "bare link with no description"
    if _REMOVAL.search(text):
        return "removal or moderation note"
    if _REACTION_ONLY.match(text):
        return "reaction rather than a description of a problem"
    return ""


def screen_signals(
    signals: Sequence[Signal],
) -> Tuple[List[Signal], List[str]]:
    """Content filter: drop signals that cannot describe a problem.

    Deliberately conservative - it only removes text that cannot possibly be
    evidence, so it never silently rewrites what a human actually said.
    """
    kept: List[Signal] = []
    skipped: List[str] = []
    for signal in signals:
        reason = screen_reason(signal)
        if reason:
            skipped.append("%s: %s" % (signal.id, reason))
        else:
            kept.append(signal)
    return kept, skipped


def build_prompt(signals: Sequence[Signal]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_ROLE},
        {
            "role": "user",
            "content": "%s\n\nSIGNALS (%d):\n%s"
            % (INSTRUCTION, len(signals), describe_signals(signals)),
        },
    ]


def _as_claim(raw: Any) -> Optional[Claim]:
    if not isinstance(raw, dict):
        return None
    text = str(raw.get("text") or "").strip()
    if not text:
        return None
    kind_raw = str(raw.get("kind") or "observed").strip().lower()
    kind = EvidenceKind.INFERRED if kind_raw.startswith("infer") else EvidenceKind.OBSERVED

    quotes: List[Quote] = []
    for item in raw.get("quotes") or []:
        if not isinstance(item, dict):
            continue
        signal_id = str(item.get("signal_id") or item.get("id") or "").strip()
        quote_text = str(item.get("quote") or item.get("text") or "").strip()
        if signal_id and quote_text:
            quotes.append(Quote(text=quote_text, signal_id=signal_id))
    return Claim(text=text, quotes=quotes, kind=kind)


def _as_statement(raw: Any) -> Optional[ProblemStatement]:
    if not isinstance(raw, dict):
        return None
    statement = str(raw.get("statement") or "").strip()
    if not statement:
        return None
    claims = [c for c in (_as_claim(c) for c in (raw.get("claims") or [])) if c]
    return ProblemStatement(
        statement=statement,
        domain=str(raw.get("domain") or "").strip(),
        who=[str(w).strip() for w in (raw.get("who") or []) if str(w).strip()],
        claims=claims,
        signal_ids=list({q.signal_id for c in claims for q in c.quotes}),
        sources=[],
        unanswered=[str(u).strip() for u in (raw.get("unanswered") or []) if str(u).strip()],
        understanding="llm",
    )


def prune_untraceable(statement: ProblemStatement) -> int:
    """Drop observed claims that carry no quote.

    Untraceable -> discarded, per the invariant. Inferred claims are kept
    because they are already labelled as not observed.
    """
    kept: List[Claim] = []
    removed = 0
    for claim in statement.claims:
        if claim.kind is EvidenceKind.OBSERVED and not claim.quotes:
            removed += 1
            continue
        kept.append(claim)
    statement.claims = kept
    return removed


def parse_statements(
    payload: Any,
    signals: Sequence[Signal],
) -> Tuple[List[ProblemStatement], List[str]]:
    """Turn a model reply into verified statements; return kept + rejections."""
    if not isinstance(payload, dict) or not isinstance(payload.get("problems"), list):
        raise LLMError('model reply did not contain a "problems" list')

    by_id = {s.id: s for s in signals}
    kept: List[ProblemStatement] = []
    rejected: List[str] = []

    for raw in payload["problems"]:
        statement = _as_statement(raw)
        if statement is None:
            rejected.append("no statement text")
            continue

        prune_untraceable(statement)
        attach_provenance(statement, signals)

        unknown = [sid for sid in statement.signal_ids if sid not in by_id]
        if unknown:
            rejected.append(
                "%s -> cites unknown signal(s) %s"
                % (statement.statement[:48], ", ".join(unknown))
            )
            continue
        statement.sources = sorted(
            {by_id[sid].source for sid in statement.signal_ids if sid in by_id}
        )

        report = verify(statement, signals)
        if not report.ok:
            rejected.append("%s -> %s" % (statement.statement[:48], report.summary))
            continue
        kept.append(statement)

    return kept, rejected


def understand(
    signals: Sequence[Signal],
    config: Optional[LLMConfig] = None,
    transport: Optional[Any] = None,
) -> UnderstandingResult:
    """Understand a batch of signals. Never raises for 'no model configured'."""
    if config is None:
        return UnderstandingResult(
            statements=[],
            rejected=[],
            screened_out=[],
            reason=unavailable_reason() or "no model configured",
            model="",
            signals_seen=len(signals),
        )
    if not signals:
        return UnderstandingResult(reason="", model=config.model, signals_seen=0)

    usable, skipped = screen_signals(signals)
    if not usable:
        return UnderstandingResult(
            statements=[],
            rejected=[],
            screened_out=skipped,
            reason="every signal was screened out by the content filter",
            model=config.model,
            signals_seen=len(signals),
        )

    try:
        payload = chat_json(config, build_prompt(usable), transport=transport)
        statements, rejected = parse_statements(payload, usable)
    except LLMError as exc:
        return UnderstandingResult(
            statements=[],
            rejected=[],
            screened_out=skipped,
            reason=str(exc),
            model=config.model,
            signals_seen=len(signals),
        )
    return UnderstandingResult(
        statements=statements,
        rejected=rejected,
        screened_out=skipped,
        reason="",
        model=config.model,
        signals_seen=len(signals),
    )


def understand_batch(
    signals: Sequence[Signal],
    config: Optional[LLMConfig] = None,
    transport: Optional[Any] = None,
    batch_size: int = 40,
) -> UnderstandingResult:
    """Run understanding over ``signals`` in bounded batches."""
    if config is None:
        return understand(signals, config=config, transport=transport)
    if batch_size < 1:
        batch_size = 1

    merged = UnderstandingResult(model=config.model, signals_seen=len(signals))
    for start in range(0, len(signals), batch_size):
        chunk = signals[start:start + batch_size]
        result = understand(chunk, config=config, transport=transport)
        merged.statements.extend(result.statements)
        merged.rejected.extend(result.rejected)
        merged.screened_out.extend(result.screened_out)
        if result.reason:
            merged.reason = result.reason
            break
    return merged
