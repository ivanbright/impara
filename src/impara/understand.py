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
from .models import Signal
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

# ---- Evidence boundary -------------------------------------------------- #
# The model may still slip into solution-writing, product wishes, or
# un-evidenced assessments even with a tight prompt. These checks are the
# deterministic wall: a candidate that fails them never enters the pool.
_EMOTION_ONLY = re.compile(
    r"^\s*(?:(?:this|that|it)(?:'s| is)?|honestly|seriously|just)"
    r"(?:\s+(?:honestly|seriously|just|so|really|too|totally|extremely|super|very)){0,3}"
    r"\s+(?:annoying|frustrating|terrible|lame|awful|ridiculous|horrible|stupid|dumb|"
    r"bad|crappy|unusable|useless|irritating|infuriating|janky|buggy|broken)"
    r"[.!?]*$",
    re.I,
)
_SOLUTION_PHASED = re.compile(
    r"^(?:build|create|develop|design|ship|launch|make|add|introduce|implement|"
    r"provide|offer|let users|give users|enable users|allow users|empower users)\b",
    re.I,
)
_SOLUTION_VERB = re.compile(
    r"\b(?:users?|people|customers|developers|founders|businesses|companies?|"
    r"the (?:company|team|app|system|product)|we)\s+"
    r"(?:need|want|should|could|would like|deserve|require)\s+"
    r"(?:(?:an?|the|to)\s+)?"
    r"(?:(?:[\w'-]+\s+){0,4})?"
    r"(?:add|build|create|develop|implement|provide|offer|make|ship|introduce|get\s+)?"
    r"(?:[\w'-]+\s+){0,3}"
    r"(?:app|tool|service|platform|api|system|product|solution|software|extension)\b",
    re.I,
)
_ASPIRATION_PRODUCT = re.compile(
    r"\bi wish there (?:was|were) an? (?:app|tool|service|extension|plugin|bot|website|platform)\b"
    r"|\bi want an? (?:app|tool|service|extension|plugin|bot|website|platform)\b",
    re.I,
)
_GENERIC_PITCH = re.compile(
    r"\bthere should be an? (?:app|tool|service|platform|bot|solution)\b"
    r"|\bnext big thing\b|\bgame[- ]changer\b"
    r"|\b(?:huge|big|massive|enormous|great) (?:opportunity|idea|market)\b"
    r"|\bhelp businesses succeed\b|\bimprove user experience\b"
    r"|\bincrease productivity\b"
    r"|\b(?:users?|people|customers?) need a better (?:solution|tool|app)\b",
    re.I,
)
_ASSESSMENT_TOKEN = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d+)?"
    r"|\d+(?:\.\d+)?\s?%|\b\d+(?:\.\d+)?\s*percent\b"
    r"|\b\d[\d,]*(?:\.[\d,]+)?\s*[kmb](?:illion)?\b"
    r"|\b\d{3,}\b"
    r"|\b(?:dozens|hundreds|thousands|millions|billions) of (?:users|people|organizations|"
    r"companies|businesses|developers|customers)\b"
    r"|\b(?:would pay|willing to pay|ready to pay|are paying|market size|gross margin|"
    r"adoption|churn|demand|revenue|pricing|competitive moat|unit economics)\b",
    re.I,
)
# A bare interrogative asserts nothing about a workflow: "Why doesn't GitHub
# support X?" restates an absence as a question. The evidence must report what
# failed, not merely ask about it - unless the question itself carries a
# failure description ("Why does this crash on startup?").
_QUESTION_LEAD = re.compile(
    r"^\s*(?:why|how|when|where|what|who|which|whose|is|are|was|were|do|does|did"
    r"|can|could|will|would|should|has|have|had|may|might|anyone|anybody)\b",
    re.I,
)
_OBSERVED_FAILURE = re.compile(
    r"\b(?:fail(?:s|ed|ing)?|errors?|crash(?:es|ed|ing)?|broken|bugs?|buggy"
    r"|can(?:no|['’])?t|unable|no way|missing|stuck|hang(?:s|ing)?|freez(?:es|ing)?"
    r"|does (?:not|n['’]?t) work|won['’]?t work|work(?:s)? at all"
    r"|manual(?:ly)?|workaround|blocked|impossible|deleted|corrupt(?:ed|ion|ing)?"
    r"|unusable|denied|reject(?:s|ed|ing)?|refus(?:es|ed|ing)|annoy(?:s|ed|ing)?"
    r"|frustrat(?:es|ed|ing)|pain|slow(?:er)?|losing|lost)\b",
    re.I,
)
SYSTEM_ROLE = (
    "You are an evidence analyst for a developer problem-discovery tool. You "
    "read RAW SIGNALS - text fetched verbatim from public sources - and decide "
    "whether each one describes a real, concrete problem a software or backend "
    "engineer could investigate. You are NOT a startup-idea generator and you "
    "do NOT write solutions. You report only what a signal literally shows, "
    "and you only ever quote text that is in front of you, character for "
    "character."
)

INSTRUCTION = """For every signal below, decide whether it describes a genuine, tackleable problem.

A real problem is CONCRETE: someone's workflow is failing, a capability is
missing, something breaks, a task happens by hand, data fails to move or line
up, an integration breaks, a search/retrieval or sync step fails, a
notification is missed, or access/control is wrong. A developer should be able
to read the statement and think: "I understand what is failing, and I can
investigate how to tackle it."

Set has_problem to FALSE, with a one-line reject_reason, when the signal is:
- a wish for a product ("I wish there was an app for X") or a proposed
  solution ("users need a tool that ...")
- a bare question ("Why doesn't X support Y?") that only asks for an
  explanation instead of describing a concrete workflow failure
- hype, generic advice, a joke, an insult, or emotional venting with no
  mechanism behind it
- commentary about moderation, support, or the community itself
- vague: "improve user experience", "increase productivity", "a big
  opportunity", "users need a better solution"
- anything the signal does not actually say

Return EXACTLY ONE entry per signal, even when has_problem is false:

{
  "problems": [
    {
      "signal_id": "exact id from SIGNALS",
      "has_problem": true,
      "problem": "one concise sentence naming what is failing and for whom,
                  written for a software/backend engineer; describe the
                  mechanism, never prescribe the fix",
      "who": ["roles the signal itself names"],
      "domain": "short technical label, e.g. billing automation",
      "evidence": "VERBATIM CONTIGUOUS substring of that signal's excerpt that
                   supports the problem. Copy it exactly: keep the author's
                   typos and grammar, no paraphrase, no ellipses inside",
      "unanswered": ["what the text leaves open", "..."]
    },
    {
      "signal_id": "...",
      "has_problem": false,
      "reject_reason": "one line: why this signal is not a problem"
    }
  ]
}

Rules:
1. Bind the statement to the quote: every fact in "problem" must be supported
   by "evidence" and nothing else. Never add severity, frequency, causes,
   scale, market size, or business impact the quote does not contain.
2. Never propose a solution in "problem" - name the failure, not the remedy.
3. One entry per signal. Never merge unrelated signals; a problem that appears
   only once stays a single-signal entry.
4. Prefer describing the failure in engineering terms: workflows, missing
   capabilities, manual processes, data movement, integration, search/retrieval,
   synchronization, processing, notification, access/control.
5. "who" may only name roles the signal itself states.
6. Fewer honest problems beat more invented ones.

Write COMPACT JSON: one line per entry, no indentation, so every signal fits
in a single response and the reply is never truncated.

Return ONLY JSON, no prose."""


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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "1", "y"}


def _statement_from_entry(raw: Any, by_id: Dict[str, Signal]) -> Optional[ProblemStatement]:
    """Turn one per-signal candidate into a verifiable statement.

    The candidate carries the problem text and exactly one verbatim evidence
    span from exactly one signal, so the CLAIM -> QUOTE -> SIGNAL ID -> URL
    chain has no room to fuse unrelated sources.
    """
    statement_text = str(raw.get("problem") or "").strip()
    evidence = str(raw.get("evidence") or "").strip()
    signal_id = str(raw.get("signal_id") or "").strip()
    if not statement_text or not evidence or not signal_id:
        return None
    if signal_id not in by_id:
        signal_id = ""
    return ProblemStatement(
        statement=statement_text,
        domain=str(raw.get("domain") or "").strip(),
        who=[str(w).strip() for w in (raw.get("who") or []) if str(w).strip()],
        claims=[Claim(text=statement_text, quotes=[Quote(text=evidence, signal_id=signal_id)])],
        signal_ids=[signal_id] if signal_id else [],
        sources=[],
        unanswered=[str(u).strip() for u in (raw.get("unanswered") or []) if str(u).strip()],
        understanding="llm",
    )


def _quote_text(statement: ProblemStatement) -> str:
    return normalize(
        " ".join(q.text for claim in statement.claims for q in claim.quotes)
    ).lower()


def _violation_reason(statement: ProblemStatement) -> str:
    """Why this candidate must not enter the pool, or '' when it should."""
    text = normalize(statement.statement)
    if len(text) < 12:
        return "problem statement too short to name a concrete problem"
    if text.endswith("?"):
        return "statement is phrased as a question rather than naming a problem"
    if _EMOTION_ONLY.match(text):
        return "emotional venting with no concrete mechanism"
    if _SOLUTION_PHASED.match(text) or _SOLUTION_VERB.search(text):
        return "statement proposes a solution instead of naming a problem"
    if _ASPIRATION_PRODUCT.search(text) or _GENERIC_PITCH.search(text):
        return "generic wish or opportunity language instead of a concrete problem"

    haystack = _quote_text(statement)
    if _QUESTION_LEAD.match(haystack) and haystack.rstrip().endswith("?"):
        if not _OBSERVED_FAILURE.search(haystack):
            return "evidence is a bare question and does not report what failed"
    for match in _ASSESSMENT_TOKEN.finditer(text):
        token = match.group(0).lower()
        if token not in haystack:
            return "asserts '%s' that the supporting evidence does not contain" % match.group(0)
    return ""


def parse_statements(
    payload: Any,
    signals: Sequence[Signal],
) -> Tuple[List[ProblemStatement], List[str]]:
    """Turn a model reply into verified statements; return kept + rejections.

    Three independent walls stand between the model reply and the pool:
    1. candidate shape and per-signal provenance,
    2. the deterministic evidence boundary (solution/wish/assessment checks),
    3. the verbatim verifier in ``statement.verify()``.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("problems"), list):
        raise LLMError('model reply did not contain a "problems" list')

    by_id = {s.id: s for s in signals}
    kept: List[ProblemStatement] = []
    rejected: List[str] = []

    for raw in payload["problems"]:
        if not isinstance(raw, dict):
            rejected.append("malformed entry (not an object)")
            continue

        signal_id = str(raw.get("signal_id") or "").strip()
        if not _as_bool(raw.get("has_problem")) or not signal_id:
            reason = str(raw.get("reject_reason") or "").strip()
            rejected.append(
                "%s: %s" % (signal_id or "-", reason or "signal judged not to describe a problem")
            )
            continue

        unknown = signal_id not in by_id
        statement = _statement_from_entry(raw, by_id)
        if statement is None:
            rejected.append("%s: missing problem statement or evidence" % signal_id)
            continue

        if unknown:
            rejected.append("%s -> cites unknown signal %s" % (statement.statement[:48], signal_id))
            continue

        statement.sources = [by_id[signal_id].source]
        attach_provenance(statement, signals)

        violation = _violation_reason(statement)
        if violation:
            rejected.append("%s -> %s" % (statement.statement[:48], violation))
            continue

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
    """Run understanding over ``signals`` in bounded batches.

    The prompt asks for compact, one-line-per-entry JSON so a full batch of
    per-signal entries fits inside the model's output limit.
    """
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
