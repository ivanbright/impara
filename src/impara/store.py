"""Persist the last discovery run so ``investigate`` / ``validate`` / ``define``
can work on previously fetched problems without hitting the network again.
"""
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import (
    Confidence,
    EvidenceKind,
    EvidenceRecord,
    Opportunity,
    OpportunityScore,
    Problem,
    ProblemProfile,
    ScoreDimension,
    Signal,
    SourceHealth,
    SourceStatus,
)

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".impara")
RUN_FILE = os.path.join(CACHE_DIR, "last_run.json")
SIGNALS_FILE = os.path.join(CACHE_DIR, "signals.json")

# The corpus accumulates so that repeated runs build real evidence rather than
# re-deriving a scattered snapshot each time.
MAX_CORPUS = 5000


def _dir() -> str:
    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR


def load_signals() -> List[Signal]:
    try:
        with open(SIGNALS_FILE, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    out: List[Signal] = []
    for item in raw:
        try:
            out.append(_signal(item))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def save_signals(signals: Sequence[Signal]) -> Optional[str]:
    """Merge into the persistent corpus, de-duplicated by signal id."""
    try:
        _dir()
        merged: Dict[str, Dict[str, Any]] = {
            s.id: s.to_dict() for s in load_signals()
        }
        for s in signals:
            merged[s.id] = s.to_dict()
        items = list(merged.values())[:MAX_CORPUS]
        with open(SIGNALS_FILE, "w", encoding="utf-8") as fh:
            json.dump(items, fh)
        return SIGNALS_FILE
    except OSError:
        return None


def clear_signals() -> bool:
    try:
        if os.path.exists(SIGNALS_FILE):
            os.remove(SIGNALS_FILE)
        return True
    except OSError:
        return False


def corpus_size() -> int:
    return len(load_signals())


def save(opportunities: List[Opportunity], health: List[SourceHealth]) -> Optional[str]:
    try:
        _dir()
        payload = {
            "opportunities": [o.to_dict() for o in opportunities],
            "health": [
                {
                    "key": h.key,
                    "label": h.label,
                    "status": h.status.value,
                    "note": h.note,
                    "signals_found": h.signals_found,
                }
                for h in health
            ],
        }
        with open(RUN_FILE, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return RUN_FILE
    except OSError:
        return None


def load() -> Optional[Tuple[List[Opportunity], List[SourceHealth]]]:
    try:
        with open(RUN_FILE, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    opps: List[Opportunity] = []
    for raw in payload.get("opportunities") or []:
        try:
            opps.append(_opp_from_dict(raw))
        except (KeyError, ValueError, TypeError):
            continue

    health: List[SourceHealth] = []
    for raw in payload.get("health") or []:
        try:
            health.append(
                SourceHealth(
                    key=raw["key"],
                    label=raw.get("label", raw["key"]),
                    status=SourceStatus(raw.get("status", "unavailable")),
                    note=raw.get("note", ""),
                    signals_found=int(raw.get("signals_found", 0)),
                )
            )
        except (KeyError, ValueError, TypeError):
            continue
    return opps, health


def find(opp_id: str) -> Optional[Opportunity]:
    loaded = load()
    if not loaded:
        return None
    opportunities, _ = loaded
    target = opp_id.strip()
    if not target.startswith("#"):
        target = "#" + target
    target = target.zfill(4) if target[1:].isdigit() else target
    for o in opportunities:
        if o.problem.id.lower() == target.lower():
            return o
    # Allow bare numbers like "17" -> "#017"
    if opp_id.strip().isdigit():
        want = "#%03d" % int(opp_id.strip())
        for o in opportunities:
            if o.problem.id == want:
                return o
    return None


def ids() -> List[str]:
    loaded = load()
    if not loaded:
        return []
    return [o.problem.id for o in loaded[0]]


# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #
def _signal(d: Dict[str, Any]) -> Signal:
    return Signal(
        id=d["id"],
        source=d["source"],
        title=d.get("title", ""),
        excerpt=d.get("excerpt", ""),
        url=d.get("url", ""),
        patterns=list(d.get("patterns", [])),
        terms=list(d.get("terms", [])),
        author=d.get("author", ""),
        engagement=int(d.get("engagement", 0)),
        observed_at=d.get("observed_at", ""),
        thread=d.get("thread", ""),
    )


def _problem(d: Dict[str, Any]) -> Problem:
    p = Problem(id=d["id"], title=d.get("title", ""), terms=list(d.get("terms", [])))
    for s in d.get("signals", []):
        p.add_signal(_signal(s))
    p.terms = list(d.get("terms", []))
    return p


def _dimension(d: Dict[str, Any]) -> ScoreDimension:
    return ScoreDimension(
        key=d["key"],
        label=d.get("label", d["key"]),
        value=int(d["value"]),
        kind=EvidenceKind(d["kind"]),
        reason=d.get("reason", ""),
    )


def _score(d: Dict[str, Any]) -> OpportunityScore:
    return OpportunityScore(
        total=int(d["total"]),
        dimensions=[_dimension(x) for x in d.get("dimensions", [])],
        evidence_strength=Confidence(d.get("evidence_strength", "none")),
        caveats=list(d.get("caveats", [])),
    )


def _evidence(d: Dict[str, Any]) -> EvidenceRecord:
    return EvidenceRecord(
        kind=EvidenceKind(d["kind"]),
        claim=d.get("claim", ""),
        detail=d.get("detail", ""),
        url=d.get("url", ""),
    )


def _profile(d: Dict[str, Any], problem: Problem) -> ProblemProfile:
    return ProblemProfile(
        problem=problem,
        who=d.get("who", ""),
        trying_to_accomplish=d.get("trying_to_accomplish", ""),
        why_difficult=d.get("why_difficult", ""),
        frequency=d.get("frequency", ""),
        workarounds=list(d.get("workarounds", [])),
        why_existing_insufficient=d.get("why_existing_insufficient", ""),
        evidence=[_evidence(x) for x in d.get("evidence", [])],
        evidence_strength=Confidence(d.get("evidence_strength", "none")),
        severity=d.get("severity", ""),
        paying_now=d.get("paying_now", ""),
        product_type=d.get("product_type", ""),
        insufficient_evidence=bool(d.get("insufficient_evidence", False)),
    )


def _opp_from_dict(d: Dict[str, Any]) -> Opportunity:
    profile_raw = d["profile"]
    problem = _problem(profile_raw["problem"])
    profile = _profile(profile_raw, problem)
    return Opportunity(problem=problem, profile=profile, score=_score(d["score"]))
