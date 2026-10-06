"""Cluster verified signals into distinct problems.

Two mechanisms, in order:

1. **Thread identity** - signals from the same HN story or Stack Exchange
   question are grouped directly. This is observed source structure, not an
   inference, and it is what makes aggregation reliable on short comments.
2. **Term similarity** - remaining signals (and distinct threads) are merged
   when their vocabulary overlaps enough.

Clustering is deterministic, so a repeated run over the same signals yields
the same problems.
"""
import re
from typing import Dict, List, Sequence

from .models import Problem, Signal
from .terms import extract_terms, similarity

DEFAULT_THRESHOLD = 0.34
MAX_MERGE_PASSES = 6


def cluster_signals(
    signals: Sequence[Signal],
    threshold: float = DEFAULT_THRESHOLD,
) -> List[Problem]:
    """Greedy clustering ordered by engagement."""
    if not signals:
        return []

    ordered = sorted(signals, key=lambda s: (-s.engagement, s.source, s.id))
    groups: List[List[Signal]] = _initial_groups(ordered)
    groups = _merge_groups(groups, threshold)

    problems = [_to_problem(g) for g in groups if g]
    problems.sort(key=lambda p: (-_cluster_weight(p), p.title))
    for i, problem in enumerate(problems, start=1):
        problem.id = "#%03d" % i
    return problems


def _initial_groups(ordered: Sequence[Signal]) -> List[List[Signal]]:
    """Bucket signals that share an observed thread, keep the rest alone."""
    threaded: Dict[str, List[Signal]] = {}
    groups: List[List[Signal]] = []
    for sig in ordered:
        if sig.thread:
            if sig.thread in threaded:
                threaded[sig.thread].append(sig)
            else:
                threaded[sig.thread] = [sig]
        else:
            groups.append([sig])
    # Emit thread buckets in signal order to keep ids stable.
    for bucket in threaded.values():
        groups.append(bucket)
    return groups


def _merge_groups(groups: List[List[Signal]], threshold: float) -> List[List[Signal]]:
    for _ in range(MAX_MERGE_PASSES):
        groups.sort(key=lambda g: (-len(g), -_weight(g), g[0].id))
        merged = False
        i = 0
        while i < len(groups):
            j = i + 1
            while j < len(groups):
                if _fits(groups[i], groups[j], threshold):
                    groups[i].extend(groups.pop(j))
                    merged = True
                else:
                    j += 1
            i += 1
        if not merged:
            break
    return groups


def _terms(group: Sequence[Signal]) -> List[str]:
    counts: Dict[str, int] = {}
    for s in group:
        for t in s.terms:
            counts[t] = counts.get(t, 0) + 1
    return [t for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def _fits(a: List[Signal], b: List[Signal], threshold: float) -> bool:
    # Two independent threads with no shared vocabulary are different problems.
    return similarity(_terms(a), _terms(b)) >= threshold


def _weight(group: Sequence[Signal]) -> int:
    return sum(10 + min(s.engagement, 100) for s in group)


def _cluster_weight(problem: Problem) -> int:
    return len(problem.signals) * 10 + min(problem.engagements, 200)


def _to_problem(group: List[Signal]) -> Problem:
    head = max(group, key=lambda s: s.engagement)
    terms = _merged_terms(group)
    problem = Problem(id="", title=_title_for(head, group), terms=terms)
    for s in group:
        problem.add_signal(s)
    problem.terms = terms
    return problem


def _merged_terms(group: List[Signal]) -> List[str]:
    counts: Dict[str, int] = {}
    for s in group:
        for t in s.terms:
            counts[t] = counts.get(t, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [t for t, _ in ranked[:14]]


def _title_for(head: Signal, group: List[Signal]) -> str:
    """Prefer the human-written title from the source - it is observed text,
    not something Impara composed. Only fall back to shared vocabulary."""
    raw = re.sub(r"\s+", " ", (head.title or "").strip())
    raw = raw.strip(".,;:-—|")
    if raw and len(raw) >= 8:
        return raw[:95]
    if len(group) > 1:
        shared = _merged_terms(group)
        if shared:
            return " | ".join(shared[:4])
    if raw:
        return raw[:95]
    return (head.excerpt or "Unnamed problem")[:95]
