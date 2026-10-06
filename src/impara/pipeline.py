"""High-level Python API.

``discover`` runs the full pipeline over *already-fetched* signals, so callers
can inject their own text sources and remain offline.
"""
from typing import List, Optional, Sequence, Tuple

from .cluster import cluster_signals
from .extract import collect as _collect
from .models import Opportunity, Problem, Signal, SourceHealth
from .profile import build_profile
from .scoring import score_opportunity


def discover(
    signals: Optional[Sequence[Signal]] = None,
    sources: Optional[List[str]] = None,
    market: str = "",
    country: str = "",
    threshold: float = 0.34,
    min_signals: int = 1,
    per_source: int = 30,
) -> Tuple[List[Opportunity], List[SourceHealth]]:
    """Return ``(opportunities, source_health)``.

    If ``signals`` is None, real signals are fetched from public sources.
    """
    health: List[SourceHealth] = []
    if signals is None:
        fetched, health = _collect(
            sources=sources, market=market, country=country, per_source=per_source
        )
        signals = fetched

    problems = cluster_signals(signals, threshold=threshold)
    opportunities: List[Opportunity] = []
    for problem in problems:
        if len(problem.signals) < min_signals:
            continue
        profile = build_profile(problem)
        score = score_opportunity(problem)
        opportunities.append(Opportunity(problem=problem, profile=profile, score=score))

    opportunities.sort(key=lambda o: (-o.score.total, -len(o.problem.signals), o.problem.id))
    return opportunities, health


def investigate(opportunities: Sequence[Opportunity], problem_id: str):
    """Look up one opportunity from a discovery result by id."""
    target = problem_id.strip()
    if target.isdigit():
        target = "#%03d" % int(target)
    elif not target.startswith("#"):
        target = "#" + target
    for o in opportunities:
        if o.problem.id == target:
            return o
    return None
