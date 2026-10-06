"""impara command-line interface."""
import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from . import __version__, add, average, multiply
from .cluster import cluster_signals
from .define import build_definition
from .extract import collect
from .models import Confidence, Opportunity, SourceHealth
from .profile import build_profile
from .llm import config_from_env
from .render import (
    render_definition,
    render_health,
    render_opportunity_list,
    render_problem_pool,
    render_profile,
    render_score,
    render_validation,
)
from .scoring import score_opportunity
from .store import find, load, save
from .understand import understand_batch
from .validate import build_validation_plan

SOURCE_CHOICES = ["github", "hackernews", "stackexchange"]


def _fmt(x) -> str:
    """Show 35.0 as 35, but keep 2.5 as 2.5."""
    return str(int(x)) if x == int(x) else str(x)


def _dump(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# Core pipeline
# --------------------------------------------------------------------------- #
def _collect_signals(args) -> "List[Any]":
    """Fetch and merge signals. Every command that reads evidence uses this."""
    sources = args.source or None
    quiet = bool(getattr(args, "json", False))

    def progress(msg: str) -> None:
        sink = getattr(args, "_progress", None)
        if sink is not None:
            sink(msg)
            return
        if not quiet:
            sys.stderr.write(msg)
            sys.stderr.flush()

    signals, health = collect(
        sources=sources,
        market=getattr(args, "market", "") or "",
        country=getattr(args, "country", "") or "",
        per_source=args.per_source,
        deep=args.deep,
        progress=progress,
    )

    from . import store

    # Evidence accumulates: every run merges into a local corpus so repeated
    # discovery builds corroboration instead of re-scattering each time.
    stored = [] if args.fresh else store.load_signals()
    known = {s.id for s in stored}
    new = [s for s in signals if s.id not in known]
    corpus = stored + new
    if not args.fresh:
        store.save_signals(signals)
        progress(
            "corpus: %d signal(s) total, %d new this run\n"
            % (len(corpus), len(new))
        )

    args._health = health
    args._signals = len(corpus)
    args._sources = len({s.source for s in corpus})
    return corpus


def _run_discovery(args) -> "List[Opportunity]":
    corpus = _collect_signals(args)
    quiet = bool(getattr(args, "json", False))

    def progress(msg: str) -> None:
        sink = getattr(args, "_progress", None)
        if sink is not None:
            sink(msg)
            return
        if not quiet:
            sys.stderr.write(msg)
            sys.stderr.flush()

    progress("clustering %d verified signal(s) ...\n" % len(corpus))
    problems = cluster_signals(corpus, threshold=args.threshold)

    opportunities: List[Opportunity] = []
    for problem in problems:
        profile = build_profile(problem)
        score = score_opportunity(problem)
        opportunities.append(Opportunity(problem=problem, profile=profile, score=score))

    opportunities.sort(key=lambda o: (-o.score.total, -len(o.problem.signals), o.problem.id))

    total_found = len(opportunities)
    floor = 1 if args.all_leads else max(1, args.min_signals)
    if floor > 1:
        opportunities = [o for o in opportunities if len(o.problem.signals) >= floor]
    args._filtered = total_found - len(opportunities)
    if args.min_evidence != "any":
        allowed = {
            "strong": [Confidence.STRONG],
            "moderate": [Confidence.MODERATE, Confidence.STRONG],
            "any": [Confidence.WEAK, Confidence.MODERATE, Confidence.STRONG, Confidence.NONE],
        }[args.min_evidence]
        opportunities = [o for o in opportunities if o.profile.evidence_strength in allowed]

    health: List[SourceHealth] = getattr(args, "_health", [])
    save(opportunities, health)
    return opportunities


def cmd_discover(args) -> int:
    opportunities = _run_discovery(args)
    health: List[SourceHealth] = getattr(args, "_health", [])

    if args.json:
        payload: Dict[str, Any] = {
            "version": __version__,
            "sources": [
                {
                    "key": h.key,
                    "label": h.label,
                    "status": h.status.value,
                    "note": h.note,
                    "signals_found": h.signals_found,
                }
                for h in health
            ],
            "count": len(opportunities),
            "opportunities": [o.to_dict() for o in opportunities],
        }
        if args.market:
            payload["market"] = args.market
        if args.country:
            payload["country"] = args.country
        _dump(payload)
        return 0

    if args.show_sources:
        print(render_health(health))
    print(
        render_opportunity_list(
            opportunities,
            limit=args.limit,
            filtered=getattr(args, "_filtered", 0),
            total_signals=getattr(args, "_signals", 0),
        )
    )

    if not opportunities:
        return 1
    return 0


def cmd_problems(args) -> int:
    """ingest -> filter -> understand -> verify -> present the pool."""
    signals = _collect_signals(args)
    config = config_from_env()
    result = understand_batch(signals, config=config)

    if args.json:
        payload = result.to_dict()
        payload["version"] = __version__
        payload["signal_count"] = len(signals)
        payload["source_count"] = len({s.source for s in signals})
        _dump(payload)
    else:
        print(render_problem_pool(result, limit=args.limit))

    # No model configured is a documented state, not a failure: the
    # deterministic pipeline is still doing useful work.
    if config is not None and not result.available:
        return 1
    return 0


def cmd_sources(args) -> int:
    from .sources import probe_sources

    health = probe_sources()
    if args.json:
        _dump(
            [
                {
                    "key": h.key,
                    "label": h.label,
                    "status": h.status.value,
                    "note": h.note,
                    "signals_found": h.signals_found,
                }
                for h in health
            ]
        )
        return 0
    print(render_health(health))
    return 0


def cmd_corpus(args) -> int:
    from . import store

    if args.clear:
        removed = store.clear_signals()
        if args.json:
            _dump({"cleared": removed, "size": store.corpus_size()})
        else:
            print("corpus cleared" if removed else "could not clear corpus")
        return 0 if removed else 1

    size = store.corpus_size()
    if args.json:
        _dump({"size": size, "path": store.SIGNALS_FILE})
        return 0
    print("Corpus: %d verified signal(s)" % size)
    print("Path:   %s" % store.SIGNALS_FILE)
    if size == 0:
        print("Run `impara discover` to start accumulating evidence.")
    return 0


def _load_or_explain(opp_id: str) -> Optional[Opportunity]:
    opp = find(opp_id)
    if opp:
        return opp
    loaded = load()
    if not loaded:
        print("No discovery run found. Run `impara discover` first.", file=sys.stderr)
    else:
        print(
            "Problem %s not found. Available: %s"
            % (opp_id, ", ".join(o.problem.id for o in loaded[0]) or "none"),
            file=sys.stderr,
        )
    return None


def cmd_investigate(args) -> int:
    opp = _load_or_explain(args.problem_id)
    if opp is None:
        return 1
    if args.json:
        _dump(opp.to_dict())
        return 0
    print(render_profile(opp.profile))
    print(render_score(opp.score))
    return 0


def cmd_validate(args) -> int:
    opp = _load_or_explain(args.problem_id)
    if opp is None:
        return 1
    plan = build_validation_plan(opp.problem, opp.profile)
    if args.json:
        _dump(plan.to_dict())
        return 0
    print(render_validation(plan))
    return 0


def cmd_define(args) -> int:
    opp = _load_or_explain(args.problem_id)
    if opp is None:
        return 1
    definition = build_definition(opp.problem, opp.profile)
    if args.json:
        _dump(definition.to_dict())
        return 0
    print(render_definition(definition))
    return 0


def cmd_tui(args) -> int:
    """Interactive terminal UI. Textual requires Python 3.9+."""
    try:
        from . import tui
    except ImportError as exc:
        sys.stderr.write(
            "The interactive UI needs Textual, which requires Python 3.9+.\n"
            "  %s\n"
            "Install it with:  pip install \"impara[tui]\"\n"
            "The plain CLI keeps working without it.\n" % exc
        )
        return 1

    return tui.run(show_all=args.all)


def cmd_list(args) -> int:
    loaded = load()
    if not loaded:
        print("No discovery run found. Run `impara discover` first.", file=sys.stderr)
        return 1
    opportunities, health = loaded
    if args.json:
        _dump({"count": len(opportunities), "ids": [o.problem.id for o in opportunities]})
        return 0
    if args.show_sources:
        print(render_health(health))
    print(render_opportunity_list(opportunities, limit=args.limit))
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument(
        "--limit",
        type=int,
        default=15,
        help="max problems to display (text mode, default 15)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="impara",
        description="Discover real, evidence-backed problems worth building.",
    )
    parser.add_argument("--version", action="version", version="impara %s" % __version__)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("discover", help="find recurring problems from real signals")
    _add_common(p)
    p.add_argument("--market", default="", help="filter to a market, e.g. education")
    p.add_argument("--country", default="", help="filter to a country, e.g. rwanda")
    p.add_argument(
        "--source",
        action="append",
        choices=SOURCE_CHOICES,
        help="restrict to a source (repeatable)",
    )
    p.add_argument("--per-source", type=int, default=30, help="candidates to fetch per query")
    p.add_argument("--threshold", type=float, default=0.34, help="clustering similarity 0-1")
    p.add_argument(
        "--min-signals",
        type=int,
        default=2,
        help="only report problems corroborated by at least N signals (default 2)",
    )
    p.add_argument(
        "--all",
        dest="all_leads",
        action="store_true",
        help="report single-signal leads too (implies --min-signals 1)",
    )
    p.add_argument(
        "--deep",
        action="store_true",
        help="expand matched threads for extra comments (slower, many more requests)",
    )
    p.add_argument(
        "--min-evidence",
        choices=["any", "weak", "moderate", "strong"],
        default="any",
        help="minimum evidence strength to report",
    )
    p.add_argument("--show-sources", action="store_true", help="include the source health table")
    p.add_argument(
        "--fresh",
        action="store_true",
        help="ignore the saved signal corpus and use only this run's signals",
    )
    p.set_defaults(func=cmd_discover)

    p = sub.add_parser("problems", help="problem statements backed by verbatim evidence")
    _add_common(p)
    p.add_argument(
        "--source",
        action="append",
        choices=SOURCE_CHOICES,
        help="restrict to a source (repeatable)",
    )
    p.add_argument("--per-source", type=int, default=30, help="candidates to fetch per query")
    p.add_argument("--market", default="", help="filter to a market, e.g. education")
    p.add_argument("--country", default="", help="filter to a country, e.g. rwanda")
    p.add_argument(
        "--deep",
        action="store_true",
        help="expand matched threads for extra comments (slower, many more requests)",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="ignore the saved signal corpus and use only this run's signals",
    )
    p.set_defaults(func=cmd_problems)

    p = sub.add_parser("sources", help="show which signal sources are reachable")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_sources)

    p = sub.add_parser("corpus", help="inspect or clear the accumulated signal corpus")
    p.add_argument("--clear", action="store_true", help="delete saved signals")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("list", help="re-show the last discovery run")
    _add_common(p)
    p.add_argument("--show-sources", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("investigate", help="full profile of one problem")
    p.add_argument("problem_id", help="e.g. #003 or 3")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_investigate)

    p = sub.add_parser("validate", help="validation plan for one problem")
    p.add_argument("problem_id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("define", help="turn a problem into a product opportunity")
    p.add_argument("problem_id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_define)

    p = sub.add_parser("tui", help="interactive terminal UI")
    p.add_argument("--all", action="store_true", help="include single-signal leads")
    p.set_defaults(func=cmd_tui)

    # -- back-compat calculator commands (published in 0.2.0) --
    p = sub.add_parser("add", help="add two numbers")
    p.add_argument("a", type=float)
    p.add_argument("b", type=float)
    p.set_defaults(func=lambda x: add(x.a, x.b))

    p = sub.add_parser("multiply", help="multiply two numbers")
    p.add_argument("a", type=float)
    p.add_argument("b", type=float)
    p.set_defaults(func=lambda x: multiply(x.a, x.b))

    p = sub.add_parser("average", help="average of one or more numbers")
    p.add_argument("numbers", type=float, nargs="+")
    p.set_defaults(func=lambda x: average(x.numbers))

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    # Calculator subcommands return a value; discovery subcommands return exit code.
    result = args.func(args)
    if isinstance(result, int):
        return result
    print(_fmt(result))
    return 0
