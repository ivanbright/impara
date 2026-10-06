"""Offline tests for the interactive TUI.

Everything here runs headless via Textual's ``run_test`` pilot. No test opens
a real terminal and none touches the network.
"""
import asyncio
import io

from rich.console import Console

from impara.extract import _make_signal
from impara.pipeline import discover
from impara.tui import ImparaApp, render_definition, render_profile, render_score

BODIES = [
    (
        "github",
        "I wish there was a way to export my recipes to PDF without retyping them.",
    ),
    (
        "github",
        "I wish there was a way to export my recipes to PDF without retyping them.",
    ),
    (
        "hackernews",
        "Why doesn't Dropbox support scheduled backups for my photo archive?",
    ),
    (
        "stackexchange",
        "I've tried several tools but none let me share a calendar with the team.",
    ),
]


def _opportunities():
    signals = []
    for i, (source, body) in enumerate(BODIES):
        s = _make_signal(
            source=source,
            title="item %d" % i,
            body=body,
            url="https://example.com/%d" % i,
            author="user%d" % i,
            engagement=i * 4,
            observed_at="2026-01-01",
        )
        assert s is not None, "complaint pattern missing: %r" % body
        signals.append(s)
    opps, _health = discover(signals=signals, min_signals=1)
    return opps


def text_of(widget) -> str:
    """Flatten whatever renderable a Textual Static currently holds."""
    content = widget.content
    if isinstance(content, str):
        return content
    return render_to_text(content)


def render_to_text(renderable) -> str:
    """Render to plain text without touching the real stdout (cp1252-safe)."""
    console = Console(
        width=400, record=True, force_terminal=False, file=io.StringIO()
    )
    console.print(renderable)
    return console.export_text()


def run(app: ImparaApp, body) -> None:
    async def _inner():
        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause()
            await body(pilot)

    asyncio.run(_inner())


# --------------------------------------------------------------------------- #
# Listing and filtering
# --------------------------------------------------------------------------- #
def test_tui_lists_uncorroborated_leads_when_asked():
    app = ImparaApp(opportunities=_opportunities(), from_store=False, show_all=True)
    seen = {}

    async def body(pilot):
        table = app.query_one("#table")
        seen["rows"] = table.row_count

    run(app, body)
    assert seen["rows"] == 3, "4 signals cluster into 3 problems"


def test_tui_hides_single_signal_leads_by_default():
    app = ImparaApp(opportunities=_opportunities(), from_store=False, show_all=False)
    seen = {}

    async def body(pilot):
        seen["rows"] = app.query_one("#table").row_count
        seen["title"] = text_of(app.query_one("#titlebar"))

    run(app, body)
    assert seen["rows"] == 1, "only the corroborated problem should be listed"
    assert "recipes" in seen["title"] or "export" in seen["title"]


def test_tui_toggle_all_reveals_hidden_leads():
    app = ImparaApp(opportunities=_opportunities(), from_store=False, show_all=False)
    seen = {}

    async def body(pilot):
        seen["before"] = app.query_one("#table").row_count
        await pilot.press("a")
        await pilot.pause()
        seen["after"] = app.query_one("#table").row_count

    run(app, body)
    assert seen["before"] == 1
    assert seen["after"] == 3


def test_tui_filter_narrows_the_list():
    app = ImparaApp(opportunities=_opportunities(), from_store=False, show_all=True)
    seen = {}

    async def body(pilot):
        seen["all"] = app.query_one("#table").row_count
        await pilot.press("f")
        for ch in "calendar":
            await pilot.press(ch)
        await pilot.pause()
        seen["filtered"] = app.query_one("#table").row_count

    run(app, body)
    assert seen["all"] == 3
    assert seen["filtered"] == 1


def test_tui_filter_that_matches_nothing_shows_empty_state():
    app = ImparaApp(opportunities=_opportunities(), from_store=False, show_all=True)
    seen = {}

    async def body(pilot):
        await pilot.press("f")
        for ch in "zzzzzz":
            await pilot.press(ch)
        await pilot.pause()
        seen["rows"] = app.query_one("#table").row_count
        seen["title"] = text_of(app.query_one("#titlebar"))

    run(app, body)
    assert seen["rows"] == 0
    assert "INSUFFICIENT EVIDENCE" in seen["title"]


def test_tui_with_no_results_starts_clean():
    app = ImparaApp(opportunities=[], from_store=False)
    seen = {}

    async def body(pilot):
        seen["rows"] = app.query_one("#table").row_count
        seen["title"] = text_of(app.query_one("#titlebar"))

    run(app, body)
    assert seen["rows"] == 0
    assert "INSUFFICIENT EVIDENCE" in seen["title"]


# --------------------------------------------------------------------------- #
# Detail panels
# --------------------------------------------------------------------------- #
def test_tui_renders_all_five_panels_for_the_selected_problem():
    app = ImparaApp(opportunities=_opportunities(), from_store=False, show_all=True)
    seen = {}

    async def body(pilot):
        await pilot.pause()
        seen["profile"] = text_of(app.query_one("#profile"))
        seen["evidence"] = text_of(app.query_one("#evidence"))
        await pilot.press("3")
        await pilot.pause()
        seen["score"] = text_of(app.query_one("#score"))
        await pilot.press("4")
        await pilot.pause()
        seen["validate"] = text_of(app.query_one("#validate"))
        await pilot.press("5")
        await pilot.pause()
        seen["define"] = text_of(app.query_one("#define"))

    run(app, body)

    assert "WHO" in seen["profile"]
    assert "SIGNALS" in seen["evidence"]
    assert "opportunity score" in seen["score"]
    assert "CONFIRMING EVIDENCE LOOKS LIKE" in seen["validate"]
    assert "PROBLEM STATEMENT" in seen["define"]


def test_tui_score_panel_keeps_observed_and_inferred_apart():
    app = ImparaApp(opportunities=_opportunities(), from_store=False, show_all=True)
    seen = {}

    async def body(pilot):
        await pilot.press("3")
        await pilot.pause()
        seen["score"] = text_of(app.query_one("#score"))

    run(app, body)
    assert "observed" in seen["score"]
    assert "inferred" in seen["score"]
    assert "CAVEATS" in seen["score"]


def test_tui_titlebar_shows_score_and_evidence_strength():
    app = ImparaApp(opportunities=_opportunities(), from_store=False, show_all=True)
    seen = {}

    async def body(pilot):
        await pilot.pause()
        seen["title"] = text_of(app.query_one("#titlebar"))

    run(app, body)
    assert "/100" in seen["title"]
    assert "signal(s)" in seen["title"]


# --------------------------------------------------------------------------- #
# Modals
# --------------------------------------------------------------------------- #
def test_tui_help_screen_opens_and_closes():
    app = ImparaApp(opportunities=_opportunities(), from_store=False, show_all=True)
    seen = {}

    async def body(pilot):
        await pilot.press("question_mark")
        await pilot.pause()
        seen["open"] = app.screen.__class__.__name__
        seen["body"] = text_of(app.screen.query_one("#dialog"))
        await pilot.press("escape")
        await pilot.pause()
        seen["closed"] = app.screen.__class__.__name__

    run(app, body)
    assert seen["open"] == "HelpScreen"
    assert "keyboard reference" in seen["body"]
    assert "Help" not in seen["closed"], "escape should return to the main screen"


def test_tui_sources_screen_reports_source_health():
    from impara.models import SourceHealth, SourceStatus

    health = [
        SourceHealth("github", "GitHub issues", SourceStatus.OK, "fine", 5),
        SourceHealth(
            "reddit", "Reddit", SourceStatus.UNAVAILABLE, "blocked", 0
        ),
    ]
    app = ImparaApp(
        opportunities=[], health=health, from_store=False, show_all=True
    )
    seen = {}

    async def body(pilot):
        await pilot.press("s")
        await pilot.pause()
        seen["body"] = text_of(app.screen.query_one("#dialog"))

    run(app, body)
    assert "GitHub issues" in seen["body"]
    assert "UNAVAILABLE" in seen["body"]
    assert "blocked" in seen["body"]


# --------------------------------------------------------------------------- #
# Renderers are pure and never raise
# --------------------------------------------------------------------------- #
def test_renderers_handle_a_single_signal_problem_without_crashing():
    from impara.define import build_definition

    opps = _opportunities()
    assert opps
    chunks = []
    for op in opps:
        chunks.append(render_to_text(render_profile(op)))
        chunks.append(render_to_text(render_score(op)))
        chunks.append(
            render_to_text(render_definition(build_definition(op.problem, op.profile)))
        )
    text = "\n".join(chunks)
    assert "PROBLEM STATEMENT" in text
    assert "opportunity score" in text


def test_tui_status_bar_reports_corpus_and_hidden_count():
    app = ImparaApp(opportunities=_opportunities(), from_store=False, show_all=False)
    seen = {}

    async def body(pilot):
        await pilot.pause()
        seen["status"] = text_of(app.query_one("#status"))

    run(app, body)
    assert "problem(s)" in seen["status"]
    assert "hidden" in seen["status"]


# --------------------------------------------------------------------------- #
# CLI wiring for `impara tui`
# --------------------------------------------------------------------------- #
def test_tui_subparser_accepts_the_all_flag():
    from impara.cli import build_parser

    assert build_parser().parse_args(["tui"]).all is False
    assert build_parser().parse_args(["tui", "--all"]).all is True


def test_cmd_tui_explains_how_to_install_textual_when_missing(monkeypatch, capsys):
    import sys

    import impara
    from impara.cli import build_parser, cmd_tui

    monkeypatch.setitem(sys.modules, "impara.tui", None)
    monkeypatch.delattr(impara, "tui", raising=False)

    code = cmd_tui(build_parser().parse_args(["tui"]))
    captured = capsys.readouterr()

    assert code == 1
    assert "impara[tui]" in captured.err
    assert "plain CLI keeps working" in captured.err


def test_discovery_progress_is_routed_to_the_tui_sink(monkeypatch):
    from impara import store
    from impara.cli import _run_discovery, build_parser
    from impara.models import SourceHealth, SourceStatus

    collected = []

    def fake_collect(**kwargs):
        kwargs["progress"]("scanning github...\n")
        kwargs["progress"]("scanning hackernews...\n")
        collected.append(kwargs)
        return [], [
            SourceHealth("github", "GitHub issues", SourceStatus.OK, "ok", 0)
        ]

    monkeypatch.setattr("impara.cli.collect", fake_collect)
    monkeypatch.setattr(store, "save_signals", lambda signals: None)
    monkeypatch.setattr(store, "save", lambda opportunities, health: None)

    args = build_parser().parse_args(["discover", "--fresh", "--json"])
    seen = []
    args._progress = seen.append
    opportunities = _run_discovery(args)

    assert opportunities == []
    assert collected and collected[0]["progress"] is not None
    assert seen[:2] == ["scanning github...\n", "scanning hackernews...\n"]
    assert seen[-1].startswith("clustering 0 verified signal(s)")
    assert args._health and args._health[0].key == "github"
