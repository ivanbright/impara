"""Interactive TUI for impara.

Loaded lazily by ``impara tui`` so the plain CLI stays dependency-free for
users on Python 3.8 (Textual requires >=3.9).
"""
from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Static,
    TabbedContent,
    TabPane,
)

from . import __version__
from .define import build_definition
from .models import (
    Confidence,
    EvidenceKind,
    Opportunity,
    ProductDefinition,
    SourceHealth,
    ValidationPlan,
)
from .validate import build_validation_plan

PANELS = ["profile", "evidence", "score", "validate", "define"]
TAB_ORDER = ["tab_" + p for p in PANELS]

STRENGTH_STYLE = {
    Confidence.NONE: ("grey58", "NO DATA"),
    Confidence.WEAK: ("red", "WEAK"),
    Confidence.MODERATE: ("yellow", "MODERATE"),
    Confidence.STRONG: ("green", "STRONG"),
}

KEY_MIN = {
    "q": "quit",
    "d": "discover",
    "a": "toggle leads",
    "r": "reload",
    "s": "sources",
    "?": "help",
    "f": "filter",
    "1-5": "switch panel",
}

SPINNER = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"


def _strength_style(conf: Confidence):
    """Return ``(colour, label)`` for an evidence-strength value."""
    return STRENGTH_STYLE.get(conf, ("grey58", "NO DATA"))


def _kind_tag(kind: EvidenceKind) -> Text:
    if kind is EvidenceKind.OBSERVED:
        return Text("observed", style="bold green")
    return Text("inferred", style="bold italic yellow")


def _bar(value: int, width: int = 24, style: str = "green") -> Text:
    v = max(0, min(100, int(value)))
    filled = int(round(v / 100.0 * width))
    t = Text()
    t.append("\u2588" * filled, style=style)
    t.append("\u2591" * (width - filled), style="grey30")
    t.append(" %3d" % v, style="bold white")
    return t


def _wrap(text: str) -> Text:
    return Text(text or "-", style="dim" if not text else "")


def _section(title: str, body: Text) -> Group:
    return Group(
        Text(title, style="bold cyan"),
        body,
        Text(""),
    )


def _bullet(text: str, color: str = "white", marker: str = "\u2022") -> Text:
    t = Text()
    t.append("  " + marker + " ", style=color)
    t.append(text)
    return t


# --------------------------------------------------------------------------- #
# Panel renderers
# --------------------------------------------------------------------------- #
def render_titlebar(op: Opportunity) -> Group:
    style, label = _strength_style(op.profile.evidence_strength)
    head = Text()
    head.append(op.problem.id + "  ", style="bold white")
    head.append(op.problem.title, style="bold")
    head.append("\n")

    head.append("%3d/100 " % op.score.total, style="bold")
    head.append(_bar(op.score.total, 30, _bar_color(op.score.total)))
    head.append("   ")
    head.append("\u25cf " + label, style="bold " + style)
    head.append("  %d signal(s)" % len(op.problem.signals), style="dim")
    head.append("  \u00b7  ", style="grey30")
    head.append(", ".join(op.problem.sources) or "no source", style="cyan")
    return Group(head)


def _bar_color(value: int) -> str:
    if value >= 70:
        return "green"
    if value >= 45:
        return "yellow"
    return "red"


def render_profile(op: Opportunity) -> Group:
    p = op.profile
    parts: List[Any] = []
    parts.append(_section("WHO", _wrap(p.who)))
    parts.append(_section("TRYING TO ACCOMPLISH", _wrap(p.trying_to_accomplish)))
    parts.append(_section("WHY DIFFICULT", _wrap(p.why_difficult)))
    parts.append(_section("FREQUENCY", _wrap(p.frequency)))
    if p.workarounds:
        parts.append(
            _section(
                "CURRENT WORKAROUND",
                Group(*[_bullet(w) for w in p.workarounds]),
            )
        )
    else:
        parts.append(
            _section(
                "CURRENT WORKAROUND",
                Text("not stated in any observed signal", style="dim italic"),
            )
        )
    parts.append(
        _section("WHY EXISTING SOLUTIONS FAIL", _wrap(p.why_existing_insufficient))
    )
    parts.append(_section("SEVERITY", _wrap(p.severity)))
    parts.append(_section("IS SOMEONE PAYING", _wrap(p.paying_now)))
    parts.append(_section("PRODUCT TYPE", _wrap(p.product_type)))

    if p.insufficient_evidence:
        style, label = _strength_style(p.evidence_strength)
        parts.append(
            Panel(
                Text(
                    "INSUFFICIENT EVIDENCE \u2014 treat this as an unverified "
                    "opportunity, not a validated problem.",
                    style="bold " + style,
                    justify="center",
                ),
                border_style=style,
                title="evidence strength: " + label,
            )
        )
    return Group(*parts)


def render_evidence(op: Opportunity) -> Group:
    p = op.profile
    parts: List[Any] = []

    style, label = _strength_style(p.evidence_strength)
    head = Text()
    head.append("EVIDENCE STRENGTH  ")
    head.append("\u25cf " + label, style="bold " + style)
    head.append("   (%d record(s))" % len(p.evidence), style="dim")
    parts.append(Group(head, Text("")))

    for ev in p.evidence:
        row = Text("  ")
        row.append(_kind_tag(ev.kind))
        row.append("  ")
        row.append(ev.claim, style="bold")
        parts.append(row)
        if ev.detail:
            parts.append(Text("        " + ev.detail, style="dim"))
        if ev.url:
            parts.append(Text("        " + ev.url, style="underline cyan"))
        parts.append(Text(""))

    parts.append(Text("SIGNALS", style="bold cyan"))
    parts.append(
        Text(
            "  %d distinct author(s) \u00b7 %d engagement action(s)"
            % (len(op.problem.authors), op.problem.engagements),
            style="dim",
        )
    )
    parts.append(Text(""))
    for s in op.problem.signals:
        parts.append(
            Text("  [\u2022] ", style="cyan")
            + Text(s.source, style="bold cyan")
            + Text("  " + (s.author or "unknown"), style="dim")
        )
        parts.append(Text("      " + (s.title or s.excerpt[:90])))
        if s.patterns:
            parts.append(
                Text("      matched: ", style="dim")
                + Text(", ".join(s.patterns), style="bold yellow")
            )
        if s.url:
            parts.append(Text("      " + s.url, style="underline grey45"))
        parts.append(Text(""))

    return Group(*parts)


def render_score(op: Opportunity) -> Group:
    sc = op.score
    style, label = _strength_style(sc.evidence_strength)

    total = Text(justify="center")
    total.append(str(sc.total), style="bold " + _bar_color(sc.total))
    total.append(" / 100", style="dim")

    head = Group(
        Panel(total, border_style=_bar_color(sc.total), title="opportunity score"),
        Text(""),
        Text("  evidence strength: ", style="dim"),
        Text("\u25cf " + label, style="bold " + style),
        Text(
            "   (%d%% of dimensions are observed)" % int(sc.observed_ratio * 100),
            style="dim",
        ),
        Text(""),
    )

    dims: List[Any] = []
    for d in sc.dimensions:
        row = Text()
        row.append("  %-12s " % d.label, style="bold")
        row.append(_bar(d.value, 22, "green" if d.kind is EvidenceKind.OBSERVED else "blue"))
        row.append("  ")
        row.append(_kind_tag(d.kind))
        dims.append(row)
        dims.append(Text("    " + d.reason, style="grey62"))
        dims.append(Text(""))

    caveats: List[Any] = []
    if sc.caveats:
        caveats.append(Text("CAVEATS", style="bold yellow"))
        for c in sc.caveats:
            caveats.append(_bullet(c, "yellow", "\u26a0"))
    else:
        caveats.append(Text("CAVEATS", style="bold green"))
        caveats.append(_bullet("none recorded", "green", "\u2713"))

    return Group(head, Group(*dims), Text(""), Group(*caveats))


def render_validate(plan: ValidationPlan) -> Group:
    parts: List[Any] = []

    def block(title: str, items: Sequence[str], color: str, marker: str) -> None:
        parts.append(Text(title, style="bold cyan"))
        for it in items:
            parts.append(_bullet(it, color, marker))
        parts.append(Text(""))

    block("INTERVIEW THESE PEOPLE", plan.interviewees, "white", "\u2022")
    block("ASK", plan.questions, "white", "?")
    block("OBSERVE BEHAVIOUR", plan.behaviors, "white", "\u2022")
    block("COMPARE AGAINST", plan.alternatives_to_compare, "white", "\u2022")
    block("ASSUMPTIONS TO TEST", plan.assumptions_to_test, "yellow", "\u2022")
    block("CONFIRMING EVIDENCE LOOKS LIKE", plan.confirming_signals, "green", "\u2713")
    block("INVALIDATING EVIDENCE LOOKS LIKE", plan.invalidating_signals, "red", "\u2717")

    parts.append(
        Text(
            "This is a plan for gathering evidence. It does not mean the "
            "problem is validated.",
            style="dim italic",
        )
    )
    return Group(*parts)


def render_definition(d: ProductDefinition) -> Group:
    parts: List[Any] = []

    parts.append(_section("PROBLEM STATEMENT", _wrap(d.problem_statement)))
    parts.append(_section("TARGET USER", _wrap(d.target_user)))
    parts.append(_section("JOB TO BE DONE", _wrap(d.job_to_be_done)))
    parts.append(_section("PRODUCT OPPORTUNITY", _wrap(d.product_opportunity)))

    parts.append(Text("EXISTING ALTERNATIVES", style="bold cyan"))
    if d.existing_alternatives:
        for a in d.existing_alternatives:
            parts.append(_bullet(a))
    else:
        parts.append(Text("  none identified from observed signals", style="dim"))
    parts.append(Text(""))

    parts.append(Text("PROPOSED MVP", style="bold cyan"))
    for i, m in enumerate(d.proposed_mvp, 1):
        t = Text()
        t.append("  %d. " % i, style="bold green")
        t.append(m)
        parts.append(t)
    parts.append(Text(""))

    parts.append(Text("ESSENTIAL FEATURES", style="bold cyan"))
    for f in d.essential_features:
        parts.append(_bullet(f, "green", "+"))
    parts.append(Text(""))

    parts.append(Text("DELIBERATELY EXCLUDED", style="bold magenta"))
    if d.excluded_features:
        for f in d.excluded_features:
            parts.append(_bullet(f, "magenta", "\u2716"))
    else:
        parts.append(Text("  nothing excluded yet", style="dim"))
    parts.append(Text(""))

    parts.append(_section("BUSINESS MODEL", _wrap(d.business_model)))
    parts.append(_section("TECHNICAL COMPLEXITY", _wrap(d.technical_complexity)))

    parts.append(Text("VALIDATION RISKS", style="bold yellow"))
    for r in d.validation_risks:
        parts.append(_bullet(r, "yellow", "\u26a0"))
    return Group(*parts)


def render_empty() -> Panel:
    body = Text(justify="center")
    body.append("INSUFFICIENT EVIDENCE\n\n", style="bold red")
    body.append(
        "Impara only reports problems backed by literally-verified\n"
        "complaint patterns in real fetched text.\n\n",
        style="dim",
    )
    body.append("press ", style="dim")
    body.append("d", style="bold cyan")
    body.append(" to discover, ", style="dim")
    body.append("a", style="bold cyan")
    body.append(" to include single-signal leads", style="dim")
    return Panel(
        body,
        border_style="red",
        title="no corroborated problems",
        padding=(1, 2),
    )


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #
class StatusMsg(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class DiscoveryDone(Message):
    def __init__(
        self,
        opportunities: List[Opportunity],
        health: List[SourceHealth],
        error: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.opportunities = opportunities
        self.health = health
        self.error = error


# --------------------------------------------------------------------------- #
# Modal screens
# --------------------------------------------------------------------------- #
class HelpScreen(ModalScreen[None]):
    CSS = """
    HelpScreen { align: center middle; }
    #dialog {
        width: 74; height: auto; max-width: 94%;
        border: round $accent; background: $surface;
        padding: 1 2;
    }
    """

    BINDINGS = [("escape", "dismiss", "Close"), ("q", "dismiss", "Close")]

    def compose(self) -> ComposeResult:
        t = Text()
        t.append("impara  " + __version__, style="bold")
        t.append("  \u2014  keyboard reference\n\n", style="dim")
        rows = [
            ("d", "run discovery against live sources"),
            ("a", "show / hide single-signal leads"),
            ("r", "reload the saved run"),
            ("s", "source health"),
            ("f", "focus the filter box"),
            ("1-5", "Profile / Evidence / Score / Validate / Define"),
            ("ctrl+\u2190 \u2192", "previous / next panel"),
            ("arrows", "move between problems (table focused)"),
            ("enter", "open the highlighted problem"),
            ("q", "quit"),
        ]
        for k, desc in rows:
            t.append("  ")
            t.append(k.ljust(11), style="bold cyan")
            t.append(desc + "\n")
        t.append("\nEvidence rule: ", style="bold")
        t.append(
            "observed = literally read in fetched text.\n"
            "            inferred = computed by Impara, not a fact.",
            style="dim",
        )
        yield Static(t, id="dialog")

    def action_dismiss(self) -> None:
        self.dismiss()


class SourcesScreen(ModalScreen[None]):
    CSS = """
    SourcesScreen { align: center middle; }
    #dialog {
        width: 78; height: auto; max-width: 94%;
        border: round $accent; background: $surface;
        padding: 1 2;
    }
    """

    BINDINGS = [("escape", "dismiss", "Close"), ("q", "dismiss", "Close")]

    def __init__(self, health: List[SourceHealth]) -> None:
        super().__init__()
        self.health = health

    def compose(self) -> ComposeResult:
        t = Text()
        t.append("SIGNAL SOURCES\n\n", style="bold")
        if not self.health:
            t.append(
                "  No health report yet \u2014 run discovery first.\n",
                style="dim",
            )
        for h in self.health:
            ok = h.available
            t.append("  ")
            t.append("\u25cf ", style="green" if ok else "red")
            t.append(h.label, style="bold" if ok else "")
            t.append("\n      ")
            t.append("available" if ok else "UNAVAILABLE", style="green" if ok else "red")
            if h.note:
                t.append(" \u2014 " + h.note, style="dim")
            t.append("\n      ")
            t.append("%d verified signal(s)" % h.signals_found, style="cyan")
            t.append("\n\n")
        yield Static(t, id="dialog")

    def action_dismiss(self) -> None:
        self.dismiss()


# --------------------------------------------------------------------------- #
# The app
# --------------------------------------------------------------------------- #
APP_CSS = """
Screen {
    background: $background;
}

#body {
    height: 1fr;
}

#sidebar {
    width: 46%;
    min-width: 42;
    border-right: solid $accent;
    padding: 1;
}

#filter {
    height: auto;
    margin-bottom: 1;
}

#table {
    height: 1fr;
}

#detail {
    width: 1fr;
    padding: 0 1;
}

#titlebar {
    height: auto;
    padding: 1 0;
}

#tabs {
    height: 1fr;
}

#status {
    height: 1;
    background: $panel;
    color: $text-muted;
    padding: 0 1;
}
"""


class ImparaApp(App[None]):
    CSS = APP_CSS
    TITLE = "impara"
    SUB_TITLE = "problem discovery"
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("d", "discover", "Discover"),
        Binding("a", "toggle_all", "Leads"),
        Binding("r", "reload", "Reload"),
        Binding("s", "sources", "Sources"),
        Binding("question_mark", "help", "Help"),
        Binding("f", "focus_filter", "Filter", show=False),
        Binding("1", "tab('tab_profile')", show=False),
        Binding("2", "tab('tab_evidence')", show=False),
        Binding("3", "tab('tab_score')", show=False),
        Binding("4", "tab('tab_validate')", show=False),
        Binding("5", "tab('tab_define')", show=False),
        Binding("ctrl+right", "next_tab", show=False),
        Binding("ctrl+left", "prev_tab", show=False),
    ]

    def __init__(
        self,
        opportunities: Optional[List[Opportunity]] = None,
        health: Optional[List[SourceHealth]] = None,
        from_store: bool = True,
        show_all: bool = False,
    ) -> None:
        super().__init__()
        self._injected = opportunities
        self._injected_health = health
        self._from_store = from_store
        self.show_all = show_all
        self._all: List[Opportunity] = []
        self._shown: List[Opportunity] = []
        self._health: List[SourceHealth] = list(health or [])
        self._index: Dict[str, Opportunity] = {}
        self._current: Optional[Opportunity] = None
        self._busy = False
        self._frame = 0
        self._status = ""
        self._hidden = 0
        self._spin = None

    # -- lifecycle ---------------------------------------------------------- #
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Input(placeholder="filter problems...", id="filter")
                yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
            with Vertical(id="detail"):
                yield Static(id="titlebar")
                with TabbedContent(id="tabs"):
                    for key in PANELS:
                        yield TabPane(
                            key.capitalize(),
                            VerticalScroll(Static(id=key)),
                            id="tab_" + key,
                        )
        yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#table", DataTable)
        table.add_columns("ID", "Score", "Sig", "Evidence", "Sources", "Title")
        self.query_one("#titlebar", Static).update(render_empty())
        for key in PANELS:
            self.query_one("#" + key, Static).update(Text(""))

        if self._injected is not None:
            self._set_results(
                self._injected,
                list(self._injected_health or []),
                None,
                "sample run \u2014 press d for a real scan",
            )
        elif self._from_store:
            self._load_saved()
        else:
            self._paint_status()

        self._spin = self.set_interval(0.1, self._tick, name="spin", pause=True)
        self.query_one("#table", DataTable).focus()

    # -- data --------------------------------------------------------------- #
    def _load_saved(self) -> None:
        from . import store

        loaded = store.load()
        if not loaded:
            self._status = "no saved run \u2014 press d to discover"
            self._paint_status()
            return
        opportunities, health = loaded
        self._set_results(
            list(opportunities), list(health), None, "loaded last saved run"
        )
        self._paint_status()

    def _set_results(
        self,
        opportunities: List[Opportunity],
        health: List[SourceHealth],
        error: Optional[str],
        status: Optional[str] = None,
    ) -> None:
        self._all = opportunities
        if health:
            self._health = health
        if error:
            self._status = error
        elif status is not None:
            self._status = status
        self._populate()

    def _apply_filter(self, needle: str = "") -> List[Opportunity]:
        needle = (needle or "").lower().strip()
        out = []
        for o in self._all:
            if not self.show_all and len(o.problem.signals) < 2:
                continue
            if needle:
                hay = " ".join(
                    [o.problem.id, o.problem.title] + list(o.problem.terms)
                ).lower()
                if needle not in hay:
                    continue
            out.append(o)
        return out

    def _populate(self) -> None:
        try:
            query = self.query_one("#filter", Input).value
        except Exception:
            query = ""
        self._shown = self._apply_filter(query)
        self._hidden = len(self._all) - len(self._shown)
        self._index = {o.problem.id: o for o in self._shown}

        table = self.query_one("#table", DataTable)
        table.clear()
        for o in self._shown:
            style, label = _strength_style(o.profile.evidence_strength)
            table.add_row(
                o.problem.id,
                Text("%d" % o.score.total, style=_bar_color(o.score.total) + " bold"),
                str(len(o.problem.signals)),
                Text(label, style="bold " + style),
                ", ".join(o.problem.sources),
                o.problem.title,
                key=o.problem.id,
            )
        self._paint_status()

        if self._shown:
            table.move_cursor(row=0)
            self._show(self._shown[0])
        else:
            self._current = None
            self.query_one("#titlebar", Static).update(render_empty())
            for key in PANELS:
                self.query_one("#" + key, Static).update(Text(""))

    def _show(self, op: Opportunity) -> None:
        self._current = op
        self.query_one("#titlebar", Static).update(render_titlebar(op))
        self.query_one("#profile", Static).update(render_profile(op))
        self.query_one("#evidence", Static).update(render_evidence(op))
        self.query_one("#score", Static).update(render_score(op))
        try:
            self.query_one("#validate", Static).update(
                render_validate(build_validation_plan(op.problem, op.profile))
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.query_one("#validate", Static).update(
                Text("validation plan unavailable: %s" % exc, style="red")
            )
        try:
            self.query_one("#define", Static).update(
                render_definition(build_definition(op.problem, op.profile))
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.query_one("#define", Static).update(
                Text("definition unavailable: %s" % exc, style="red")
            )

    # -- status bar --------------------------------------------------------- #
    def _paint_status(self) -> None:
        try:
            bar = self.query_one("#status", Static)
        except Exception:
            return
        t = Text()
        if self._busy:
            t.append(SPINNER[self._frame % len(SPINNER)] + " ", style="bold cyan")
            t.append("scanning  ", style="bold cyan")
        for h in self._health:
            ok = h.available
            t.append("\u25cf ", style="green" if ok else "red")
            t.append(h.key, style="" if ok else "dim")
            t.append("  ")
        if self._health:
            t.append(" \u00b7 ", style="grey30")

        try:
            from . import store

            t.append("corpus %d" % store.corpus_size(), style="cyan")
        except Exception:
            t.append("corpus ?", style="cyan")

        t.append("  \u00b7  %d problem(s)" % len(self._shown), style="white")
        if self._hidden:
            t.append(
                "  (%d hidden \u2014 press a)" % self._hidden, style="yellow"
            )
        if self._status:
            t.append("  \u00b7  ", style="grey30")
            t.append(self._status, style="dim")
        bar.update(t)

    def _tick(self) -> None:
        self._frame += 1
        if self._busy:
            self._paint_status()

    # -- actions ------------------------------------------------------------ #
    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_sources(self) -> None:
        self.push_screen(SourcesScreen(self._health))

    def action_focus_filter(self) -> None:
        self.query_one("#filter", Input).focus()

    def action_tab(self, name: str) -> None:
        self.query_one("#tabs", TabbedContent).active = name

    def action_next_tab(self) -> None:
        tabs = self.query_one("#tabs", TabbedContent)
        i = TAB_ORDER.index(tabs.active) if tabs.active in TAB_ORDER else 0
        self.action_tab(TAB_ORDER[(i + 1) % len(TAB_ORDER)])

    def action_prev_tab(self) -> None:
        tabs = self.query_one("#tabs", TabbedContent)
        i = TAB_ORDER.index(tabs.active) if tabs.active in TAB_ORDER else 0
        self.action_tab(TAB_ORDER[(i - 1) % len(TAB_ORDER)])

    def action_reload(self) -> None:
        if self._injected is not None:
            self._set_results(self._injected, list(self._injected_health or []), None)
            return
        self._load_saved()

    def action_toggle_all(self) -> None:
        self.show_all = not self.show_all
        self._status = (
            "showing single-signal leads" if self.show_all else "hiding weak leads"
        )
        self._populate()

    def action_discover(self) -> None:
        if self._busy:
            return
        self.run_discovery()

    # -- discovery worker --------------------------------------------------- #
    @work(thread=True, exclusive=True, group="discovery")
    def run_discovery(self) -> None:
        from .cli import _run_discovery
        from . import store

        args = SimpleNamespace(
            source=None,
            market="",
            country="",
            per_source=30,
            deep=False,
            fresh=False,
            threshold=0.34,
            all_leads=True,
            min_signals=1,
            min_evidence="any",
            json=True,
            limit=15,
            show_sources=False,
        )

        def progress(msg: str) -> None:
            self.post_message(StatusMsg(_clean(msg)))

        args._progress = progress
        self.post_message(StatusMsg("starting discovery..."))
        try:
            opportunities = _run_discovery(args)
            health = getattr(args, "_health", [])
            self.post_message(
                DiscoveryDone(list(opportunities), list(health), None)
            )
            del store
        except Exception as exc:
            self.post_message(
                DiscoveryDone([], [], "%s: %s" % (type(exc).__name__, exc))
            )

    # -- message handlers --------------------------------------------------- #
    def on_status_msg(self, ev: StatusMsg) -> None:
        self._status = ev.text
        if not self._busy:
            self._busy = True
            self._frame = 0
            if self._spin:
                self._spin.resume()
        self._paint_status()

    def on_discovery_done(self, ev: DiscoveryDone) -> None:
        self._busy = False
        if self._spin:
            self._spin.pause()

        health = ev.health or []
        self._set_results(ev.opportunities, health, ev.error, "scanned just now")

        # Persist what the user is actually looking at so `impara list`,
        # `investigate` and the TUI all agree.
        if not ev.error:
            try:
                from . import store

                store.save(self._shown, health)
            except Exception:
                pass
        self._paint_status()
        self.query_one("#table", DataTable).focus()

    @on(DataTable.RowHighlighted, "#table")
    def _row_highlighted(self, ev: DataTable.RowHighlighted) -> None:
        if ev.row_key is None:
            return
        op = self._index.get(str(ev.row_key.value))
        if op is not None:
            self._show(op)

    @on(Input.Submitted, "#filter")
    def _filter_submitted(self, ev: Input.Submitted) -> None:
        self.query_one("#table", DataTable).focus()

    @on(Input.Changed, "#filter")
    def _filter_changed(self, ev: Input.Changed) -> None:
        self._populate()


def _clean(msg: str) -> str:
    return re.sub(r"\s+", " ", msg).strip()


def run(
    opportunities: Optional[List[Opportunity]] = None,
    health: Optional[List[SourceHealth]] = None,
    show_all: bool = False,
) -> int:
    """Start the TUI. Returns the process exit code."""
    app = ImparaApp(
        opportunities=opportunities, health=health, show_all=show_all
    )
    app.run()
    return 0
