"""Text rendering for the CLI.

Output is plain text by design: it must be readable in a terminal, pasteable
into a document, and stable enough to diff between runs.
"""
from typing import TYPE_CHECKING, List, Optional, Sequence

from .models import (
    Opportunity,
    OpportunityScore,
    ProblemProfile,
    ProductDefinition,
    SourceHealth,
    ValidationPlan,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .understand import UnderstandingResult

RULE = "-" * 66
DOUBLE = "=" * 66


def wrap(text: str, width: int = 66) -> List[str]:
    text = (text or "").strip()
    if not text:
        return [""]
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def block(label: str, text: str, indent: int = 0) -> List[str]:
    out = [" " * indent + label]
    pad = " " * indent
    for line in wrap(text):
        out.append(pad + line)
    out.append("")
    return out


def bullet_block(label: str, items: Sequence[str], indent: int = 0) -> List[str]:
    out = [" " * indent + label]
    if not items:
        out.append(" " * (indent + 2) + "(none)")
    for i, item in enumerate(items, start=1):
        lines = wrap(item, width=64 - indent)
        out.append(" " * (indent + 2) + "%d. %s" % (i, lines[0]))
        for extra in lines[1:]:
            out.append(" " * (indent + 4) + extra)
    out.append("")
    return out


def render_health(health: Sequence[SourceHealth]) -> str:
    lines = [DOUBLE, "SOURCES", DOUBLE]
    for h in health:
        flag = "OK" if h.available else h.status.value.upper().replace("_", " ")
        lines.append("  [%s] %s" % (flag.ljust(11), h.label))
        if h.note:
            lines.extend("      " + l for l in wrap(h.note, 58))
        lines.append("      verified signals collected: %d" % h.signals_found)
        lines.append("")
    lines.append(RULE)
    return "\n".join(lines)


def render_score(score: OpportunityScore) -> str:
    lines = [
        "OPPORTUNITY SCORE",
        "%d / 100" % score.total,
        "",
        "Evidence strength: %s" % score.evidence_strength.value.upper(),
        "",
    ]
    width = max(len(d.label) for d in score.dimensions) if score.dimensions else 10
    for d in score.dimensions:
        shown = "unknown" if d.value is None else str(d.value)
        lines.append(
            "%s %s   [%s]" % (d.label.ljust(width), shown.rjust(7), d.kind.value)
        )
    lines.append("")
    for d in score.dimensions:
        lines.append("  %s: %s" % (d.label, d.reason))
    if score.caveats:
        lines.append("")
        lines.append("CAVEATS")
        for c in score.caveats:
            lines.extend("  - " + l for l in wrap(c, 62))
    return "\n".join(lines)


def render_opportunity_list(opportunities: Sequence[Opportunity], limit: int = 0,
                            filtered: int = 0, total_signals: int = 0,
                            pre_filter: int = 0) -> str:
    if not opportunities:
        lines = [DOUBLE, "INSUFFICIENT EVIDENCE", DOUBLE, ""]
        if total_signals:
            lines.append(
                (
                    "  %d verified signal(s) were collected, but none corroborated\n"
                    "  the same problem (2+ independent signals required by default).\n"
                    "\n"
                    "  Impara would rather report nothing than present a single\n"
                    "  unverified complaint as an opportunity."
                ) % total_signals
            )
            lines.append("")
            lines.append("  Try:  impara discover --all        (show single-signal leads)")
            lines.append("        impara discover --deep      (scan more of each thread)")
        elif pre_filter:
            lines.extend(
                (
                    "  %d signal(s) matched the fetch, but --market/--country\n"
                    "  filtered every one of them out. Nothing here aligns with\n"
                    "  that filter, and Impara would rather report nothing than\n"
                    "  stretch a mismatch into a problem."
                    % pre_filter
                ).splitlines()
            )
        else:
            lines.extend(
                "  No fetched text contained a complaint pattern, so nothing is\n"
                "  reported. Impara would rather return nothing than invent a problem."
                .splitlines()
            )
        lines.append("")
        lines.append(RULE)
        return "\n".join(lines)

    shown = opportunities[:limit] if limit else opportunities
    lines = [DOUBLE, "DISCOVERED OPPORTUNITIES  (%d)" % len(opportunities), DOUBLE, ""]
    for opp in shown:
        p = opp.problem
        lines.append("%s  %s" % (p.id.ljust(8), p.title[:56]))
        lines.append("  signals %d   sources %s   score %d/100   evidence %s"
                     % (len(p.signals), ",".join(p.sources), opp.score.total,
                        opp.score.evidence_strength.value))
        if opp.profile.insufficient_evidence:
            lines.append("  ** INSUFFICIENT EVIDENCE - investigate before building **")
        lines.append("")
    if limit and len(opportunities) > limit:
        lines.append("... %d more (use --limit or investigate by id)"
                     % (len(opportunities) - limit))
        lines.append("")
    if filtered:
        lines.append("%d weaker lead(s) hidden by default - use --all to include them."
                     % filtered)
        lines.append("")
    lines.append(RULE)
    lines.append("Next: impara investigate <id>     then: impara validate <id>")
    return "\n".join(lines)


def render_profile(profile: ProblemProfile) -> str:
    p = profile.problem
    lines = [DOUBLE, "PROBLEM %s" % p.id, DOUBLE, ""]
    for line in wrap(p.title):
        lines.append(line)
    lines.append("")
    lines.extend(block("WHO", profile.who))
    lines.extend(block("PROBLEM", profile.trying_to_accomplish))
    lines.extend(block("WHY DIFFICULT", profile.why_difficult))
    lines.extend(block("FREQUENCY", profile.frequency))
    lines.extend(block("CURRENT WORKAROUND", "; ".join(profile.workarounds)))
    lines.extend(block("WHY EXISTING SOLUTIONS ARE INSUFFICIENT",
                       profile.why_existing_insufficient))
    lines.extend(block("SEVERITY", profile.severity))
    lines.extend(block("IS SOMEONE PAYING", profile.paying_now))
    lines.extend(block("PRODUCT TYPE", profile.product_type))

    lines.append("EVIDENCE")
    for e in profile.evidence:
        tag = "observed" if e.kind.value == "observed" else "inferred"
        lines.append("  [%s] %s" % (tag, e.claim))
        if e.detail:
            lines.extend("        " + l for l in wrap(e.detail, 56))
    lines.append("")
    lines.append("EVIDENCE STRENGTH: %s" % profile.evidence_strength.value.upper())
    if profile.insufficient_evidence:
        lines.append("")
        lines.extend("  ** INSUFFICIENT EVIDENCE - treat as an unverified opportunity **"
                     .splitlines())
    lines.append("")
    lines.append(RULE)
    return "\n".join(lines)


def render_validation(plan: ValidationPlan) -> str:
    lines = [DOUBLE, "VALIDATION PLAN  %s" % plan.problem_id, DOUBLE, ""]
    lines.extend(bullet_block("PEOPLE TO INTERVIEW", plan.interviewees))
    lines.extend(bullet_block("QUESTIONS TO ASK", plan.questions))
    lines.extend(bullet_block("BEHAVIOURS TO INVESTIGATE", plan.behaviors))
    lines.extend(bullet_block("EXISTING SOLUTIONS TO COMPARE", plan.alternatives_to_compare))
    lines.extend(bullet_block("ASSUMPTIONS TO TEST", plan.assumptions_to_test))
    lines.extend(bullet_block("SIGNALS THAT CONFIRM THE PROBLEM", plan.confirming_signals))
    lines.extend(bullet_block("SIGNALS THAT INVALIDATE THE PROBLEM", plan.invalidating_signals))
    lines.append(RULE)
    return "\n".join(lines)


def render_definition(d: ProductDefinition) -> str:
    lines = [DOUBLE, "PRODUCT OPPORTUNITY  %s" % d.problem_id, DOUBLE, ""]
    lines.extend(block("PROBLEM STATEMENT", d.problem_statement))
    lines.extend(block("TARGET USER", d.target_user))
    lines.extend(block("JOB TO BE DONE", d.job_to_be_done))
    lines.extend(bullet_block("EXISTING ALTERNATIVES", d.existing_alternatives))
    lines.extend(block("PRODUCT OPPORTUNITY", d.product_opportunity))
    lines.extend(bullet_block("PROPOSED MVP", d.proposed_mvp))
    lines.extend(bullet_block("ESSENTIAL FEATURES", d.essential_features))
    lines.extend(bullet_block("DELIBERATELY EXCLUDED (anti feature-creep)", d.excluded_features))
    lines.extend(block("BUSINESS MODEL", d.business_model))
    lines.extend(block("TECHNICAL COMPLEXITY", d.technical_complexity))
    lines.extend(bullet_block("VALIDATION RISKS", d.validation_risks))
    lines.append(RULE)
    return "\n".join(lines)


def render_problem_pool(result: "UnderstandingResult", limit: int = 15) -> str:
    """Present the problem pool, or say plainly why there is none."""
    lines: List[str] = []
    kept = result.statements

    lines.append(DOUBLE)
    header = "PROBLEM STATEMENTS  %d kept" % len(kept)
    if result.dropped:
        header += " / %d dropped" % result.dropped
    lines.append(header)
    lines.append(DOUBLE)
    summary = "model: %s    signals: %d" % (result.model or "-", result.signals_seen)
    if result.screened:
        summary += "    screened out: %d" % result.screened
    lines.append(summary)

    if not result.available:
        lines.append("")
        lines.append("SEMANTIC UNDERSTANDING UNAVAILABLE")
        for line in (result.reason or "no model configured").splitlines():
            for wrapped in wrap(line, 64):
                lines.append("  " + wrapped)
        lines.append("")
        lines.append(
            "Nothing was guessed. The deterministic evidence pipeline still runs:"
        )
        lines.append("  impara discover   real problems from the same signals")
        lines.append("  impara corpus     the raw evidence behind them")
        lines.append(DOUBLE)
        return "\n".join(lines)

    if not kept:
        lines.append("")
        lines.append(
            "The model returned no statements that survived verification."
        )
        lines.append("The signals are still inspectable: impara discover")
        lines.append(DOUBLE)
        return "\n".join(lines)

    shown = kept[: max(1, limit)]
    for n, statement in enumerate(shown, 1):
        lines.append("")
        lines.append(RULE)
        lines.append("PROBLEM %03d" % n)
        lines.append(RULE)
        for line in wrap(statement.statement, 66):
            lines.append("  " + line)
        lines.append("")

        observed = [
            c for c in statement.claims if c.kind.value == "observed"
        ]
        lines.append(
            "  %-16s %s"
            % ("domain", statement.domain or "-")
        )
        if statement.who:
            lines.append("  %-16s %s" % ("who", ", ".join(statement.who)))
        lines.append(
            "  %-16s %s observed claim(s), %d quote(s)"
            % ("evidence", len(observed), statement.quote_count)
        )
        if statement.sources:
            lines.append("  %-16s %s" % ("sources", ", ".join(statement.sources)))
        if statement.evidence_signals:
            lines.append(
                "  %-16s %s" % ("signals", ", ".join(statement.evidence_signals))
            )

        if observed:
            lines.append("")
            lines.append("  EVIDENCE")
        for claim in observed:
            if claim.text != statement.statement:
                for line in wrap(claim.text, 60):
                    lines.append("    " + line)
            for quote in claim.quotes:
                for line in wrap('"%s"' % quote.text, 56):
                    lines.append("      " + line)
                lines.append(
                    "      <- %s (%s) %s"
                    % (quote.signal_id, quote.source or "?", quote.url or "<no url>")
                )
            lines.append("")

        if statement.unanswered:
            lines.append("  OPEN QUESTIONS")
            for question in statement.unanswered:
                for line in wrap(question, 62):
                    lines.append("    - " + line)
            lines.append("")

    if len(kept) > len(shown):
        lines.append(RULE)
        lines.append(
            "... %d more (use --limit to show more)" % (len(kept) - len(shown))
        )

    if result.dropped:
        lines.append(RULE)
        lines.append("DROPPED (rejected at the evidence boundary)")
        for reason in result.rejected[:10]:
            for line in wrap(reason, 62):
                lines.append("  - " + line)
        if result.dropped > 10:
            lines.append("  ... %d more" % (result.dropped - 10))

    lines.append(RULE)
    return "\n".join(lines)
