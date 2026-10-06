"""Text rendering for the CLI.

Output is plain text by design: it must be readable in a terminal, pasteable
into a document, and stable enough to diff between runs.
"""
from typing import List, Optional, Sequence

from .models import (
    Opportunity,
    OpportunityScore,
    ProblemProfile,
    ProductDefinition,
    SourceHealth,
    ValidationPlan,
)

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
        kind = "observed" if d.kind.value == "observed" else "inferred"
        lines.append("%s %s   [%s]" % (d.label.ljust(width), str(d.value).rjust(3), kind))
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
                            filtered: int = 0, total_signals: int = 0) -> str:
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
