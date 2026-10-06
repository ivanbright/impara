"""Validation planning.

A validation plan is deliberately *procedural*: it tells the developer what to
go and check. It never asserts that the problem has been validated.
"""
from typing import List

from .models import Problem, ProblemProfile, ValidationPlan


def build_validation_plan(problem: Problem, profile: ProblemProfile) -> ValidationPlan:
    src = ", ".join(problem.sources) or "the original sources"
    n = len(problem.signals)

    interviewees = [
        "The %d distinct author(s) behind these signals - open the source links and reply"
        % len(problem.authors),
        "Two people who work in the same domain but never posted a complaint "
        "(silent majority check)",
        "Someone who already pays for an existing tool in this space",
        "A person who tried and abandoned a related tool",
    ]

    questions = [
        "Walk me through the last time this happened. What did you do next?",
        "What did this cost you - time, money, or data?",
        "What are you using today, and why did you choose it?",
        "What have you already tried that did not work?",
        "What is the worst part of the current process?",
        "If this were solved, what would you do differently?",
        "Would you pay to remove this? Who signs off on that spend?",
    ]

    behaviors = [
        "Confirm the workaround is actually used, not just mentioned once.",
        "Measure how often the problem occurs per week or per project.",
        "Check whether the person has spent money on a partial fix.",
        "Look for repeated manual steps described in their own words.",
        "Note whether they built an internal script or spreadsheet for it.",
    ]

    alternatives = [
        "Existing tools named in the signals",
        "Spreadsheets and manual processes (the default competitor)",
        "An internal script or script-killed workflow",
        "Doing nothing - the most common alternative",
    ]

    assumptions = [
        "The problem recurs often enough to matter (observed %d time(s), source: %s)."
        % (n, src),
        "The person posting also has budget authority to pay.",
        "The existing tools cannot be configured to fix this today.",
        "The problem is not specific to one project or one team.",
        "This cluster represents a population, not a vocal minority.",
    ]

    confirming = [
        "Multiple independent authors describe the same manual step.",
        "People have already built a private workaround (spreadsheet, script).",
        "Someone has paid for a partial or adjacent solution.",
        "The complaint appears across more than one source (%s)." % src,
        "Interviewees say the frequency is weekly or worse.",
    ]

    invalidating = [
        "Interviewees cannot recall the last time it actually happened.",
        "The existing tool has a setting that already solves it.",
        "Only one author is behind all %d signals." % n,
        "Nobody has built a workaround - the pain is tolerable.",
        "The problem disappears once a single config change is made.",
        "Only the originators would pay; nobody else experiences it.",
    ]

    return ValidationPlan(
        problem_id=problem.id,
        interviewees=interviewees,
        questions=questions,
        behaviors=behaviors,
        alternatives_to_compare=alternatives,
        assumptions_to_test=assumptions,
        confirming_signals=confirming,
        invalidating_signals=invalidating,
    )
