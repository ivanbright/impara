"""Turn a problem into a product opportunity and a deliberately narrow MVP."""
from typing import List

from .models import ProductDefinition, Problem, ProblemProfile


def _job_to_be_done(profile: ProblemProfile, label: str) -> str:
    """Only state a job when one was actually observed.

    Vocabulary lists are not a job-to-be-done, and inventing one would turn an
    inference into a product requirement.
    """
    goal = (profile.trying_to_accomplish or "").strip()
    if goal.startswith("Complete work currently done with:"):
        detail = goal.split(":", 1)[1].strip()
        return (
            "When I am doing %s, I want to get it done without doing it by hand "
            "with %s, so that I stop paying the cost described in the signals."
            % (label, detail)
        )
    if goal.startswith("Observable intent keywords") or goal == "not stated in observed text":
        return (
            "Not determinable from observed signals. No job statement appeared in "
            "the fetched text - interview the authors before defining a product."
        )
    if goal:
        return goal
    return (
        "Not determinable from observed signals. No job statement appeared in the "
        "fetched text - interview the authors before defining a product."
    )


def build_definition(problem: Problem, profile: ProblemProfile) -> ProductDefinition:
    label = ", ".join(problem.terms[:3]) or "this workflow"
    subject = (problem.title or label).strip()
    n = len(problem.signals)

    problem_statement = (
        "%d independent signal(s) across %s describe difficulty with: %s. "
        "Current process: %s."
        % (
            n,
            ", ".join(problem.sources) or "unknown sources",
            subject,
            profile.why_difficult,
        )
    )

    target_user = (
        "People who posted in %s about %s (%d distinct author(s) observed). "
        "Broader population is unverified."
        % (", ".join(problem.sources) or "the source", subject, len(problem.authors))
    )

    jtbd = _job_to_be_done(profile, label)

    alternatives = profile.workarounds or [
        "spreadsheets",
        "manual process",
        "doing nothing",
    ]

    opportunity = (
        "A narrow tool covering %s, aimed at the workflow the signals actually "
        "describe rather than the whole domain." % subject
    )

    mvp: List[str] = [
        "Start: capture the input the signal says is handled manually today",
        "Core step: %s" % (profile.why_difficult.split(";")[0] if profile.why_difficult else "the primary task"),
        "Connect: replace one stated workaround (%s)" % alternatives[0],
        "Finish: produce the outcome the signals ask for",
        "Measure: record whether the manual step disappeared",
    ]

    essential = [
        "The single workflow the signals describe",
        "Import or entry of the data currently handled manually",
        "The one output people say they cannot get today",
        "Observable success: the manual step is gone",
    ]

    excluded = [
        "Authentication and accounts (not needed to test the problem)",
        "Teams, roles and permissions",
        "Dashboards beyond the single required output",
        "Integrations beyond the one stated workaround",
        "Mobile apps",
        "Billing and subscriptions",
        "Notifications, feeds, or social features",
        "Customisation and theming",
    ]

    business_model = (
        "Start free. Do not design pricing until willingness-to-pay is observed; "
        "current evidence on payment behaviour: %s." % profile.paying_now
    )

    complexity_terms = set(problem.terms)
    if complexity_terms & {"automate", "automation", "integration", "integrate", "sync", "synchronize"}:
        complexity = "Medium - requires connecting to an existing system"
    elif complexity_terms & {"data", "database", "export", "import", "report", "reports"}:
        complexity = "Medium - data modelling and import"
    else:
        complexity = "Low - single-user, single-workflow tool"

    risks: List[str] = [
        "Evidence strength is %s; %d signal(s) may not represent a population."
        % (profile.evidence_strength.value, n),
        "Cluster may contain unrelated complaints joined by shared vocabulary.",
        "No competitor scan has been run - the space may already be served.",
        "No willingness-to-pay evidence yet." if "no payment" in profile.paying_now
        else "Payment evidence is language only; validate with a real price test.",
        "Problem may be tolerable enough that nobody switches tools.",
    ]
    if profile.insufficient_evidence:
        risks.insert(0, "INSUFFICIENT EVIDENCE - validate before building anything.")

    return ProductDefinition(
        problem_id=problem.id,
        problem_statement=problem_statement,
        target_user=target_user,
        job_to_be_done=jtbd,
        existing_alternatives=alternatives,
        product_opportunity=opportunity,
        proposed_mvp=mvp,
        essential_features=essential,
        excluded_features=excluded,
        business_model=business_model,
        technical_complexity=complexity,
        validation_risks=risks,
    )
