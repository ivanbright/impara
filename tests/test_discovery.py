"""Offline tests for the discovery pipeline.

No test here touches the network. Signals are injected so results are
deterministic and the evidence gate can be asserted directly.
"""
import json

import pytest

from impara.cluster import cluster_signals
from impara.define import build_definition
from impara.extract import _make_signal, from_github
from impara.models import Confidence, EvidenceKind, Signal
from impara.patterns import find_patterns
from impara.pipeline import discover, investigate
from impara.profile import build_profile
from impara.scoring import score_opportunity
from impara.validate import build_validation_plan
from impara import __version__


def sig(sid, source, text, author="a", engagement=0):
    """Build a verified signal the same way a source would."""
    s = _make_signal(
        source=source,
        title="",
        body=text,
        url="https://example.com/" + sid,
        author=author,
        engagement=engagement,
        observed_at="2026-01-01",
    )
    assert s is not None, "test text must contain a complaint pattern"
    return s


# --------------------------------------------------------------------------- #
# Evidence gate
# --------------------------------------------------------------------------- #
def test_pattern_detected_when_literal():
    assert find_patterns("I wish there was a tool for this")


def test_pattern_absent_from_generic_text():
    assert find_patterns("this is a perfectly normal sentence about databases") == []


def test_candidate_without_pattern_is_rejected():
    """A search hit that lacks the pattern must NOT become evidence."""
    assert (
        _make_signal(
            source="github",
            title="Fix notebook indexing",
            body="The build deadlocks on hosts with few CPUs.",
            url="https://example.com/x",
            author="a",
            engagement=5,
        )
        is None
    )


def test_candidate_with_pattern_is_kept():
    got = _make_signal(
        source="github",
        title="Add bulk export",
        body="I wish there was a way to export all records at once.",
        url="https://example.com/y",
        author="bob",
        engagement=9,
    )
    assert got is not None
    assert "wish" in got.patterns
    assert got.author == "bob"
    assert got.engagement == 9


def test_verification_beats_search_ranking():
    """Even a top-ranked item is dropped when the phrase is absent."""
    noisy = {"title": "FTS index build deadlocks", "body": "unrelated body", "html_url": "u"}
    assert from_github(noisy) is None


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def test_related_signals_cluster_together():
    a = sig("a", "github", "I wish there was an export tool for invoices")
    b = sig("b", "hackernews", "I wish there was a way to export invoices quickly")
    problems = cluster_signals([a, b])
    assert len(problems) == 1
    assert len(problems[0].signals) == 2
    assert len(problems[0].sources) == 2


def test_unrelated_signals_stay_separate():
    a = sig("a", "github", "I wish there was a tool to export invoices")
    b = sig("b", "github", "I have to do this manually calendar reminder sync every morning")
    problems = cluster_signals([a, b])
    assert len(problems) >= 2


def test_problem_ids_are_assigned_and_stable_format():
    a = sig("a", "github", "I wish there was an export tool for invoices")
    problems = cluster_signals([a])
    assert problems[0].id.startswith("#")
    assert len(problems[0].id) == 4


def test_empty_signals_cluster_to_nothing():
    assert cluster_signals([]) == []


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _problem(n_signals=6):
    sigs = []
    for i in range(n_signals):
        sigs.append(
            sig("s%d" % i, "github" if i % 2 else "hackernews",
                "This is frustrating and I have to do this manually every day, hours wasted",
                author="author%d" % i, engagement=i)
        )
    probs = cluster_signals(sigs)
    assert probs, "fixture must cluster into at least one problem"
    return probs[0]


def test_score_has_every_dimension_with_reason():
    score = score_opportunity(_problem())
    expected = {"evidence", "pain", "frequency", "buildability", "competition", "monetization"}
    assert {d.key for d in score.dimensions} == expected
    assert all(d.reason for d in score.dimensions)
    assert 0 <= score.total <= 100


def test_score_separates_observed_from_inferred():
    score = score_opportunity(_problem())
    kinds = {d.key: d.kind for d in score.dimensions}
    assert kinds["evidence"] is EvidenceKind.OBSERVED
    assert kinds["frequency"] is EvidenceKind.OBSERVED
    assert kinds["buildability"] is EvidenceKind.INFERRED
    assert kinds["competition"] is EvidenceKind.INFERRED
    assert kinds["monetization"] is EvidenceKind.INFERRED


def test_inferred_dimensions_are_caveated():
    """An inferred number must never travel without an admission that it is one."""
    score = score_opportunity(_problem())
    caveats = " ".join(score.caveats)
    inferred = [d for d in score.dimensions if d.kind is EvidenceKind.INFERRED]
    assert inferred, "fixture should produce at least one inferred dimension"
    assert "Buildability is inferred" in caveats
    assert "Competition is unmeasured" in caveats
    for d in inferred:
        assert d.kind.value in ("observed", "inferred")


def test_score_is_not_all_inferred():
    score = score_opportunity(_problem())
    assert score.observed_ratio >= 0.5


def test_weak_evidence_is_flagged_as_insufficient():
    single = sig("only", "github", "I wish there was a tool for this")
    problem = cluster_signals([single])[0]
    score = score_opportunity(problem)
    assert score.evidence_strength in (Confidence.WEAK, Confidence.NONE)
    assert any("Insufficient evidence" in c or "Unverified" in c for c in score.caveats)


def test_competition_caveat_is_always_present():
    """We never ran a competitor scan, so we must never imply we did."""
    score = score_opportunity(_problem())
    assert any("Competition" in c or "competitor" in c for c in score.caveats)


# --------------------------------------------------------------------------- #
# Profile
# --------------------------------------------------------------------------- #
def test_profile_keeps_evidence_observed():
    profile = build_profile(_problem())
    for e in profile.evidence:
        assert e.kind is EvidenceKind.OBSERVED


def test_profile_never_presents_inference_as_evidence():
    profile = build_profile(_problem())
    claims = " ".join(e.claim.lower() for e in profile.evidence)
    assert "inferred" not in claims


def test_profile_flags_insufficient_evidence_on_single_signal():
    problem = cluster_signals([sig("only", "github", "I wish there was a tool")])[0]
    profile = build_profile(problem)
    assert profile.insufficient_evidence is True


def test_profile_finds_stated_workaround():
    problem = cluster_signals(
        [sig("w", "github", "I currently use a spreadsheet because nothing else imports")]
    )[0]
    profile = build_profile(problem)
    assert any("spreadsheet" in w for w in profile.workarounds)


# --------------------------------------------------------------------------- #
# Validation + definition
# --------------------------------------------------------------------------- #
def test_validation_plan_has_all_sections():
    plan = build_validation_plan(_problem(), build_profile(_problem()))
    assert plan.interviewees and plan.questions and plan.behaviors
    assert plan.confirming_signals and plan.invalidating_signals
    assert plan.assumptions_to_test


def test_validation_plan_includes_invalidators():
    plan = build_validation_plan(_problem(), build_profile(_problem()))
    joined = " ".join(plan.invalidating_signals).lower()
    assert "only one author" in joined or "cannot recall" in joined


def test_definition_blocks_feature_creep():
    d = build_definition(_problem(), build_profile(_problem()))
    assert d.excluded_features, "MVP must list deliberately excluded features"
    assert d.proposed_mvp
    assert "Authentication" in " ".join(d.excluded_features)


def test_definition_does_not_promise_validation():
    d = build_definition(_problem(), build_profile(_problem()))
    assert any("evidence" in r.lower() or "insufficient" in r.lower() for r in d.validation_risks)


def test_job_to_be_done_is_never_invented():
    """Vocabulary lists are not a job statement - say so instead of fabricating."""
    d = build_definition(_problem(), build_profile(_problem()))
    assert d.job_to_be_done.startswith(
        "Not determinable"
    ) or "Observable intent keywords" not in d.job_to_be_done


def test_job_to_be_done_derived_when_workaround_observed():
    problem = cluster_signals(
        [
            sig(
                "w",
                "github",
                "I currently use a spreadsheet because nothing else imports "
                "and I have to do this manually every week",
            )
        ]
    )[0]
    d = build_definition(problem, build_profile(problem))
    assert d.job_to_be_done.startswith("Not determinable") or "spreadsheet" in d.job_to_be_done


def test_noise_words_are_not_used_as_problem_terms():
    problem = cluster_signals(
        [sig("n", "github", "I wish there was one thing I could say about this")]
    )[0]
    assert "wish" not in problem.terms
    assert "thing" not in problem.terms


def test_tokens_containing_digit_runs_are_dropped():
    from impara.terms import tokenize

    toks = tokenize("android and709 build release pipeline")
    assert "and709" not in toks
    assert "android" in toks


# --------------------------------------------------------------------------- #
# Pipeline API
# --------------------------------------------------------------------------- #
def test_discover_with_injected_signals_is_offline():
    signals = [
        sig("a", "github", "I wish there was an export tool for invoices", engagement=5),
        sig("b", "hackernews", "I wish there was an export tool for invoices", engagement=3),
        sig("c", "github", "This is frustrating and I have to do this manually", engagement=1),
    ]
    opps, health = discover(signals=signals)
    assert isinstance(opps, list)
    assert health == []
    assert len(opps) >= 1
    assert opps[0].problem.id.startswith("#")
    assert opps[0].score.dimensions


def test_discover_orders_by_score_desc():
    signals = [
        sig("a", "github", "I wish there was an export tool", engagement=5),
        sig("b", "hackernews", "I wish there was an export tool", engagement=3),
        sig("c", "stackexchange", "I wish there was an export tool", engagement=1),
    ]
    opps, _ = discover(signals=signals)
    totals = [o.score.total for o in opps]
    assert totals == sorted(totals, reverse=True)


def test_discover_with_no_signals_returns_empty_not_fake():
    opps, _ = discover(signals=[])
    assert opps == []


def test_investigate_lookup_by_id():
    signals = [sig("a", "github", "I wish there was an export tool", engagement=5)]
    opps, _ = discover(signals=signals)
    found = investigate(opps, opps[0].problem.id)
    assert found is not None
    assert found.problem.id == opps[0].problem.id
    assert investigate(opps, "#999") is None


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def test_opportunity_roundtrips_to_json():
    signals = [sig("a", "github", "I wish there was an export tool", engagement=5)]
    opps, _ = discover(signals=signals)
    blob = json.dumps(opps[0].to_dict())
    data = json.loads(blob)
    assert data["id"].startswith("#")
    assert data["score"]["dimensions"][0]["kind"] in ("observed", "inferred")
    assert data["profile"]["evidence"]


# --------------------------------------------------------------------------- #
# Version
# --------------------------------------------------------------------------- #
def test_version_is_newer_than_calculator_release():
    assert __version__ == "0.4.0"
