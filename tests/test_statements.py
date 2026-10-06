"""Tests for the problem-statement slice: llm -> statement -> understand -> CLI.

Everything here is offline. The model is an injected fake transport, so the
suite proves the pipeline's behaviour rather than its luck.
"""
import io
import json

import pytest

from impara.cli import build_parser, main
from impara.llm import (
    LLMConfig,
    LLMError,
    available,
    chat,
    config_from_env,
    extract_json,
    unavailable_reason,
)
from impara.models import EvidenceKind, Signal
from impara.statement import (
    Claim,
    ProblemStatement,
    Quote,
    attach_provenance,
    normalize,
    verify,
)
from impara.understand import (
    build_prompt,
    parse_statements,
    screen_reason,
    screen_signals,
    understand,
    understand_batch,
)

LONG_TEXT = (
    "I have been maintaining the same billing spreadsheet for three years "
    "because nothing imports my invoicing history, and every month I "
    "reconcile it by hand."
)


def sig(sid, text, source="hackernews", author="alice", url=None, title=""):
    return Signal(
        id=sid,
        source=source,
        title=title,
        excerpt=text,
        url=url if url is not None else "https://example.com/%s" % sid,
        patterns=["spreadsheet"],
        terms=["billing", "spreadsheet"],
        author=author,
        engagement=10,
        observed_at="2026-01-01",
        thread="t-%s" % sid,
    )


def config():
    return LLMConfig(base_url="http://localhost:9/v1", model="test-model", api_key="k")


def fake_transport(payload):
    def send(cfg, body):
        assert body["model"] == "test-model"
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    return send


# --------------------------------------------------------------------------- #
# llm
# --------------------------------------------------------------------------- #
def test_no_model_configured_means_no_config():
    assert config_from_env({}) is None
    assert available({}) is False
    reason = unavailable_reason({})
    assert "IMPARA_LLM_BASE_URL" in reason and "IMPARA_LLM_MODEL" in reason


def test_config_reads_base_model_key_and_timeout():
    got = config_from_env(
        {
            "IMPARA_LLM_BASE_URL": "https://api.example.com/v1/",
            "IMPARA_LLM_MODEL": "gpt-4o-mini",
            "IMPARA_LLM_API_KEY": "sk-test",
            "IMPARA_LLM_TIMEOUT": "5",
        }
    )
    assert got is not None
    assert got.endpoint == "https://api.example.com/v1/chat/completions"
    assert got.model == "gpt-4o-mini"
    assert got.api_key == "sk-test"
    assert got.timeout == 5


def test_endpoint_never_doubles_the_suffix():
    cfg = LLMConfig(base_url="http://x/v1/chat/completions", model="m")
    assert cfg.endpoint == "http://x/v1/chat/completions"


def test_bad_timeout_falls_back_to_the_default():
    got = config_from_env(
        {"IMPARA_LLM_BASE_URL": "http://x/v1", "IMPARA_LLM_MODEL": "m",
         "IMPARA_LLM_TIMEOUT": "soon"}
    )
    assert got is not None and got.timeout == 60


def test_chat_reports_a_malformed_response_instead_of_crashing():
    with pytest.raises(LLMError):
        chat(config(), [{"role": "user", "content": "hi"}], transport=lambda c, b: {})


def test_chat_sends_the_exact_configured_payload_to_the_transport():
    sent = {}

    def capture(cfg, body):
        sent.update(body)
        return {"choices": [{"message": {"content": "ok"}}]}

    cfg = LLMConfig(
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-120b",
        api_key="sk-test-3",
    )
    chat(
        cfg,
        [{"role": "user", "content": "Reply with exactly: hello"}],
        transport=capture,
    )
    assert sent == {
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
        "temperature": 0.0,
    }


def test_outgoing_request_has_an_explicit_identifiable_user_agent(monkeypatch):
    import urllib.request

    from impara import __version__

    class FakeResponse:
        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    seen = []

    def fake_urlopen(request, timeout=None):
        seen.append(request)
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    cfg = LLMConfig(
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-120b",
        api_key="sk-test-3",
    )
    chat(cfg, [{"role": "user", "content": "Reply with exactly: hello"}])

    assert len(seen) == 1
    req = seen[0]
    assert req.method == "POST"
    assert req.full_url == "https://api.groq.com/openai/v1/chat/completions"
    assert req.get_header("Content-type") == "application/json"
    assert req.get_header("Authorization") == "Bearer sk-test-3"
    user_agent = req.get_header("User-agent")
    assert user_agent == "impara/%s" % __version__
    assert "urllib" not in user_agent.lower()
    body = json.loads(req.data)
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["messages"] == [{"role": "user", "content": "Reply with exactly: hello"}]
    assert body["temperature"] == 0.0
    assert set(body) == {"model", "messages", "temperature"}
    assert "sk-test-3" not in req.full_url
    assert "sk-test-3" not in req.data.decode("utf-8")


def test_rate_limited_requests_are_retried_with_backoff(monkeypatch):
    import urllib.error
    import urllib.request

    from impara import llm as llm_module

    calls = []
    sleeps = []

    class FakeResponse:
        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        if len(calls) < 3:
            raise urllib.error.HTTPError(
                request.full_url, 429, "Too Many Requests", {"Retry-After": "0"}, None
            )
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    cfg = LLMConfig(base_url="https://api.groq.com/openai/v1", model="m", api_key="k")
    assert chat(cfg, [{"role": "user", "content": "hi"}]) == "ok"
    assert len(calls) == 3
    assert sleeps == [0.0, 0.0]


def test_giving_up_on_a_rate_limit_still_reports_it(monkeypatch):
    import urllib.error

    from impara import llm as llm_module

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 429, "Too Many Requests", {"Retry-After": "300"}, None
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: None)

    cfg = LLMConfig(base_url="https://api.groq.com/openai/v1", model="m", api_key="k")
    with pytest.raises(LLMError) as exc_info:
        chat(cfg, [{"role": "user", "content": "hi"}])
    assert "HTTP 429" in str(exc_info.value)


def test_extract_json_tolerates_code_fences():
    got = extract_json('```json\n{"problems": []}\n```')
    assert got == {"problems": []}


def test_extract_json_without_json_is_an_error():
    with pytest.raises(LLMError):
        extract_json("I could not do that.")


# --------------------------------------------------------------------------- #
# The provenance invariant: claim -> quote -> signal id -> url
# --------------------------------------------------------------------------- #
def _statement(signal, quote_text, claim_text="Nothing imports my history."):
    return ProblemStatement(
        statement="Freelancers re-enter invoicing data by hand every month.",
        domain="Finance",
        who=["freelancer"],
        claims=[
            Claim(
                text=claim_text,
                quotes=[Quote(text=quote_text, signal_id=signal.id)],
                kind=EvidenceKind.OBSERVED,
            )
        ],
        signal_ids=[signal.id],
        understanding="llm",
    )


def test_verbatim_quote_verifies_and_carries_provenance():
    s = sig("sig-001", LONG_TEXT)
    statement = _statement(s, LONG_TEXT[:60])
    attach_provenance(statement, [s])

    report = verify(statement, [s])
    assert report.ok is True
    assert report.checked == 1

    quote = statement.claims[0].quotes[0]
    assert quote.url == s.url
    assert quote.source == s.source
    assert quote.author == s.author
    assert statement.urls == [s.url]


def test_fabricated_quote_fails_verification():
    s = sig("sig-001", LONG_TEXT)
    statement = _statement(s, "I cannot believe how expensive this software is.")
    report = verify(statement, [s])
    assert report.ok is False
    assert any("not verbatim" in e for e in report.errors)


def test_modified_quote_fails_verification():
    s = sig("sig-001", LONG_TEXT)
    edited = LONG_TEXT[:60] + " and more"
    statement = _statement(s, edited)
    report = verify(statement, [s])
    assert report.ok is False


def test_line_wrapped_quote_still_verifies():
    s = sig("sig-001", LONG_TEXT)
    idx = LONG_TEXT.index(" ", 25)
    first = LONG_TEXT[:idx]
    wrapped = "  \n   ".join([first, LONG_TEXT[idx + 1:]])
    assert normalize(wrapped) in normalize(LONG_TEXT)
    assert verify(_statement(s, wrapped), [s]).ok is True


def test_quote_citing_a_missing_signal_fails():
    s = sig("sig-001", LONG_TEXT)
    statement = _statement(s, LONG_TEXT[:60])
    statement.claims[0].quotes[0].signal_id = "sig-404"
    report = verify(statement, [s])
    assert report.ok is False
    assert any("not in the corpus" in e for e in report.errors)


def test_observed_claim_without_any_quote_fails():
    s = sig("sig-001", LONG_TEXT)
    statement = ProblemStatement(
        statement="Freelancers re-enter data.",
        claims=[Claim(text="Nobody likes typing.", quotes=[])],
        signal_ids=[s.id],
    )
    report = verify(statement, [s])
    assert report.ok is False
    assert any("no quote" in e for e in report.errors)


def test_statement_with_no_observed_evidence_fails():
    s = sig("sig-001", LONG_TEXT)
    statement = ProblemStatement(
        statement="Maybe freelancers are annoyed by spreadsheets.",
        claims=[Claim(text="Speculation.", quotes=[], kind=EvidenceKind.INFERRED)],
        signal_ids=[],
    )
    report = verify(statement, [s])
    assert report.ok is False
    assert any("no observed evidence" in e for e in report.errors)


def test_url_tampering_is_detected():
    s = sig("sig-001", LONG_TEXT)
    statement = _statement(s, LONG_TEXT[:60])
    attach_provenance(statement, [s])
    statement.claims[0].quotes[0].url = "https://attacker.example/evidence"
    report = verify(statement, [s])
    assert report.ok is False
    assert any("wrong URL" in e for e in report.errors)


def test_inferred_claim_alone_is_not_enough_evidence():
    statement = ProblemStatement(
        statement="Freelancers re-enter data.",
        claims=[Claim(text="Speculation.", quotes=[], kind=EvidenceKind.INFERRED)],
    )
    report = verify(statement, [])
    assert report.ok is False
    assert any("no observed evidence" in e for e in report.errors)


# --------------------------------------------------------------------------- #
# Content filter (ingest -> filter)
# --------------------------------------------------------------------------- #
def test_commentary_is_screened_out():
    removal = sig("sig-r", "This comment was removed by a moderator after a report.")
    reaction = sig("sig-j", "lol")
    assert screen_reason(removal)
    assert screen_reason(reaction)


def test_unrelated_bare_link_is_screened_out():
    link = sig("sig-l", "https://example.com/blog/post/12345?utm_source=newsletter")
    assert screen_reason(link) == "bare link with no description"


def test_a_real_complaint_is_not_screened_out():
    real = sig("sig-1", LONG_TEXT)
    assert screen_reason(real) == ""
    kept, skipped = screen_signals([real])
    assert [s.id for s in kept] == ["sig-1"]
    assert skipped == []


def test_screened_signals_never_reach_the_prompt():
    usable = sig("sig-1", LONG_TEXT)
    noise = sig("sig-n", "https://example.com/some/long/link/with/a/deep/path")
    kept, skipped = screen_signals([usable, noise])
    assert [s.id for s in kept] == ["sig-1"]
    prompt = build_prompt(kept)
    assert "sig-1" in prompt[1]["content"]
    assert "sig-n" not in prompt[1]["content"]


# --------------------------------------------------------------------------- #
# Understand (with an injected model)
# --------------------------------------------------------------------------- #
def valid_payload(signal):
    return {
        "problems": [
            {
                "signal_id": signal.id,
                "has_problem": True,
                "problem": "Freelancers re-enter invoicing data by hand every month.",
                "who": ["freelancer"],
                "domain": "billing automation",
                "evidence": signal.excerpt[:60],
                "unanswered": ["What triggers the manual step?"],
            }
        ]
    }


def entry_payload(signal, statement, evidence=None, **extra):
    entry = {
        "signal_id": signal.id,
        "has_problem": True,
        "problem": statement,
        "who": ["freelancer"],
        "domain": "billing automation",
        "evidence": evidence if evidence is not None else signal.excerpt[:60],
    }
    entry.update(extra)
    return {"problems": [entry]}


def test_without_a_model_understanding_is_marked_unavailable():
    result = understand([sig("sig-1", LONG_TEXT)], config=None)
    assert result.available is False
    assert result.statements == []
    assert "IMPARA_LLM_BASE_URL" in result.reason
    assert result.to_dict()["understanding"] == "unavailable"


def test_batch_is_always_safe_without_a_model():
    result = understand_batch([sig("sig-1", LONG_TEXT)], config=None)
    assert result.available is False
    assert result.statements == []


def test_valid_statement_survives_the_model_round_trip():
    s = sig("sig-001", LONG_TEXT, url="https://news.example/story/1")
    result = understand([s], config=config(), transport=fake_transport(valid_payload(s)))

    assert result.available is True
    assert result.dropped == 0
    assert len(result.statements) == 1

    statement = result.statements[0]
    assert statement.understanding == "llm"
    assert statement.domain == "billing automation"
    assert statement.unanswered == ["What triggers the manual step?"]

    quote = statement.claims[0].quotes[0]
    assert quote.url == "https://news.example/story/1"
    assert quote.signal_id == "sig-001"
    assert quote.source == "hackernews"
    assert verify(statement, [s]).ok is True


def test_model_output_with_a_fabricated_quote_is_dropped():
    s = sig("sig-001", LONG_TEXT)
    payload = valid_payload(s)
    payload["problems"][0]["evidence"] = (
        "This is not text that appears anywhere in the fetched signal."
    )
    result = understand([s], config=config(), transport=fake_transport(payload))

    assert result.statements == []
    assert result.dropped == 1
    assert any("not verbatim" in r for r in result.rejected)


def test_model_cannot_cite_a_screened_out_signal():
    usable = sig("sig-1", LONG_TEXT)
    noise = sig("sig-n", "https://example.com/some/link")
    payload = {
        "problems": [
            {
                "signal_id": "sig-n",
                "has_problem": True,
                "problem": "Something is broken.",
                "evidence": noise.excerpt,
            }
        ]
    }
    result = understand(
        [usable, noise], config=config(), transport=fake_transport(payload)
    )
    assert result.statements == []
    assert result.screened >= 1
    assert any("sig-n" in r for r in result.rejected)


def test_model_reply_without_a_problems_list_is_reported_not_raised():
    s = sig("sig-1", LONG_TEXT)

    def broken(cfg, body):
        return {"choices": [{"message": {"content": "I refuse."}}]}

    result = understand([s], config=config(), transport=broken)
    assert result.available is False
    assert "JSON" in result.reason


def test_model_kept_fully_rejects_commentary_as_a_statement():
    commentary = sig("sig-c", "lol, moderators removed that post anyway.")
    kept, skipped = screen_signals([commentary])
    assert kept == []
    result = understand([commentary], config=config(), transport=fake_transport({}))
    assert result.available is False
    assert result.statements == []


def test_parse_statements_rejects_empty_output():
    kept, rejected = parse_statements({"problems": []}, [])
    assert kept == [] and rejected == []


# --------------------------------------------------------------------------- #
# Evidence boundary: the extraction contract is enforced twice - in the prompt
# and again as a deterministic wall in parse_statements/_violation_reason.
# --------------------------------------------------------------------------- #
def test_genuine_workflow_problem_is_accepted():
    s = sig("sig-1", LONG_TEXT)
    result = understand([s], config=config(), transport=fake_transport(valid_payload(s)))
    assert len(result.statements) == 1
    assert result.dropped == 0


def test_solution_suggestion_is_rejected():
    s = sig("sig-1", LONG_TEXT)
    payload = entry_payload(
        s, "Build an AI-powered email automation platform for freelancers."
    )
    result = understand([s], config=config(), transport=fake_transport(payload))
    assert result.statements == []
    assert result.dropped == 1
    assert any("solution" in r for r in result.rejected)


def test_generic_wish_for_an_app_is_rejected():
    s = sig("sig-1", LONG_TEXT)
    payload = entry_payload(s, "I wish there was an app for organizing receipts.")
    result = understand([s], config=config(), transport=fake_transport(payload))
    assert result.statements == []
    assert any("wish or opportunity language" in r for r in result.rejected)


def test_emotional_venting_without_a_mechanism_is_rejected():
    s = sig("sig-1", LONG_TEXT)
    payload = entry_payload(s, "This is just so annoying.")
    result = understand([s], config=config(), transport=fake_transport(payload))
    assert result.statements == []
    assert any("emotional venting" in r for r in result.rejected)


def test_commentary_only_signal_is_rejected_at_the_contract():
    s = sig("sig-c", LONG_TEXT)
    payload = {
        "problems": [
            {
                "signal_id": "sig-c",
                "has_problem": False,
                "reject_reason": "commentary on moderation, not a problem",
            }
        ]
    }
    result = understand([s], config=config(), transport=fake_transport(payload))
    assert result.statements == []
    assert result.dropped == 1
    assert any("sig-c" in r and "commentary" in r for r in result.rejected)


def test_exact_verbatim_quote_is_accepted():
    s = sig("sig-1", LONG_TEXT)
    payload = entry_payload(s, "Billing data is re-keyed by hand every month.")
    result = understand([s], config=config(), transport=fake_transport(payload))
    assert len(result.statements) == 1
    quote = result.statements[0].claims[0].quotes[0]
    assert quote.text == s.excerpt[:60]
    assert quote.signal_id == s.id


def test_altered_quote_is_rejected_by_the_verifier():
    s = sig("sig-1", LONG_TEXT)
    payload = entry_payload(
        s, "Billing data is re-keyed by hand every month.", evidence=LONG_TEXT[:20] + " KEPT THE SAME"
    )
    result = understand([s], config=config(), transport=fake_transport(payload))
    assert result.statements == []
    assert any("not verbatim" in r for r in result.rejected)


def test_statement_with_unsupported_facts_is_rejected():
    s = sig("sig-1", LONG_TEXT)
    payload = entry_payload(
        s, "90% of freelancers re-enter billing data by hand every month."
    )
    result = understand([s], config=config(), transport=fake_transport(payload))
    assert result.statements == []
    assert any("evidence does not contain" in r for r in result.rejected)


def test_statement_proposing_a_specific_solution_is_rejected():
    s = sig("sig-1", LONG_TEXT)
    payload = entry_payload(
        s, "The company should add a one-click import API to end manual entry."
    )
    result = understand([s], config=config(), transport=fake_transport(payload))
    assert result.statements == []
    assert any("solution" in r for r in result.rejected)


def test_concrete_engineering_shaped_statement_is_accepted():
    s = sig("sig-1", LONG_TEXT)
    statement = (
        "Billing amounts from incoming emails require manual extraction "
        "into Google Sheets."
    )
    result = understand([s], config=config(), transport=fake_transport(entry_payload(s, statement)))
    assert len(result.statements) == 1
    assert result.statements[0].statement == statement


def test_supported_number_in_the_statement_is_not_rejected():
    s = sig("sig-1", LONG_TEXT + " It costs $15 per dispute to process.")
    payload = entry_payload(
        s,
        "Processing a disputed billing charge costs $15 per dispute.",
        evidence=LONG_TEXT + " It costs $15 per dispute to process.",
    )
    result = understand([s], config=config(), transport=fake_transport(payload))
    assert len(result.statements) == 1


def test_question_only_evidence_is_rejected():
    question = "Why doesn't GitHub support the fast-forward merge strategy?"
    s = sig("sig-q1", question)
    payload = entry_payload(
        s,
        "GitHub does not support the fast-forward merge strategy.",
        evidence=question,
    )
    result = understand([s], config=config(), transport=fake_transport(payload))
    assert result.statements == []
    assert result.dropped == 1
    assert any("bare question" in r for r in result.rejected)


def test_question_evidence_that_reports_a_failure_is_accepted():
    question = "Why does the exporter crash on every large file?"
    s = sig("sig-q2", question)
    payload = entry_payload(
        s,
        "The exporter crashes on every large file.",
        evidence=question,
    )
    result = understand([s], config=config(), transport=fake_transport(payload))
    assert len(result.statements) == 1
    assert result.dropped == 0


def test_statement_that_is_itself_a_question_is_rejected():
    s = sig("sig-1", LONG_TEXT)
    payload = entry_payload(s, "Is there no way to import my invoicing history?")
    result = understand([s], config=config(), transport=fake_transport(payload))
    assert result.statements == []
    assert any("question" in r for r in result.rejected)


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #
def test_problems_subcommand_exists_with_expected_flags():
    parser = build_parser()
    args = parser.parse_args(["problems", "--json", "--limit", "5", "--fresh"])
    assert args.command == "problems"
    assert args.func.__name__ == "cmd_problems"
    assert args.limit == 5 and args.fresh is True


def test_problems_without_a_model_says_so_and_exits_zero(monkeypatch, capsys):
    import impara.cli as cli

    monkeypatch.setattr(cli, "_collect_signals", lambda args: [sig("sig-1", LONG_TEXT)])
    monkeypatch.setattr(cli, "config_from_env", lambda: None)

    code = main(["problems", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert payload["available"] is False
    assert payload["problems"] == []
    assert payload["signal_count"] == 1
    assert "IMPARA_LLM_BASE_URL" in payload["reason"]


def test_problems_text_output_names_the_fallback(monkeypatch, capsys):
    import impara.cli as cli

    monkeypatch.setattr(cli, "_collect_signals", lambda args: [])
    monkeypatch.setattr(cli, "config_from_env", lambda: None)

    code = main(["problems"])
    out = capsys.readouterr().out
    assert code == 0
    assert "SEMANTIC UNDERSTANDING UNAVAILABLE" in out
    assert "impara discover" in out


def test_problems_presents_a_verified_pool(monkeypatch, capsys):
    import impara.cli as cli

    s = sig("sig-1", LONG_TEXT)
    monkeypatch.setattr(cli, "_collect_signals", lambda args: [s])
    monkeypatch.setattr(cli, "config_from_env", lambda: config())
    monkeypatch.setattr(
        cli,
        "understand_batch",
        lambda signals, config=None, batch_size=40: understand(
            signals, config=config, transport=fake_transport(valid_payload(s))
        ),
    )

    code = main(["problems", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["available"] is True
    assert len(payload["problems"]) == 1
    claim = payload["problems"][0]["claims"][0]
    assert claim["quotes"][0]["url"] == s.url
    assert claim["quotes"][0]["signal_id"] == "sig-1"


def test_rendered_pool_shows_the_evidence_chain():
    from impara.render import render_problem_pool
    from impara.understand import UnderstandingResult

    s = sig("sig-001", LONG_TEXT, url="https://news.example/story/1")
    kept, _ = parse_statements(valid_payload(s), [s])
    result = UnderstandingResult(
        statements=kept, model="test-model", signals_seen=1
    )
    text = render_problem_pool(result, limit=5)
    assert "PROBLEM 001" in text
    assert "EVIDENCE" in text
    assert "<- sig-001 (hackernews) https://news.example/story/1" in text
