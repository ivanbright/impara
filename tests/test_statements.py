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
                "statement": "Freelancers re-enter invoicing data every month.",
                "domain": "Finance",
                "who": ["freelancer"],
                "claims": [
                    {
                        "text": "Nothing imports their invoicing history.",
                        "kind": "observed",
                        "quotes": [
                            {"signal_id": signal.id, "quote": signal.excerpt[:60]}
                        ],
                    }
                ],
                "unanswered": ["Would they pay to fix it?"],
            }
        ]
    }


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
    assert statement.domain == "Finance"
    assert statement.unanswered == ["Would they pay to fix it?"]

    quote = statement.claims[0].quotes[0]
    assert quote.url == "https://news.example/story/1"
    assert quote.signal_id == "sig-001"
    assert quote.source == "hackernews"
    assert verify(statement, [s]).ok is True


def test_model_output_with_a_fabricated_quote_is_dropped():
    s = sig("sig-001", LONG_TEXT)
    payload = valid_payload(s)
    payload["problems"][0]["claims"][0]["quotes"][0]["quote"] = (
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
                "statement": "Something is broken.",
                "claims": [
                    {
                        "text": "It is broken.",
                        "kind": "observed",
                        "quotes": [{"signal_id": "sig-n", "quote": noise.excerpt}],
                    }
                ],
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


def test_inferred_claims_without_quotes_are_pruned_not_trusted():
    s = sig("sig-001", LONG_TEXT)
    payload = {
        "problems": [
            {
                "statement": "Freelancers re-enter invoicing data every month.",
                "claims": [
                    {
                        "text": "Nothing imports their invoicing history.",
                        "kind": "observed",
                        "quotes": [
                            {"signal_id": s.id, "quote": s.excerpt[:60]}
                        ],
                    },
                    {
                        "text": "They would pay for a fix.",
                        "kind": "observed",
                        "quotes": [],
                    },
                ],
            }
        ]
    }
    kept, rejected = parse_statements(payload, [s])
    assert len(kept) == 1
    assert rejected == []
    assert len(kept[0].claims) == 1
    assert kept[0].quote_count == 1


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
        lambda signals, config=None: understand(
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
