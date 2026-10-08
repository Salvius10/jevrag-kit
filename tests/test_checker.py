import json

import httpx2
import pytest

from helpers import choice_response, typesafe_transport
from jevrag_kit.checker import (
    TypeSafeClaimVerifier,
    assemble_answer,
    check_claim,
    feedback_for,
    locate,
    needs_regeneration,
    release,
    verify_draft,
    withheld_sentence,
)
from jevrag_kit.config import AnswerConfig, CheckerConfig, parse_config
from jevrag_kit.types import Claim, Draft, Passage, RelationResult

RENAL = "For patients with a creatinine clearance below 30 mL/min, reduce the dose\nto 25 mg once daily."
PREG = "Corvalan is not approved for use during pregnancy; discontinue it as soon as pregnancy is detected."


def P(pid: str, text: str) -> Passage:
    return Passage(id=pid, title=f"Label: {pid}", text=text)


SUPPLIED = {"renal": P("renal", RENAL), "preg": P("preg", PREG)}
POLICY = CheckerConfig()


def C(quote: str, pid: str = "renal", cid: str = "c1", ctype: str = "answer", text: str = "Reduce to 25 mg.") -> Claim:
    return Claim(id=cid, type=ctype, text=text, passage_id=pid, quote=quote)


# --- locate ----------------------------------------------------------------------------------


def test_locate_matches_across_line_wraps():
    r = locate(C("reduce the dose to 25 mg once daily"), SUPPLIED, 20)
    assert r.status == "found" and r.passage.id == "renal"


def test_locate_folds_curly_quotes_and_unicode_forms():
    passage = P("q", "The label says “not recommended for patients on dialysis” in bold.")
    assert locate(C('"not recommended for patients on dialysis"', pid="q"), {"q": passage}, 20).status == "found"
    wide = P("w", "Dose is　twenty five milligrams daily.")  # NBSP and ideographic space
    assert locate(C("Dose is twenty five milligrams", pid="w"), {"w": wide}, 20).status == "found"


def test_locate_too_short_uses_the_configured_minimum():
    assert locate(C("25 mg"), SUPPLIED, 20).status == "too_short"
    assert locate(C("25 mg once daily"), SUPPLIED, 5).status == "found"


def test_locate_missing_and_reworded_quotes_fail():
    assert locate(C("reduce the dose to 20 mg once daily"), SUPPLIED, 20).status == "missing"
    assert locate(C("lower the dose to 25 mg once daily"), SUPPLIED, 20).status == "missing"


def test_locate_reattributes_to_the_passage_that_has_the_quote():
    r = locate(C("not approved for use during pregnancy", pid="renal"), SUPPLIED, 20)
    assert r.status == "reattributed" and r.passage.id == "preg"
    r = locate(C("not approved for use during pregnancy", pid="unknown-id"), SUPPLIED, 20)
    assert r.status == "reattributed" and r.passage.id == "preg"


# --- release policy: one test per row --------------------------------------------------------


def test_release_table():
    assert release("verified", 0.90, POLICY) == ("ship", None)
    assert release("verified", 0.89, POLICY) == ("review", "low_confidence")
    assert release("verified", None, POLICY) == ("review", "low_confidence")
    assert release("unsupported", 0.99, POLICY) == ("review", "unsupported")
    assert release("contradicted", 0.99, POLICY) == ("drop", None)
    assert release("fabricated", None, POLICY) == ("drop", None)


def test_release_actions_are_configurable():
    strict = CheckerConfig(auto_accept=0.8, low_confidence_action="drop", unsupported_action="drop")
    assert release("verified", 0.8, strict) == ("ship", None)
    assert release("verified", 0.79, strict) == ("drop", None)
    assert release("unsupported", 0.99, strict) == ("drop", None)


# --- claim checks ----------------------------------------------------------------------------


class StubVerifier:
    def __init__(self, choice="supports", confidence=0.95, raises=False):
        self.calls = []
        self.choice, self.confidence, self.raises = choice, confidence, raises

    def relation(self, claim, section):
        self.calls.append((claim, section))
        if self.raises:
            raise RuntimeError("timeout")
        probs = {"supports": 0.02, "contradicts": 0.02, "says_nothing": 0.02, self.choice: 0.96}
        return RelationResult(choice=self.choice, probabilities=probs, confidence=self.confidence, input_tokens=40, output_tokens=2)


def test_fabricated_claim_makes_no_model_call():
    v = StubVerifier()
    check = check_claim(C("this sentence is not in the passage at all"), SUPPLIED, v, POLICY, 1)
    assert (check.verdict, check.action, check.locate, v.calls) == ("fabricated", "drop", "missing", [])


def test_reattributed_claim_is_checked_against_the_real_passage():
    v = StubVerifier()
    check = check_claim(C("not approved for use during pregnancy", pid="renal"), SUPPLIED, v, POLICY, 1)
    assert check.claim.passage_id == "preg" and check.cited_passage_id == "renal"
    assert v.calls[0] == ("Reduce to 25 mg.", PREG)
    assert check.action == "ship" and (check.input_tokens, check.output_tokens) == (40, 2)


def test_contradicted_claim_drops():
    check = check_claim(C("reduce the dose to 25 mg once daily"), SUPPLIED, StubVerifier("contradicts"), POLICY, 1)
    assert (check.verdict, check.action, check.relation) == ("contradicted", "drop", "contradicts")


def test_verifier_error_never_ships():
    check = check_claim(C("reduce the dose to 25 mg once daily"), SUPPLIED, StubVerifier(raises=True), POLICY, 1)
    assert (check.verdict, check.action) == ("unsupported", "drop") and "timeout" in check.error


def test_unexpected_relation_label_never_ships():
    class Odd:
        def relation(self, claim, section):
            return RelationResult(choice="maybe", probabilities={}, confidence=0.9)  # invalid label

    check = check_claim(C("reduce the dose to 25 mg once daily"), SUPPLIED, Odd(), POLICY, 1)
    assert check.action == "drop" and "ValidationError" in check.error


def test_claim_check_to_dict_shape():
    d = check_claim(C("reduce the dose to 25 mg once daily"), SUPPLIED, StubVerifier(), POLICY, 2).to_dict()
    assert set(d) == {
        "round", "claim", "cited_passage_id", "locate", "relation", "probabilities", "confidence",
        "verdict", "action", "review_reason", "error", "input_tokens", "output_tokens", "latency_ms",
    }
    assert d["round"] == 2 and d["claim"] == C("reduce the dose to 25 mg once daily").model_dump()


def test_verify_draft_keeps_order_and_regeneration_rule():
    draft = Draft(insufficient=False, claims=[C("reduce the dose to 25 mg once daily"), C("no such quote anywhere here", cid="c2")])
    shipped = verify_draft(draft, SUPPLIED, StubVerifier(), POLICY)
    assert [c.claim.id for c in shipped] == ["c1", "c2"] and not needs_regeneration(shipped)
    withheld = verify_draft(draft, SUPPLIED, StubVerifier("says_nothing"), POLICY, round_no=1, max_workers=1)
    assert needs_regeneration(withheld)
    assert needs_regeneration([]) is False
    assert verify_draft(Draft(insufficient=True), SUPPLIED, StubVerifier(), POLICY) == []


def test_feedback_explains_each_failure():
    checks = [
        check_claim(C("25 mg", cid="c1"), SUPPLIED, StubVerifier(), POLICY, 1),
        check_claim(C("no such quote anywhere at all", cid="c2"), SUPPLIED, StubVerifier(), POLICY, 1),
        check_claim(C("reduce the dose to 25 mg once daily", cid="c3"), SUPPLIED, StubVerifier(raises=True), POLICY, 1),
        check_claim(C("reduce the dose to 25 mg once daily", cid="c4"), SUPPLIED, StubVerifier("contradicts"), POLICY, 1),
        check_claim(C("reduce the dose to 25 mg once daily", cid="c5"), SUPPLIED, StubVerifier("says_nothing"), POLICY, 1),
        check_claim(C("reduce the dose to 25 mg once daily", cid="c6"), SUPPLIED, StubVerifier("supports", 0.5), POLICY, 1),
        check_claim(C("reduce the dose to 25 mg once daily", cid="c7"), SUPPLIED, StubVerifier(), POLICY, 1),
    ]
    assert feedback_for(checks) == [
        'c1 ("Reduce to 25 mg."): the quote is too short to verify',
        'c2 ("Reduce to 25 mg."): the quote was not found word for word in any supplied passage',
        'c3 ("Reduce to 25 mg."): it could not be checked',
        'c4 ("Reduce to 25 mg."): passage [renal] contradicts it',
        'c5 ("Reduce to 25 mg."): passage [renal] does not address it',
        'c6 ("Reduce to 25 mg."): support in [renal] was not clear enough',
    ]


# --- assembly --------------------------------------------------------------------------------


def _checks(spec, policy=POLICY):
    out = []
    for i, (ctype, choice, conf, quote, pid) in enumerate(spec, start=1):
        claim = C(quote, pid=pid, cid=f"c{i}", ctype=ctype, text=f"Claim {i}.")
        out.append(check_claim(claim, SUPPLIED, StubVerifier(choice, conf), policy, 1))
    return out


def test_assembly_orders_corrections_first_and_cites_every_sentence():
    checks = _checks(
        [
            ("answer", "supports", 0.95, "reduce the dose to 25 mg once daily", "renal"),
            ("premise_correction", "supports", 0.97, "not approved for use during pregnancy", "preg"),
        ]
    )
    answer = assemble_answer(checks, Draft(insufficient=False))
    assert answer.status == "answered" and answer.withheld_count == 0 and answer.reason is None
    assert answer.text.split("\n") == ["Claim 2. [preg]", "Claim 1. [renal]"]
    assert [c.id for c in answer.claims] == ["c2", "c1"] and answer.claims[0].confidence == 0.97
    assert answer.source_ids == ["preg", "renal"]


def test_assembly_partial_appends_withheld_sentence():
    checks = _checks(
        [
            ("answer", "supports", 0.95, "reduce the dose to 25 mg once daily", "renal"),
            ("answer", "supports", 0.50, "not approved for use during pregnancy", "preg"),
            ("answer", "says_nothing", 0.80, "creatinine clearance below 30 mL/min", "renal"),
        ]
    )
    answer = assemble_answer(checks, Draft(insufficient=False))
    assert answer.status == "partial" and answer.withheld_count == 2
    lines = answer.text.split("\n")
    assert lines[-1] == withheld_sentence(2) == "2 statements were withheld pending review."
    assert [c.id for c in answer.claims] == ["c1"] and answer.source_ids == ["renal"]


def test_assembly_abstains_when_nothing_ships():
    dropped = _checks([("answer", "contradicts", 0.99, "reduce the dose to 25 mg once daily", "renal")])
    answer = assemble_answer(dropped, Draft(insufficient=True, missing="No price information."))
    assert (answer.status, answer.text, answer.reason, answer.missing) == ("abstained", None, "insufficient_evidence", "No price information.")
    assert answer.claims == [] and answer.source_ids == []

    held = _checks([("answer", "supports", 0.5, "reduce the dose to 25 mg once daily", "renal")])
    answer = assemble_answer(held, Draft(insufficient=False))
    assert (answer.status, answer.reason, answer.withheld_count) == ("abstained", "pending_review", 1)

    assert assemble_answer(dropped, Draft(insufficient=False)).reason == "no_verified_claims"


def test_withheld_sentence_singular():
    assert withheld_sentence(1) == "1 statement was withheld pending review."


def test_assembly_templates_are_configurable():
    cfg = AnswerConfig(
        citation_template="{claim_id}/{type}: {text} (source {passage_id})",
        withheld_one="One claim awaits review.",
        withheld_many="{n} claims await review.",
        line_separator=" | ",
    )
    checks = _checks(
        [
            ("answer", "supports", 0.95, "reduce the dose to 25 mg once daily", "renal"),
            ("answer", "says_nothing", 0.80, "not approved for use during pregnancy", "preg"),
        ]
    )
    answer = assemble_answer(checks, Draft(insufficient=False), cfg)
    assert answer.text == "c1/answer: Claim 1. (source renal) | One claim awaits review."
    assert withheld_sentence(3, cfg) == "3 claims await review."


def test_claims_with_braces_in_their_text_are_safe():
    claim = Claim(id="c1", type="answer", text="Use {curly} braces.", passage_id="renal", quote="reduce the dose to 25 mg once daily")
    answer = assemble_answer([check_claim(claim, SUPPLIED, StubVerifier(), POLICY, 1)], Draft(insufficient=False))
    assert answer.text == "Use {curly} braces. [renal]"


# --- TypeSafe relation check (HTTP mocked) ----------------------------------------------------


def test_typesafe_verifier_sends_claim_section_and_configured_wording():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx2.Response(
            200, json=choice_response("supports", 0.93, {"supports": 0.95, "contradicts": 0.02, "says_nothing": 0.03})
        )

    cfg = parse_config("checker:\n  model: jev-preview\n  relation:\n    supports: The handbook states it\n")
    v = TypeSafeClaimVerifier.from_config(cfg, "k", transport=typesafe_transport(handler))
    result = v.relation("Reduce to 25 mg.", RENAL)
    body = seen["body"]
    assert body["model"] == "jev-preview" and v.model == "jev-preview"
    assert body["state"] == {"claim": "Reduce to 25 mg.", "section": RENAL}
    assert body["questions"]["relation"] == {
        "type": "choice",
        "instructions": "How does the section relate to the claim?",
        "criteria": {
            "supports": "The handbook states it",
            "contradicts": "The section states the opposite or implies the claim is false",
            "says_nothing": "The section does not address what the claim asserts",
        },
    }
    assert result.choice == "supports" and result.confidence == pytest.approx(0.93)
    assert result.probabilities["says_nothing"] == pytest.approx(0.03) and result.input_tokens == 50


def test_typesafe_verifier_retry_setting_is_used():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx2.Response(500, json={"error": {"message": "down"}})

    cfg = parse_config("checker:\n  retries: 0\n")
    v = TypeSafeClaimVerifier.from_config(cfg, "k", transport=typesafe_transport(handler))
    check = check_claim(C("reduce the dose to 25 mg once daily"), SUPPLIED, v, cfg.checker, 1)
    assert len(calls) == 1 and check.action == "drop" and check.error
