import json

import pytest

from app.agent.guardrails import (
    check_action_claims,
    check_numbers,
    check_recommendation,
    mask_card_numbers,
    validate,
)

POLICY = "R1: above $500 between 22:00 and 03:59. R7: score 0.02 review, 0.5 escalate. R5: above $300."

AUDIT = [
    {"step": 1, "tool": "get_transaction", "args": "{}",
     "result": {"time": "2020-09-04 23:54", "card": "**** 8089", "amount": 1113.85, "category": "shopping_net"}},
    {"step": 2, "tool": "get_model_assessment", "args": "{}",
     "result": {"score": 0.987, "flagged": True, "high_priority": True,
                "policy_minimum_action": {"action": "escalate", "rule": "R7"},
                "features": {"Card's usual amount": 671.08}}},
]

GOOD_NOTE = """SUMMARY: The $1,113.85 online purchase at 23:54 is high risk.
EVIDENCE:
- Amount $1,113.85 at 23:54 (R1)
- Card's usual amount $671.08 is above $300 (R5)
- Model score 0.99, at or above 0.5 (R7)
RECOMMENDATION: escalate"""


def test_good_note_passes():
    result = validate(GOOD_NOTE, AUDIT, POLICY)
    assert result["ok"], result["errors"]


def test_invented_number_is_caught():
    # 1.66x was computed by the LLM, not returned by any tool
    errors = check_numbers("Amount is 1.66x the usual", [json.dumps(AUDIT[1]["result"]), POLICY])
    assert any("1.66" in e for e in errors)


def test_rounded_number_is_accepted():
    # tool says 0.987, note says 0.99 -> fine
    assert check_numbers("score 0.99", ['{"score": 0.987}']) == []


def test_rule_ids_are_not_numbers():
    assert check_numbers("see (R7) and (R12)", ["nothing"]) == []


@pytest.mark.parametrize("rec, minimum, ok", [
    ("escalate", "escalate", True),
    ("review", "escalate", False),   # the first real bug we saw
    ("escalate", "review", True),    # being stricter than the minimum is allowed
    ("no_action", "review", False),
])
def test_recommendation_never_below_minimum(rec, minimum, ok):
    assert (check_recommendation(rec, minimum) == []) == ok


@pytest.mark.parametrize("text, claims", [
    ("The card has been blocked.", True),
    ("I have contacted the customer.", True),
    ("Recommend a temporary card block (the analyst decides).", False),
])
def test_action_claims(text, claims):
    assert bool(check_action_claims(text)) == claims


def test_full_card_number_is_masked():
    assert mask_card_numbers("card 4111111111111111 used") == "card **** 1111 used"


def test_review_below_minimum_fails_validation():
    note = GOOD_NOTE.replace("RECOMMENDATION: escalate", "RECOMMENDATION: review")
    result = validate(note, AUDIT, POLICY)
    assert not result["ok"]


def test_missing_note_fails():
    assert not validate(None, AUDIT, POLICY)["ok"]