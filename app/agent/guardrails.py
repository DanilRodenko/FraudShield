import json
import re

RANK = {"no_action": 0, "review": 1, "escalate": 2}

# Numbers like 1,113.85 / 0.02 / 23 — but not rule IDs (R7) or parts of words
NUMBER_RE = re.compile(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?")
RULE_ID_RE = re.compile(r"\bR\d+\b")
# Claims that an action was taken (the agent can only recommend)
ACTION_CLAIM_RE = re.compile(
    r"\b(has been|have been|was|were|I have|I've|we have|we've)\s+(blocked|frozen|suspended|contacted|cancelled)\b",
    re.IGNORECASE,
)
# 13-19 consecutive digits = looks like a full card number
CARD_RE = re.compile(r"\b\d{13,19}\b")


def parse_note(note: str) -> dict:
    """Splits the note into summary / evidence / recommendation (missing parts -> None)."""
    summary = re.search(r"SUMMARY:\s*(.+)", note)
    evidence = re.search(r"EVIDENCE:\s*(.+?)\s*RECOMMENDATION:", note, re.DOTALL)
    rec = re.search(r"RECOMMENDATION:\s*([a-z_]+)", note)
    return {
        "summary": summary.group(1).strip() if summary else None,
        "evidence": evidence.group(1).strip() if evidence else None,
        "recommendation": rec.group(1) if rec else None,
    }


def check_format(parsed: dict) -> list[str]:
    """All three sections present and the recommendation is one of the allowed values."""
    errors = [f"missing {part}" for part in ("summary", "evidence", "recommendation") if not parsed[part]]
    if parsed["recommendation"] and parsed["recommendation"] not in RANK:
        errors.append(f"unknown recommendation: {parsed['recommendation']}")
    return errors


def check_recommendation(recommendation: str | None, minimum: str | None) -> list[str]:
    """Policy R7: the agent may not recommend less than the minimum computed in code."""
    if recommendation not in RANK or minimum not in RANK:
        return []  # format problems are reported by check_format
    if RANK[recommendation] < RANK[minimum]:
        return [f"recommendation '{recommendation}' is below policy minimum '{minimum}' (R7)"]
    return []


def _numbers(text: str) -> list[str]:
    return NUMBER_RE.findall(RULE_ID_RE.sub(" ", text))


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def check_numbers(note: str, sources: list[str]) -> list[str]:
    """Every number in the note must appear in a source (tool results or policy).
    A number counts as found if a source number rounds to it (0.987 supports '0.99')."""
    allowed = [_to_float(n) for src in sources for n in _numbers(src)]
    errors = []
    for raw in set(_numbers(note)):
        value = _to_float(raw)
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if not any(round(a, decimals) == value for a in allowed):
            errors.append(f"number not found in tool results or policy: {raw}")
    return errors


def check_action_claims(note: str) -> list[str]:
    """The agent recommends only; it must never claim it blocked a card or contacted anyone."""
    return [f"claims an action was taken: '{m.group(0)}'" for m in ACTION_CLAIM_RE.finditer(note)]


def mask_card_numbers(note: str) -> str:
    """Defence in depth: tools already mask cards, but never show a full card number."""
    return CARD_RE.sub(lambda m: "**** " + m.group(0)[-4:], note)


def validate(note: str | None, audit: list[dict], policy_text: str) -> dict:
    """Runs all checks. ok=False means the note must not be shown as-is."""
    if not note:
        return {"ok": False, "errors": ["no note produced"], "note": None, "recommendation": None}

    note = mask_card_numbers(note)
    parsed = parse_note(note)

    # Minimum action comes from the tool output (computed in code), not from the LLM
    minimum = None
    for entry in audit:
        if entry["tool"] == "get_model_assessment" and "policy_minimum_action" in entry["result"]:
            minimum = entry["result"]["policy_minimum_action"]["action"]

    sources = [json.dumps(entry["result"]) for entry in audit] + [policy_text]
    errors = (
        check_format(parsed)
        + check_recommendation(parsed["recommendation"], minimum)
        + check_numbers(note, sources)
        + check_action_claims(note)
    )
    if minimum is None:
        errors.append("model assessment was never retrieved")
    return {"ok": not errors, "errors": errors, "note": note, "recommendation": parsed["recommendation"]}