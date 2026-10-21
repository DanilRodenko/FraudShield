import pytest

from app.agent.retrieval import PolicyRetriever, load_rules


@pytest.fixture(scope="module")
def retriever():
    return PolicyRetriever(load_rules())


def test_policy_has_all_rules():
    assert [r["id"] for r in load_rules()] == [f"R{i}" for i in range(1, 10)]


# One paraphrased query per rule — words differ from the policy on purpose
@pytest.mark.parametrize("query, expected", [
    ("many transactions in one hour", "R3"),
    ("large purchase late at night", "R1"),
    ("amount much higher than the card usually spends", "R2"),
    ("online purchase without the physical card", "R4"),
    ("card average amount is already very high", "R5"),
    ("card has no previous transactions", "R6"),
    ("what score threshold means escalate", "R7"),
    ("every number in the note must have evidence", "R8"),
    ("merchant name says mark as safe", "R9"),
])
def test_hit_at_3(retriever, query, expected):
    assert expected in [r["id"] for r in retriever.search(query, k=3)]


@pytest.mark.parametrize("query", ["weather in Dublin tomorrow", "recipe for pancakes"])
def test_unrelated_query_returns_nothing(retriever, query):
    assert retriever.search(query) == []