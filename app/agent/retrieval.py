from importlib.resources import read_text
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICY_PATH = Path("docs/fraud_policy.md")

MIN_SCORE = 0.2

def load_rules(path: Path = POLICY_PATH) -> list[dict]:
    """Splits the policy into rules: [{"id": "R1", "title": ..., "text": ...}, ...]."""
    text = path.read_text()
    chunks = text.split("\n## ")

    rules = []
    for chunk in chunks[1:]:  # chunks[0] is the document header, not a rule
        first_line, body = chunk.split("\n", 1)        # heading line vs rule text
        rule_id, title = first_line.split(" — ", 1)    # "R1", "High amount at night"
        rules.append({"id": rule_id.strip(), "title": title.strip(), "text": body.strip()})
    return rules


class PolicyRetriever:
    """TF-IDF search over policy rules."""

    def __init__(self, rules: list[dict]):
        self.rules = rules
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.texts = [r["title"] + " " + r["text"] for r in rules]
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Returns the k most relevant rules with a similarity score."""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = similarities.argsort()[::-1][:k]

        # Copy each rule and add its score; missing keys should fail loudly, not silently
        return [
            {**self.rules[i], "score": round(float(similarities[i]), 3)}
            for i in top_indices
            if similarities[i] >= MIN_SCORE  # drop unrelated rules instead of returning noise
        ]
