"""
Baseline intent classifiers.

Two baselines, per the assignment's explicit requirement -- these exist to
give the proposed system's numbers a meaningful floor to beat, not to be
straw men:

  - MajorityClassifier: always predicts the single most frequent training
    label. Any classifier that can't beat this isn't doing classification.
  - TfidfLogisticRegressionClassifier: a real, competent classical
    baseline. This is what a reasonable engineer would ship if they had a
    day and no LLM budget -- if the "AI agent" can't clearly beat this, the
    LLM/retrieval complexity isn't earning its keep.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass
class ClassificationResult:
    intent: str
    confidence: float
    all_scores: dict[str, float]


class MajorityClassifier:
    def __init__(self) -> None:
        self.majority_label: str | None = None
        self.label_distribution: Counter | None = None

    def fit(self, texts: list[str], labels: list[str]) -> "MajorityClassifier":
        self.label_distribution = Counter(labels)
        self.majority_label = self.label_distribution.most_common(1)[0][0]
        return self

    def predict(self, texts: list[str]) -> list[ClassificationResult]:
        if self.majority_label is None:
            raise RuntimeError("MajorityClassifier not fit yet.")
        total = sum(self.label_distribution.values())
        conf = self.label_distribution[self.majority_label] / total
        return [
            ClassificationResult(intent=self.majority_label, confidence=conf, all_scores={self.majority_label: conf})
            for _ in texts
        ]


class TfidfLogisticRegressionClassifier:
    """TF-IDF vectorizer + multinomial logistic regression.

    Deliberately a scikit-learn `Pipeline` so the fitted vectorizer and
    classifier serialize/deserialize together, and so evaluation code
    doesn't need to know the two are separate stages.
    """

    def __init__(self, seed: int = 42, max_features: int = 20000, C: float = 2.0) -> None:
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=max_features, min_df=2, max_df=0.5, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, C=C, class_weight="balanced", random_state=seed)),
        ])
        self.classes_: list[str] = []

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfLogisticRegressionClassifier":
        # See TfidfEmbeddingProvider.fit for why small corpora need
        # adaptive min_df/max_df (avoids "empty vocabulary" / "max_df
        # corresponds to < documents than min_df" on tiny test fixtures).
        if len(texts) < 20:
            self.pipeline.set_params(tfidf__min_df=1, tfidf__max_df=1.0)
        self.pipeline.fit(texts, labels)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        return self

    def predict(self, texts: list[str]) -> list[ClassificationResult]:
        probs = self.pipeline.predict_proba(texts)
        results = []
        for row in probs:
            scores = dict(zip(self.classes_, row.tolist()))
            best_intent = max(scores, key=scores.get)
            results.append(ClassificationResult(intent=best_intent, confidence=scores[best_intent], all_scores=scores))
        return results
