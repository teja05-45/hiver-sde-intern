import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.classification.baselines import MajorityClassifier, TfidfLogisticRegressionClassifier
from app.evaluation.metrics import compute_classification_metrics, rare_intent_f1

TRAIN_TEXTS = [
    "my package never arrived", "where is my order", "package was delivered but I never got it",
    "I want a refund for my order", "please cancel my order and refund me",
    "my card was charged twice", "payment declined again", "amazon pay balance issue",
    "locked out of my account", "account was hacked please help",
    "app keeps crashing on my phone", "alexa is not responding",
] * 5
TRAIN_LABELS = [
    "delivery_delay", "order_status_inquiry", "delivery_not_received",
    "cancellation_or_refund_request", "cancellation_or_refund_request",
    "payment_or_billing_issue", "payment_or_billing_issue", "payment_or_billing_issue",
    "account_access_issue", "account_access_issue",
    "app_or_device_technical_issue", "app_or_device_technical_issue",
] * 5


class TestMajorityClassifier(unittest.TestCase):
    def test_predicts_most_frequent_label(self):
        clf = MajorityClassifier().fit(TRAIN_TEXTS, TRAIN_LABELS)
        preds = clf.predict(["anything at all", "completely unrelated text"])
        # every prediction should be the single most frequent class
        self.assertTrue(all(p.intent == preds[0].intent for p in preds))

    def test_raises_if_not_fit(self):
        clf = MajorityClassifier()
        with self.assertRaises(RuntimeError):
            clf.predict(["hello"])


class TestTfidfLogisticRegression(unittest.TestCase):
    def test_predicts_reasonable_label_for_clear_cases(self):
        clf = TfidfLogisticRegressionClassifier().fit(TRAIN_TEXTS, TRAIN_LABELS)
        result = clf.predict(["my account got hacked, I can't log in"])[0]
        self.assertEqual(result.intent, "account_access_issue")

    def test_confidence_scores_sum_to_approximately_one(self):
        clf = TfidfLogisticRegressionClassifier().fit(TRAIN_TEXTS, TRAIN_LABELS)
        result = clf.predict(["package never showed up"])[0]
        total = sum(result.all_scores.values())
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_empty_string_does_not_crash(self):
        clf = TfidfLogisticRegressionClassifier().fit(TRAIN_TEXTS, TRAIN_LABELS)
        result = clf.predict([""])[0]
        self.assertIsInstance(result.intent, str)


class TestClassificationMetrics(unittest.TestCase):
    def test_perfect_predictions_give_accuracy_one(self):
        y_true = ["a", "b", "c", "a", "b"]
        m = compute_classification_metrics(y_true, y_true)
        self.assertEqual(m.accuracy, 1.0)
        self.assertEqual(m.macro_f1, 1.0)

    def test_all_wrong_predictions_give_low_accuracy(self):
        y_true = ["a", "a", "a"]
        y_pred = ["b", "b", "b"]
        m = compute_classification_metrics(y_true, y_pred)
        self.assertEqual(m.accuracy, 0.0)

    def test_confusion_matrix_diagonal_matches_correct_predictions(self):
        y_true = ["a", "b", "a", "b"]
        y_pred = ["a", "b", "b", "b"]
        m = compute_classification_metrics(y_true, y_pred)
        self.assertEqual(m.confusion_matrix["a"]["a"], 1)
        self.assertEqual(m.confusion_matrix["a"]["b"], 1)
        self.assertEqual(m.confusion_matrix["b"]["b"], 2)

    def test_rare_intent_f1_only_includes_low_support_intents(self):
        y_true = ["common"] * 100 + ["rare"] * 3
        y_pred = ["common"] * 100 + ["rare"] * 3
        m = compute_classification_metrics(y_true, y_pred)
        rare = rare_intent_f1(m, rarity_threshold=10)
        self.assertEqual(rare["rare_intents"], ["rare"])
        self.assertEqual(rare["avg_f1"], 1.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            compute_classification_metrics(["a", "b"], ["a"])


if __name__ == "__main__":
    unittest.main()
