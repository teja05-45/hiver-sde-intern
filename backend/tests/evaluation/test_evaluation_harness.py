import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.evaluation.agreement import compute_agreement, _weighted_kappa
from app.evaluation.judge import parse_judge_output, JUDGE_DIMENSIONS


class TestAgreementStatistics(unittest.TestCase):
    def test_identical_scores_give_perfect_agreement(self):
        scores = [3, 2, 4, 1, 0, 3, 2]
        result = compute_agreement(scores, scores, "correctness")
        self.assertEqual(result.exact_agreement_rate, 1.0)
        self.assertEqual(result.adjacent_agreement_rate, 1.0)
        self.assertAlmostEqual(result.weighted_kappa, 1.0, places=3)

    def test_completely_opposite_scores_give_low_kappa(self):
        human = [0, 0, 0, 4, 4, 4]
        judge = [4, 4, 4, 0, 0, 0]
        result = compute_agreement(human, judge, "correctness")
        self.assertLess(result.weighted_kappa, 0)

    def test_too_few_examples_returns_none_stats(self):
        result = compute_agreement([3], [3], "correctness")
        self.assertIsNone(result.spearman_r)

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(AssertionError):
            compute_agreement([1, 2, 3], [1, 2], "correctness")

    def test_weighted_kappa_penalizes_large_disagreements_more(self):
        human = [0, 1, 2, 3, 4]
        judge_close = [1, 1, 2, 3, 3]   # off by at most 1
        judge_far = [4, 3, 2, 1, 0]     # maximally opposite
        k_close = _weighted_kappa(human, judge_close)
        k_far = _weighted_kappa(human, judge_far)
        self.assertGreater(k_close, k_far)


class TestJudgeOutputParsing(unittest.TestCase):
    def test_parses_valid_judge_json(self):
        raw = '{"correctness": 3, "groundedness": 4, "helpfulness": 3, "completeness": 2, ' \
              '"actionability": 3, "brand_consistency": 4, "safety": 4, "overall": 3.3, "reason": "ok"}'
        result = parse_judge_output(raw, is_mock=False)
        self.assertIsNone(result.parse_error)
        self.assertEqual(result.correctness, 3)

    def test_malformed_judge_output_does_not_crash(self):
        result = parse_judge_output("not json", is_mock=False)
        self.assertIsNotNone(result.parse_error)

    def test_missing_field_does_not_crash(self):
        raw = '{"correctness": 3}'  # missing required fields
        result = parse_judge_output(raw, is_mock=False)
        self.assertIsNotNone(result.parse_error)

    def test_all_dimensions_defined(self):
        self.assertEqual(len(JUDGE_DIMENSIONS), 7)


if __name__ == "__main__":
    unittest.main()
