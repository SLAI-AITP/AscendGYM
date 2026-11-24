import unittest

from eval.native_harness.core.metrics import sts


class StsTests(unittest.TestCase):
    def test_scores_only_best_passed_improvements_within_budget(self):
        rows = [
            {"status": "passed", "cumulative_tokens": 110, "duration_ms": 0.5},
            {"status": "passed", "cumulative_tokens": 50, "duration_ms": 2},
            {"status": "passed", "cumulative_tokens": 20, "duration_ms": 4},
            {"status": "passed", "cumulative_tokens": 50, "duration_ms": 3},
            {"status": "passed", "cumulative_tokens": 60, "duration_ms": 3},
            {"status": "test_failed", "cumulative_tokens": 75, "duration_ms": None},
            {"status": "passed", "cumulative_tokens": 100, "duration_ms": 1},
        ]

        result = sts(rows, 8, [10, 25, 50, 100], "cumulative_tokens")

        self.assertEqual(result["status"], "observed")
        self.assertEqual(
            [(event["budget"], event["best_duration_ms"]) for event in result["improvements"]],
            [(20, 4), (50, 2), (100, 1)],
        )
        self.assertEqual(
            [point["speedup"] for point in result["curve"]],
            [1, 2, 4, 8],
        )
        self.assertAlmostEqual(result["base_score"], 300)
        self.assertAlmostEqual(result["delay_penalty"], 85)
        self.assertAlmostEqual(result["STS"], 215)
        self.assertEqual(result["S_at_Bmax"], 8)

    def test_interior_checkpoints_do_not_change_score(self):
        rows = [{"status": "passed", "cumulative_tokens": 90, "duration_ms": 4}]

        for checkpoints in ([100], [20, 80, 100], [20, 90, 100]):
            with self.subTest(checkpoints=checkpoints):
                result = sts(rows, 8, checkpoints, "cumulative_tokens")
                self.assertAlmostEqual(result["STS"], 55)

    def test_improvement_at_zero_has_no_delay_penalty(self):
        rows = [{"status": "passed", "cumulative_tokens": 0, "duration_ms": 4}]

        result = sts(rows, 8, [100], "cumulative_tokens")

        self.assertEqual(result["delay_penalty"], 0)
        self.assertAlmostEqual(result["STS"], 100)

    def test_missing_token_count_makes_score_unavailable(self):
        rows = [
            {"status": "passed", "cumulative_tokens": 20, "duration_ms": 4},
            {"status": "test_failed", "duration_ms": None},
        ]

        result = sts(rows, 8, [100], "cumulative_tokens")

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reason"], "incomplete cumulative_tokens coverage")
