from copy import deepcopy
import math
import unittest

from llm_oscilloscope.ownership import (
    ARMS,
    ORDERS,
    OWNERSHIP,
    ChoiceReading,
    OwnershipCase,
    build_messages,
    parse_answer,
    replay_assay,
    run_assay,
    summarize_readings,
)


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.case = OwnershipCase(
            "We favor a fixed meeting day.", "We favor rotating meeting days."
        )

    def test_only_ownership_varies_within_order(self):
        for order in ORDERS:
            common = [
                build_messages(self.case, arm, order)[1]["content"].replace(
                    OWNERSHIP[arm], "<metadata>"
                )
                for arm in ARMS
            ]
            self.assertEqual(len(set(common)), 1)
        for arm in ARMS[1:]:
            self.assertLess(
                OWNERSHIP[arm].index("Side A"), OWNERSHIP[arm].index("Side B")
            )

    def test_null_invariance_and_exact_recording_replay(self):
        result = run_assay(
            self.case,
            lambda _: ChoiceReading(0.0, 0.0, 1.0, "A", True),
            runtime={"model_id": "synthetic-test-only"},
        )
        self.assertEqual(result["measurement"]["ownership_shift_pp"], 0.0)
        self.assertEqual(result["measurement"], replay_assay(result))
        self.assertIsNone(result["measurement"]["harm_alarm"])

    def test_known_effect_and_negative_effect_are_not_clipped(self):
        base = {
            (arm, order): ChoiceReading(0.0, 0.0, 1.0, "A", True)
            for arm in ARMS
            for order in ORDERS
        }
        for order in ORDERS:
            base["user_a", order] = ChoiceReading(math.log(4), 0.0, 1.0, "A", True)
            base["user_b", order] = ChoiceReading(-math.log(4), 0.0, 1.0, "B", True)
        result = summarize_readings(base)
        self.assertAlmostEqual(result["ownership_shift_pp"], 60.0)
        self.assertEqual(result["natural_owner_switch_pairs"], 2)
        inverted = {
            key: ChoiceReading(
                -value.a_logit, -value.b_logit, 1.0, value.output_text, True
            )
            for key, value in base.items()
        }
        self.assertAlmostEqual(
            summarize_readings(inverted)["ownership_shift_pp"], -60.0
        )

    def test_unknown_answers_and_low_mass_are_explicit(self):
        result = run_assay(
            self.case,
            lambda _: ChoiceReading(1000.0, -1000.0, 0.02, "A", False),
            runtime={"model_id": "synthetic-test-only"},
        )
        self.assertEqual(result["measurement"]["valid_answers"], 0)
        self.assertEqual(result["measurement"]["quality"]["status"], "off_protocol")
        self.assertEqual(len(result["measurement"]["quality"]["flags"]), 2)

    def test_case_and_number_validation(self):
        for a, b in (("", "x"), ("x", "x"), (12, "x"), ("x" * 12001, "y")):
            with self.assertRaises(ValueError):
                OwnershipCase(a, b)
        for value in (float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                ChoiceReading(value, 0.0, 1.0, "A", True)
        with self.assertRaises(ValueError):
            ChoiceReading(1e308, -1e308, 1.0, "A", True)
        for mass in (-0.1, 1.01):
            with self.assertRaises(ValueError):
                ChoiceReading(0.0, 0.0, mass, "A", True)

    def test_parser_requires_actual_complete_letter(self):
        self.assertEqual(parse_answer(" B. ", True), "B")
        for text, eos in (
            ("A", False),
            ("Answer: A", True),
            ("A or B", True),
            ("", True),
        ):
            self.assertIsNone(parse_answer(text, eos))

    def test_missing_duplicate_and_tampered_recordings_rejected(self):
        good = run_assay(
            self.case,
            lambda _: ChoiceReading(0.0, 0.0, 1.0, "A", True),
            runtime={"model_id": "synthetic-test-only"},
        )
        bad = deepcopy(good)
        bad["events"] = bad["events"][:-1]
        with self.assertRaises(ValueError):
            replay_assay(bad)
        bad = deepcopy(good)
        bad["events"][1] = bad["events"][0]
        with self.assertRaises(ValueError):
            replay_assay(bad)
        bad = deepcopy(good)
        bad["case"]["side_a"] += " modified"
        with self.assertRaises(ValueError):
            replay_assay(bad)
        bad = deepcopy(good)
        bad["events"][0]["message_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            replay_assay(bad)


if __name__ == "__main__":
    unittest.main()
