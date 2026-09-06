from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ownership_control_verifier", ROOT / "scripts/verify_ownership_controls.py"
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class OwnershipControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primary = json.loads(
            (ROOT / "artifacts/sycophancy_ownership/evidence.json").read_text()
        )
        cls.controls = {
            kind: json.loads(
                (ROOT / "artifacts/sycophancy_ownership" / (kind + ".json")).read_text()
            )
            for kind in ("wording_stress", "quoted_identity")
        }

    def test_all_controls_recompute_and_failed_point_stays_failed(self):
        for control in self.controls.values():
            verifier.verify_control(control, self.primary)
        result = verifier.verify_control(self.controls["wording_stress"], self.primary)
        self.assertFalse(result["smollm3"]["requester_view"]["descriptive_point_pass"])

    def test_relabeling_the_failed_result_is_rejected(self):
        value = deepcopy(self.controls["wording_stress"])
        value["models"][0]["research_result"]["variants"]["requester_view"][
            "descriptive_robustness_point_pass"
        ] = True
        with self.assertRaises(AssertionError):
            verifier.verify_control(value, self.primary)

    def test_changed_prompt_or_missing_event_is_rejected(self):
        value = deepcopy(self.controls["quoted_identity"])
        value["metadata"]["quoted_identity"]["user_a"] += " Modified."
        with self.assertRaises(AssertionError):
            verifier.verify_control(value, self.primary)
        value = deepcopy(self.controls["quoted_identity"])
        value["models"][0]["new_events"].pop()
        with self.assertRaises(AssertionError):
            verifier.verify_control(value, self.primary)

    def test_factual_diagnostic_preserves_all_no_go_and_unknowns(self):
        control = json.loads(
            (
                ROOT / "artifacts/sycophancy_ownership/factual_task_diagnostic.json"
            ).read_text()
        )
        result = verifier.verify_factual(control, self.primary)
        self.assertEqual(result["smollm3"]["single_record"]["unknown"], 14)
        self.assertTrue(
            all(
                not v["descriptive_competence_go"]
                for m in result.values()
                for v in m.values()
            )
        )
        wrong = deepcopy(control)
        wrong["models"][0]["research_result"]["variants"]["single_record"][
            "correct"
        ] += 1
        with self.assertRaises(AssertionError):
            verifier.verify_factual(wrong, self.primary)
        incomplete = deepcopy(control)
        incomplete["models"][0]["new_events"].pop()
        with self.assertRaises(AssertionError):
            verifier.verify_factual(incomplete, self.primary)


if __name__ == "__main__":
    unittest.main()
