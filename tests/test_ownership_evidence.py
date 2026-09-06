from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ownership_evidence_verifier", ROOT / "scripts/verify_ownership.py"
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class OwnershipEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = json.loads(
            (ROOT / "artifacts/sycophancy_ownership/evidence.json").read_text()
        )

    def test_all_released_events_and_bootstraps_recompute(self):
        result = verifier.verify(self.evidence)
        self.assertEqual(set(result), {"smollm3", "qwen3"})

    def test_changed_headline_is_detected(self):
        altered = deepcopy(self.evidence)
        altered["models"][0]["frozen_summary"]["summaries"]["opinion"][
            "ownership_shift_pp"
        ]["mean"] += 1
        with self.assertRaises(AssertionError):
            verifier.verify(altered)

    def test_missing_case_and_changed_model_revision_are_detected(self):
        altered = deepcopy(self.evidence)
        altered["models"][0]["recordings"].pop()
        with self.assertRaises(AssertionError):
            verifier.verify(altered)
        altered = deepcopy(self.evidence)
        altered["models"][0]["recordings"][0]["runtime"]["model_revision"] = (
            "other-revision"
        )
        with self.assertRaises(AssertionError):
            verifier.verify(altered)

    def test_factual_labels_are_checked_from_text(self):
        altered = deepcopy(self.evidence)
        record = next(
            r
            for r in altered["models"][0]["recordings"]
            if r["research_case"]["id"] == 2001
        )
        record["research_case"]["gold"] = "B"
        with self.assertRaises(AssertionError):
            verifier.verify(altered)


if __name__ == "__main__":
    unittest.main()
