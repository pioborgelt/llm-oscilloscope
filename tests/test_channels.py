from pathlib import Path
import unittest

import numpy as np

from llm_oscilloscope.channels import QwenResearchChannels, load_cli_config


class QwenResearchChannelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).parents[1] / "artifacts"
        cls.channels = QwenResearchChannels(cls.root)

    def test_config_preserves_negative_release_boundary(self):
        config = load_cli_config()
        self.assertEqual(
            config["evidence_boundary"]["combined_release_status"],
            "NO_SUPPORT_V2_COMBINED_ALERT_RELEASE",
        )
        self.assertEqual(
            config["thresholds"]["entity_balanced"]["status"],
            "post_hoc_cli_candidate",
        )
        self.assertFalse(config["evidence_boundary"]["scores_are_probabilities"])

    def test_all_channels_score_selected_states(self):
        reading = self.channels.score(
            np.zeros(3584, dtype=np.float32),
            np.zeros(3584, dtype=np.float32),
        )
        self.assertGreaterEqual(reading.entity_end_score, 0.0)
        self.assertLessEqual(reading.entity_end_score, 1.0)
        self.assertGreaterEqual(reading.support_score, 0.0)
        self.assertLessEqual(reading.support_score, 1.0)
        self.assertAlmostEqual(
            reading.combined_score,
            reading.entity_end_score * reading.support_score,
            places=7,
        )
        self.assertEqual(len(reading.subject_scores), 8)
        self.assertAlmostEqual(sum(reading.subject_scores.values()), 1.0, places=6)

    def test_manifest_binds_layers_and_artifacts(self):
        manifest = self.channels.manifest()
        self.assertEqual(manifest["layers"], {"detector": 23, "subject_routing": 27})
        self.assertEqual(len(manifest["artifact_sha256"]), 3)


if __name__ == "__main__":
    unittest.main()
