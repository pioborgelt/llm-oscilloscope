from pathlib import Path
import unittest

import numpy as np

from llm_oscilloscope import TransportedQwenDetector


class TransportedQwenDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = (
            Path(__file__).parents[1]
            / "artifacts"
            / "qwen_native_transfer"
            / "adapter.npz"
        )
        cls.detector = TransportedQwenDetector(path)

    def test_scores_vector_and_batch(self):
        vector = self.detector.score(np.zeros(3584, dtype=np.float32))
        batch = self.detector.score(np.zeros((2, 3584), dtype=np.float32))
        self.assertEqual(vector["combined_score"].shape, (1,))
        self.assertEqual(batch["combined_score"].shape, (2,))
        np.testing.assert_allclose(
            batch["combined_score"],
            batch["entity_score"] * batch["unsupported_score"],
        )
        np.testing.assert_allclose(vector["entity_margin"], [-2.7816782], atol=1e-6)
        np.testing.assert_allclose(vector["support_margin"], [0.6333790], atol=1e-6)

    def test_rejects_bad_input(self):
        with self.assertRaisesRegex(ValueError, "hidden width must be 3584"):
            self.detector.score(np.zeros((1, 4096), dtype=np.float32))
        hidden = np.zeros((1, 3584), dtype=np.float32)
        hidden[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.detector.score(hidden)


if __name__ == "__main__":
    unittest.main()
