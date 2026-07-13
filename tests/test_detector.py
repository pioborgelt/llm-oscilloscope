from pathlib import Path
import unittest

import numpy as np

from llm_oscilloscope import DetectorHeads


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[1] / "artifacts" / "weights" / "release_head.npz"
        cls.heads = DetectorHeads(path)

    def test_release_head_scores_batches(self):
        hidden = np.zeros((2, 4096), dtype=np.float32)
        scores = self.heads.score(hidden, hidden)
        self.assertEqual(scores["hallucinated_entity_score"].shape, (2,))
        self.assertTrue(np.all(scores["hallucinated_entity_score"] >= 0))
        self.assertTrue(np.all(scores["hallucinated_entity_score"] <= 1))

    def test_release_head_scores_single_vectors(self):
        hidden = np.zeros(4096, dtype=np.float32)
        scores = self.heads.score(hidden, hidden)
        self.assertEqual(scores["hallucinated_entity_score"].shape, (1,))

    def test_rejects_wrong_hidden_width(self):
        with self.assertRaisesRegex(ValueError, "hidden width must be 4096"):
            self.heads.boundary_probability(np.zeros((1, 32), dtype=np.float32))

    def test_rejects_higher_rank_input(self):
        with self.assertRaisesRegex(ValueError, "must be a vector or matrix"):
            self.heads.unsupported_probability(np.zeros((1, 1, 4096), dtype=np.float32))

    def test_rejects_non_finite_input(self):
        hidden = np.zeros((1, 4096), dtype=np.float32)
        hidden[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.heads.boundary_probability(hidden)

    def test_rejects_mismatched_batches(self):
        with self.assertRaisesRegex(ValueError, "batches differ"):
            self.heads.score(
                np.zeros((1, 4096), dtype=np.float32),
                np.zeros((2, 4096), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
