from pathlib import Path
import unittest

import numpy as np

from llm_oscilloscope import SubjectRoutingHead, TransportedSubjectRoutingHead


class SubjectRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = Path(__file__).parents[1] / "artifacts" / "subject_routing"
        cls.llama = SubjectRoutingHead(directory / "llama_head.npz")
        cls.mistral = TransportedSubjectRoutingHead(directory / "mistral_adapter.npz")
        cls.qwen = TransportedSubjectRoutingHead(directory / "qwen_adapter.npz")

    def test_native_head_scores_batches(self):
        probabilities = self.llama.score(np.zeros((2, 4096), dtype=np.float32))
        self.assertEqual(probabilities.shape, (2, 8))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_transported_heads_score_batches(self):
        mistral = self.mistral.score(np.zeros((2, 4096), dtype=np.float32))
        qwen = self.qwen.score(np.zeros((2, 3584), dtype=np.float32))
        self.assertEqual(mistral.shape, (2, 8))
        self.assertEqual(qwen.shape, (2, 8))
        np.testing.assert_allclose(mistral.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(qwen.sum(axis=1), 1.0, atol=1e-6)

    def test_portable_head_regression_outputs(self):
        expected = {
            "llama": np.asarray(
                [
                    0.18550302,
                    0.03909865,
                    0.43273959,
                    0.03388070,
                    0.10923902,
                    0.16669865,
                    0.02255291,
                    0.01028746,
                ]
            ),
            "mistral": np.asarray(
                [
                    0.11126705,
                    0.02119408,
                    0.01439954,
                    0.01833623,
                    0.03900608,
                    0.09743777,
                    0.02675335,
                    0.67160589,
                ]
            ),
            "qwen": np.asarray(
                [
                    0.35061905,
                    0.00519322,
                    0.06628334,
                    0.20431864,
                    0.04857214,
                    0.03568552,
                    0.18468168,
                    0.10464641,
                ]
            ),
        }
        observed = {
            "llama": self.llama.score(np.zeros(4096, dtype=np.float32))[0],
            "mistral": self.mistral.score(np.zeros(4096, dtype=np.float32))[0],
            "qwen": self.qwen.score(np.zeros(3584, dtype=np.float32))[0],
        }
        for name in expected:
            np.testing.assert_allclose(observed[name], expected[name], atol=1e-7)

    def test_top_subject_returns_label_and_probability(self):
        result = self.llama.top_subject(np.zeros(4096, dtype=np.float32))
        self.assertEqual(len(result), 1)
        self.assertIn(result[0][0], self.llama.classes)
        self.assertGreaterEqual(result[0][1], 0.0)
        self.assertLessEqual(result[0][1], 1.0)

    def test_rejects_wrong_hidden_width(self):
        with self.assertRaisesRegex(ValueError, "hidden width must be 4096"):
            self.llama.score(np.zeros((1, 32), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "hidden width must be 3584"):
            self.qwen.score(np.zeros((1, 4096), dtype=np.float32))

    def test_rejects_non_finite_input(self):
        hidden = np.zeros((1, 4096), dtype=np.float32)
        hidden[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.llama.score(hidden)


if __name__ == "__main__":
    unittest.main()
