from pathlib import Path
import tempfile
import unittest

import numpy as np

from llm_oscilloscope import NativeQwenSupportHead, QwenSupportChannel


class NativeQwenSupportHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.head_path = (
            Path(__file__).parents[1]
            / "artifacts"
            / "support_v2"
            / "qwen_native_support_head.npz"
        )
        cls.trajectory_path = (
            Path(__file__).parents[1]
            / "artifacts"
            / "support_v2"
            / "trajectory_high_recall"
            / "profile.npz"
        )
        cls.head = NativeQwenSupportHead(cls.head_path)
        cls.channel = QwenSupportChannel(cls.head_path, cls.trajectory_path)

    def test_scores_vector_and_batch(self):
        vector = self.head.score(np.zeros(3584, dtype=np.float32))
        batch = self.head.score(np.zeros((2, 3584), dtype=np.float32))
        self.assertEqual(vector["support_margin"].shape, (1,))
        self.assertEqual(batch["unsupported_score"].shape, (2,))
        np.testing.assert_allclose(vector["support_margin"], [4.7716637], atol=1e-6)
        expected = 1.0 / (1.0 + np.exp(-vector["support_margin"]))
        np.testing.assert_allclose(vector["unsupported_score"], expected, atol=1e-7)

    def test_rejects_bad_input(self):
        with self.assertRaisesRegex(ValueError, "hidden width must be 3584"):
            self.head.score(np.zeros((1, 4096), dtype=np.float32))
        hidden = np.zeros((1, 3584), dtype=np.float32)
        hidden[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.head.score(hidden)

    def test_standard_profile_is_default_and_endpoint_only(self):
        rng = np.random.default_rng(20260819)
        endpoint = rng.normal(size=3584).astype(np.float32)
        short = np.stack([endpoint])
        long = np.stack(
            [rng.normal(size=3584).astype(np.float32), endpoint]
        )
        short_score = self.channel.score(short)
        long_score = self.channel.score(long, profile="standard")
        self.assertEqual(short_score["profile"], "standard")
        np.testing.assert_allclose(
            short_score["unsupported_margin"],
            long_score["unsupported_margin"],
            atol=1e-6,
        )
        self.assertEqual(short_score["alert"].dtype, np.bool_)

    def test_high_recall_profile_uses_variable_length_trajectories(self):
        rng = np.random.default_rng(20260820)
        trajectories = [
            rng.normal(size=(2, 3584)).astype(np.float32),
            rng.normal(size=(5, 3584)).astype(np.float32),
        ]
        scores = self.channel.score(trajectories, profile="high_recall")
        self.assertEqual(scores["profile"], "high_recall")
        self.assertEqual(scores["conditional_se_score"].shape, (2,))
        self.assertEqual(scores["unsupported_margin"].shape, (2,))
        self.assertEqual(scores["alert"].shape, (2,))
        self.assertTrue(np.all(np.isfinite(scores["unsupported_score"])))
        np.testing.assert_allclose(
            scores["base_support_margin"],
            [5.0648885, 4.5802121],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            scores["conditional_se_score"],
            [-0.42034724, -0.40197721],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            scores["unsupported_margin"],
            [3.6009095, 3.1687942],
            atol=1e-5,
        )

    def test_rectangular_trajectory_batch(self):
        batch = np.zeros((2, 3, 3584), dtype=np.float32)
        scores = self.channel.score(batch, profile="high_recall")
        self.assertEqual(scores["base_support_margin"].shape, (2,))

    def test_rejects_bad_profile_and_trajectory(self):
        trajectory = np.zeros((1, 3584), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Unknown support profile"):
            self.channel.score(trajectory, profile="automatic")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.channel.score([])
        with self.assertRaisesRegex(ValueError, "tokens,3584"):
            self.channel.score(np.zeros((2, 4096), dtype=np.float32))

    def test_rejects_profile_for_a_different_support_head(self):
        with np.load(self.trajectory_path, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        arrays["support_head_sha256"] = np.asarray(["0" * 64])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong_profile.npz"
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "head hashes differ"):
                QwenSupportChannel(self.head_path, path)


if __name__ == "__main__":
    unittest.main()
