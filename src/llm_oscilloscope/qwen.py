"""NumPy runtime for the transported Qwen detector channels."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class TransportedQwenDetector:
    """Read both frozen Llama channels from Qwen2.5-7B L23 post-token states."""

    def __init__(self, path: str | Path):
        with np.load(Path(path), allow_pickle=False) as stored:
            self.arrays = {key: stored[key] for key in stored.files}

    def _reconstruct(self, hidden_l23_post: np.ndarray) -> np.ndarray:
        values = np.asarray(hidden_l23_post, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        elif values.ndim != 2:
            raise ValueError("Qwen hidden states must be a vector or matrix")
        if values.shape[1] != 3584:
            raise ValueError(f"Qwen hidden width must be 3584, got {values.shape[1]}")
        if not np.isfinite(values).all():
            raise ValueError("Qwen hidden states contain non-finite values")

        scaled = (values - self.arrays["target_scaler_mean"]) / self.arrays[
            "target_scaler_scale"
        ]
        coordinates = (
            (scaled - self.arrays["target_pca_mean"])
            @ self.arrays["target_pca_components"].T
        ) / np.sqrt(self.arrays["target_pca_variance"])
        mapped = (
            coordinates @ self.arrays["ridge_coef"].T
            + self.arrays["ridge_intercept"]
        )
        source_scaled = (
            mapped * np.sqrt(self.arrays["source_pca_variance"])
        ) @ self.arrays["source_pca_components"] + self.arrays["source_pca_mean"]
        return (
            source_scaled * self.arrays["source_scaler_scale"]
            + self.arrays["source_scaler_mean"]
        )

    def _margin(self, reconstructed: np.ndarray, channel: str) -> np.ndarray:
        values = reconstructed
        if bool(self.arrays[f"{channel}_unit_normalize"][0]):
            norms = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
            values = values / norms
        return (
            values @ self.arrays[f"{channel}_coef"]
            + float(self.arrays[f"{channel}_intercept"][0])
        )

    def score(self, hidden_l23_post: np.ndarray) -> dict[str, np.ndarray]:
        reconstructed = self._reconstruct(hidden_l23_post)
        entity_margin = self._margin(reconstructed, "entity")
        support_margin = self._margin(reconstructed, "support")
        entity = _sigmoid(entity_margin).astype(np.float32)
        unsupported = _sigmoid(support_margin).astype(np.float32)
        return {
            "entity_margin": entity_margin.astype(np.float32),
            "support_margin": support_margin.astype(np.float32),
            "entity_score": entity,
            "unsupported_score": unsupported,
            "combined_score": entity * unsupported,
        }
