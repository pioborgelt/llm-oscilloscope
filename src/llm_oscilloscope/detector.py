"""NumPy runtime for exported LLM-Oscilloscope linear heads."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


class DetectorHeads:
    """Load and evaluate L24-post boundary and L30-post correctness heads."""

    def __init__(self, path: str | Path):
        with np.load(Path(path)) as stored:
            self.arrays = {key: stored[key] for key in stored.files}
        if int(self.arrays["schema_version"][0]) != 1:
            raise ValueError("unsupported detector artifact schema")
        self.metadata = json.loads(str(self.arrays["metadata_json"].item()))

    def _probability(self, prefix: str, hidden: np.ndarray) -> np.ndarray:
        matrix = np.asarray(hidden, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        elif matrix.ndim != 2:
            raise ValueError(f"{prefix} hidden states must be a vector or matrix")
        expected_width = int(self.arrays[f"{prefix}_coef"].size)
        if matrix.shape[1] != expected_width:
            raise ValueError(
                f"{prefix} hidden width must be {expected_width}, got {matrix.shape[1]}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{prefix} hidden states contain non-finite values")
        if bool(self.arrays[f"{prefix}_unit_normalize"][0]):
            norms = np.linalg.norm(matrix, axis=1)
            norms[norms == 0] = 1.0
            matrix = matrix / norms[:, None]
        raw = matrix @ self.arrays[f"{prefix}_coef"].reshape(-1) + float(
            self.arrays[f"{prefix}_intercept"][0]
        )
        if not bool(self.arrays[f"{prefix}_calibrator_has_model"][0]):
            return np.full(
                len(matrix),
                float(self.arrays[f"{prefix}_calibrator_constant"][0]),
                dtype=np.float32,
            )
        calibrated = raw * float(
            self.arrays[f"{prefix}_calibrator_coef"].reshape(-1)[0]
        ) + float(self.arrays[f"{prefix}_calibrator_intercept"][0])
        return _sigmoid(calibrated).astype(np.float32)

    def boundary_probability(self, hidden_l24_post: np.ndarray) -> np.ndarray:
        return self._probability("boundary", hidden_l24_post)

    def unsupported_probability(self, hidden_l30_post: np.ndarray) -> np.ndarray:
        return self._probability("correctness", hidden_l30_post)

    def score(
        self,
        hidden_l24_post: np.ndarray,
        hidden_l30_post: np.ndarray,
    ) -> dict[str, np.ndarray]:
        boundary = self.boundary_probability(hidden_l24_post)
        unsupported = self.unsupported_probability(hidden_l30_post)
        if len(boundary) != len(unsupported):
            raise ValueError("boundary and correctness batches differ")
        return {
            "entity_end_probability": boundary,
            "unsupported_probability": unsupported,
            "hallucinated_entity_score": boundary * unsupported,
        }
