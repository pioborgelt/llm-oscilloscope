"""Portable target-labeled Qwen support-checking readouts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NativeQwenSupportHead:
    """Rank unsupported answer endpoints from Qwen L23 post-token states.

    This head was fitted with target-model support labels. Its sigmoid output is
    convenient but is not a deployment-calibrated probability.
    """

    def __init__(self, path: str | Path):
        with np.load(Path(path), allow_pickle=False) as stored:
            self.arrays = {key: stored[key] for key in stored.files}
        required = {
            "coef",
            "intercept",
            "alpha",
            "unit_normalize",
            "target_layer",
        }
        missing = required - self.arrays.keys()
        if missing:
            raise ValueError(f"Support V2 artifact is missing {sorted(missing)}")
        if self.arrays["coef"].shape != (3584,):
            raise ValueError("Support V2 coefficient width must be 3584")
        if int(self.arrays["target_layer"][0]) != 23:
            raise ValueError("Support V2 artifact must target Qwen L23")
        if not bool(self.arrays["unit_normalize"][0]):
            raise ValueError("Support V2 artifact must use unit-normalized states")
        if not np.all(np.isfinite(self.arrays["coef"])):
            raise ValueError("Support V2 coefficients contain non-finite values")

    def score(self, hidden_l23_post: np.ndarray) -> dict[str, np.ndarray]:
        values = np.asarray(hidden_l23_post, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        elif values.ndim != 2:
            raise ValueError("Qwen hidden states must be a vector or matrix")
        if values.shape[1] != 3584:
            raise ValueError(f"Qwen hidden width must be 3584, got {values.shape[1]}")
        if not np.all(np.isfinite(values)):
            raise ValueError("Qwen hidden states contain non-finite values")
        norms = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
        normalized = values / norms
        margin = normalized @ self.arrays["coef"] + float(self.arrays["intercept"][0])
        return {
            "support_margin": margin.astype(np.float32),
            "unsupported_score": _sigmoid(margin).astype(np.float32),
        }


class _ConditionalSETrajectory:
    """Frozen trajectory feature and its two operating profiles."""

    _REQUIRED = {
        "schema_version",
        "target_layer",
        "hidden_width",
        "support_mean",
        "support_scale",
        "feature_mean",
        "support_to_feature",
        "student_coef",
        "student_intercept",
        "meta_mean",
        "meta_scale",
        "meta_coef",
        "meta_intercept",
        "threshold",
        "fast_mean",
        "fast_scale",
        "fast_coef",
        "fast_intercept",
        "fast_threshold",
        "support_head_sha256",
    }

    def __init__(self, path: str | Path):
        with np.load(Path(path), allow_pickle=False) as stored:
            self.arrays = {key: stored[key] for key in stored.files}
        missing = self._REQUIRED - self.arrays.keys()
        if missing:
            raise ValueError(
                f"Support trajectory artifact is missing {sorted(missing)}"
            )
        if int(self.arrays["schema_version"][0]) != 1:
            raise ValueError("Unsupported support trajectory artifact schema")
        if int(self.arrays["target_layer"][0]) != 23:
            raise ValueError("Support trajectory artifact must target Qwen L23")
        self.width = int(self.arrays["hidden_width"][0])
        vector_width = 2 * self.width
        expected_shapes = {
            "feature_mean": (vector_width,),
            "support_to_feature": (vector_width,),
            "student_coef": (vector_width,),
            "meta_mean": (2,),
            "meta_scale": (2,),
            "meta_coef": (2,),
        }
        for name, shape in expected_shapes.items():
            if self.arrays[name].shape != shape:
                raise ValueError(
                    f"Support trajectory {name} must have shape {shape}"
                )
        for name in self._REQUIRED - {
            "schema_version",
            "target_layer",
            "hidden_width",
            "support_head_sha256",
        }:
            if not np.all(np.isfinite(self.arrays[name])):
                raise ValueError(f"Support trajectory {name} contains non-finite values")
        if np.any(self.arrays["meta_scale"] <= 0):
            raise ValueError("Support trajectory meta scale must be positive")
        if float(self.arrays["support_scale"][0]) <= 0:
            raise ValueError("Support trajectory support scale must be positive")
        if float(self.arrays["fast_scale"][0]) <= 0:
            raise ValueError("Support trajectory fast scale must be positive")

    def validate_support_head(self, path: Path) -> None:
        expected = str(self.arrays["support_head_sha256"][0])
        if _sha256(path) != expected:
            raise ValueError("Support trajectory and native support head hashes differ")

    def normalize_trajectories(
        self, token_states: Sequence[np.ndarray] | np.ndarray
    ) -> list[np.ndarray]:
        if isinstance(token_states, np.ndarray):
            if token_states.ndim == 2:
                values = [token_states]
            elif token_states.ndim == 3:
                values = list(token_states)
            else:
                raise ValueError(
                    "Qwen trajectories must be [tokens,width] or [batch,tokens,width]"
                )
        else:
            values = list(token_states)
        if not values:
            raise ValueError("Qwen trajectory batch must not be empty")
        normalized = []
        for states in values:
            matrix = np.asarray(states, dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[1] != self.width or len(matrix) == 0:
                raise ValueError(
                    f"Each Qwen trajectory must be nonempty [tokens,{self.width}]"
                )
            if not np.all(np.isfinite(matrix)):
                raise ValueError("Qwen trajectories contain non-finite values")
            normalized.append(matrix)
        return normalized

    def _feature(self, token_states: list[np.ndarray]) -> np.ndarray:
        rows = []
        for values in token_states:
            states = values.astype(np.float64, copy=False)
            states = states / np.maximum(
                np.linalg.norm(states, axis=1, keepdims=True), 1e-12
            )
            mean = states.mean(axis=0)
            mean /= max(float(np.linalg.norm(mean)), 1e-12)
            if len(states) == 1:
                innovation = np.zeros(self.width, dtype=np.float64)
            else:
                innovation = states[-1] - states[:-1].mean(axis=0)
                innovation /= max(float(np.linalg.norm(innovation)), 1e-12)
            rows.append(np.concatenate([mean, innovation]) / np.sqrt(2.0))
        return np.asarray(rows)

    def score_standard(self, support_margin: np.ndarray) -> dict[str, np.ndarray]:
        support = np.asarray(support_margin, dtype=np.float64)
        margin = (
            (support - float(self.arrays["fast_mean"][0]))
            / float(self.arrays["fast_scale"][0])
            * float(self.arrays["fast_coef"][0])
            + float(self.arrays["fast_intercept"][0])
        )
        return {
            "unsupported_margin": margin.astype(np.float32),
            "unsupported_score": _sigmoid(margin).astype(np.float32),
            "alert": margin >= float(self.arrays["fast_threshold"][0]),
        }

    def score_high_recall(
        self,
        token_states: list[np.ndarray],
        support_margin: np.ndarray,
    ) -> dict[str, np.ndarray]:
        feature = self._feature(token_states)
        support = np.asarray(support_margin, dtype=np.float64)
        z = (
            support - float(self.arrays["support_mean"][0])
        ) / float(self.arrays["support_scale"][0])
        residual = (
            feature
            - self.arrays["feature_mean"]
            - np.outer(z, self.arrays["support_to_feature"])
        )
        student = (
            residual @ self.arrays["student_coef"]
            + float(self.arrays["student_intercept"][0])
        )
        meta = np.column_stack([support, student])
        margin = (
            (meta - self.arrays["meta_mean"]) / self.arrays["meta_scale"]
        ) @ self.arrays["meta_coef"] + float(self.arrays["meta_intercept"][0])
        return {
            "conditional_se_score": student.astype(np.float32),
            "unsupported_margin": margin.astype(np.float32),
            "unsupported_score": _sigmoid(margin).astype(np.float32),
            "alert": margin >= float(self.arrays["threshold"][0]),
        }


class QwenSupportChannel:
    """Score Qwen answer endpoints with an explicit operating profile.

    ``standard`` uses only the final L23 post-token state. ``high_recall`` adds
    the frozen prefix-mean and endpoint-innovation readout. The latter improved
    recall on two hard factual-QA gates but also increased false alarms, so it
    is opt-in and remains a research operating profile.
    """

    PROFILES = ("standard", "high_recall")

    def __init__(
        self,
        support_head_path: str | Path,
        trajectory_profile_path: str | Path,
    ):
        head_path = Path(support_head_path)
        self.endpoint_head = NativeQwenSupportHead(head_path)
        self.trajectory = _ConditionalSETrajectory(trajectory_profile_path)
        self.trajectory.validate_support_head(head_path)

    def score(
        self,
        token_states: Sequence[np.ndarray] | np.ndarray,
        *,
        profile: str = "standard",
    ) -> dict[str, np.ndarray | str]:
        if profile not in self.PROFILES:
            raise ValueError(
                f"Unknown support profile {profile!r}; choose one of {self.PROFILES}"
            )
        trajectories = self.trajectory.normalize_trajectories(token_states)
        endpoints = np.stack([states[-1] for states in trajectories])
        endpoint = self.endpoint_head.score(endpoints)
        support_margin = endpoint["support_margin"]
        if profile == "standard":
            result = self.trajectory.score_standard(support_margin)
        else:
            result = self.trajectory.score_high_recall(trajectories, support_margin)
        return {
            "profile": profile,
            "base_support_margin": support_margin,
            "base_unsupported_score": endpoint["unsupported_score"],
            **result,
        }
