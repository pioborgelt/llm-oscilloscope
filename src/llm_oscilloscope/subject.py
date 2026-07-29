"""Portable subject-routing heads for Llama, Mistral and Qwen activations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _rows(values: np.ndarray, width: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    elif array.ndim != 2:
        raise ValueError("hidden states must be a vector or matrix")
    if array.shape[1] != width:
        raise ValueError(f"hidden width must be {width}")
    if not np.isfinite(array).all():
        raise ValueError("hidden states contain non-finite values")
    return array


def _normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values -= values.max(axis=1, keepdims=True)
    exponentiated = np.exp(values)
    return (exponentiated / exponentiated.sum(axis=1, keepdims=True)).astype(
        np.float32
    )


def _metadata(stored: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata" not in stored.files:
        return {}
    return json.loads(str(stored["metadata"][0]))


class SubjectRoutingHead:
    """Read the eight-way Llama subject channel from one residual state."""

    def __init__(self, path: str | Path):
        with np.load(Path(path), allow_pickle=False) as stored:
            self.layer = int(stored["layer"][0])
            self.classes = tuple(str(value) for value in stored["classes"])
            self.center = stored["center"].astype(np.float32)
            self.basis = stored["basis"].astype(np.float32)
            self.coef = stored["coef"].astype(np.float32)
            self.intercept = stored["intercept"].astype(np.float32)
            self.temperature = float(stored["temperature"][0])
            self.metadata = _metadata(stored)
            unit_normalize = int(stored["unit_normalize"][0])

        width = len(self.center)
        rank = self.basis.shape[1]
        if unit_normalize != 1:
            raise ValueError("subject head does not specify unit normalization")
        if self.basis.shape != (width, rank):
            raise ValueError("invalid subject basis")
        if self.coef.shape != (len(self.classes), rank):
            raise ValueError("invalid subject classifier")
        if self.intercept.shape != (len(self.classes),):
            raise ValueError("invalid subject intercept")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("invalid subject temperature")
        self.hidden_width = width

    def score(self, hidden: np.ndarray) -> np.ndarray:
        values = _normalize(_rows(hidden, self.hidden_width))
        coordinates = (values - self.center) @ self.basis
        logits = coordinates @ self.coef.T + self.intercept
        return _softmax(logits / self.temperature)

    def top_subject(self, hidden: np.ndarray) -> list[tuple[str, float]]:
        probabilities = self.score(hidden)
        indices = probabilities.argmax(axis=1)
        return [
            (self.classes[int(index)], float(probabilities[row, index]))
            for row, index in enumerate(indices)
        ]


class TransportedSubjectRoutingHead:
    """Apply the frozen Llama subject channel to an aligned target model."""

    def __init__(self, path: str | Path):
        with np.load(Path(path), allow_pickle=False) as stored:
            self.target_layer = int(stored["target_layer"][0])
            self.classes = tuple(str(value) for value in stored["classes"])
            self.pca_mean = stored["pca_mean"].astype(np.float32)
            self.pca_components = stored["pca_components"].astype(np.float32)
            self.pca_scale = stored["pca_scale"].astype(np.float32)
            self.ridge_coef = stored["ridge_coef"].astype(np.float32)
            self.ridge_intercept = stored["ridge_intercept"].astype(np.float32)
            self.target_mean = stored["target_mean"].astype(np.float32)
            self.target_scale = stored["target_scale"].astype(np.float32)
            self.classifier_coef = stored["classifier_coef"].astype(np.float32)
            self.classifier_intercept = stored["classifier_intercept"].astype(
                np.float32
            )
            self.temperature = float(stored["classifier_temperature"][0])
            self.metadata = _metadata(stored)

        width = len(self.pca_mean)
        rank = self.pca_components.shape[0]
        coordinate_width = self.ridge_coef.shape[0]
        if self.pca_components.shape != (rank, width):
            raise ValueError("invalid subject PCA")
        if self.pca_scale.shape != (rank,) or np.any(self.pca_scale == 0):
            raise ValueError("invalid subject PCA scale")
        if self.ridge_coef.shape[1] != rank:
            raise ValueError("invalid subject adapter")
        if self.ridge_intercept.shape != (coordinate_width,):
            raise ValueError("invalid subject adapter intercept")
        if self.target_mean.shape != (coordinate_width,):
            raise ValueError("invalid subject target mean")
        if self.target_scale.shape != (coordinate_width,):
            raise ValueError("invalid subject target scale")
        if self.classifier_coef.shape != (len(self.classes), coordinate_width):
            raise ValueError("invalid transported subject classifier")
        if self.classifier_intercept.shape != (len(self.classes),):
            raise ValueError("invalid transported subject intercept")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("invalid transported subject temperature")
        self.hidden_width = width

    def score(self, hidden: np.ndarray) -> np.ndarray:
        values = _normalize(_rows(hidden, self.hidden_width))
        compressed = (
            (values - self.pca_mean) @ self.pca_components.T
        ) / self.pca_scale
        standardized = compressed @ self.ridge_coef.T + self.ridge_intercept
        coordinates = standardized * self.target_scale + self.target_mean
        logits = coordinates @ self.classifier_coef.T + self.classifier_intercept
        return _softmax(logits / self.temperature)

    def top_subject(self, hidden: np.ndarray) -> list[tuple[str, float]]:
        probabilities = self.score(hidden)
        indices = probabilities.argmax(axis=1)
        return [
            (self.classes[int(index)], float(probabilities[row, index]))
            for row, index in enumerate(indices)
        ]
