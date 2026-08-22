"""Runtime composition for the Qwen research-preview measurement channels."""

from __future__ import annotations

import hashlib
import json
import os
import sysconfig
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from .qwen import TransportedQwenDetector
from .subject import TransportedSubjectRoutingHead
from .support_v2 import NativeQwenSupportHead


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cli_config() -> dict[str, Any]:
    resource = files("llm_oscilloscope").joinpath("cli_config.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def find_artifacts_root(explicit: str | Path | None = None) -> Path:
    """Locate the evidence package's artifact directory.

    An explicit path or ``LLM_OSCILLOSCOPE_ARTIFACTS`` wins. The repository
    layout is discovered next, followed by ``./artifacts`` for installed
    development checkouts.
    """

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    environment = os.environ.get("LLM_OSCILLOSCOPE_ARTIFACTS")
    if environment:
        candidates.append(Path(environment))
    candidates.extend(
        [
            Path(__file__).resolve().parents[2] / "artifacts",
            Path(__file__).resolve().parents[1]
            / "share"
            / "llm-oscilloscope"
            / "artifacts",
            Path(sysconfig.get_path("data"))
            / "share"
            / "llm-oscilloscope"
            / "artifacts",
            Path.cwd() / "artifacts",
        ]
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate the LLM-Oscilloscope artifacts directory. "
        f"Searched: {searched}. Set LLM_OSCILLOSCOPE_ARTIFACTS."
    )


@dataclass(frozen=True)
class ChannelReading:
    entity_end_score: float
    support_score: float
    combined_score: float
    entity_alert: bool
    combined_alert: bool
    subject_scores: dict[str, float]

    @property
    def top_subjects(self) -> list[tuple[str, float]]:
        return sorted(self.subject_scores.items(), key=lambda row: row[1], reverse=True)

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_end_score": self.entity_end_score,
            "support_score": self.support_score,
            "combined_score": self.combined_score,
            "entity_alert": self.entity_alert,
            "combined_alert": self.combined_alert,
            "subject_scores": self.subject_scores,
        }


class QwenResearchChannels:
    """Score the released Qwen channels with explicit research boundaries.

    Entity completion and Support V2 consume Qwen L23 post-token states.
    Subject routing consumes Qwen L27 post-token states. The combined score
    uses the frozen Support V2 Platt calibration, while its balanced threshold
    is a post-hoc CLI candidate and must not be presented as a released alarm.
    """

    def __init__(self, artifacts_root: str | Path | None = None):
        self.config = load_cli_config()
        self.artifacts_root = find_artifacts_root(artifacts_root)
        paths: dict[str, Path] = {}
        for name, specification in self.config["artifacts"].items():
            path = self.artifacts_root / specification["path"]
            if not path.is_file():
                raise FileNotFoundError(f"Missing {name} artifact: {path}")
            observed = _sha256(path)
            if observed != specification["sha256"]:
                raise ValueError(
                    f"{name} artifact hash changed: expected "
                    f"{specification['sha256']}, observed {observed}"
                )
            paths[name] = path

        self.entity = TransportedQwenDetector(paths["entity_transport"])
        self.support = NativeQwenSupportHead(paths["support_v2"])
        self.subject = TransportedSubjectRoutingHead(paths["subject_routing"])
        self.entity_threshold = float(
            self.config["thresholds"]["entity_balanced"]["value"]
        )
        self.combined_threshold = float(
            self.config["thresholds"]["combined_balanced"]["value"]
        )

    def score(self, hidden_l23_post: np.ndarray, hidden_l27_post: np.ndarray) -> ChannelReading:
        entity = self.entity.score(hidden_l23_post)
        support = self.support.score(hidden_l23_post)
        calibration = self.config["support_calibration"]
        support_score = float(
            _sigmoid(
                support["support_margin"] * float(calibration["coefficient"])
                + float(calibration["intercept"])
            )[0]
        )
        entity_score = float(entity["entity_score"][0])
        combined_score = entity_score * support_score
        subject_values = self.subject.score(hidden_l27_post)[0]
        subject_scores = {
            label: float(subject_values[index])
            for index, label in enumerate(self.subject.classes)
        }
        return ChannelReading(
            entity_end_score=entity_score,
            support_score=support_score,
            combined_score=combined_score,
            entity_alert=entity_score >= self.entity_threshold,
            combined_alert=combined_score >= self.combined_threshold,
            subject_scores=subject_scores,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "profile": self.config["profile"],
            "model_id": self.config["model_id"],
            "model_revision": self.config["model_revision"],
            "layers": dict(self.config["layers"]),
            "thresholds": dict(self.config["thresholds"]),
            "artifact_sha256": {
                name: specification["sha256"]
                for name, specification in self.config["artifacts"].items()
            },
            "evidence_boundary": dict(self.config["evidence_boundary"]),
        }
