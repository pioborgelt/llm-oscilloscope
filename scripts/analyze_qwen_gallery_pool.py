#!/usr/bin/env python3
"""Apply the current CLI heads to the frozen Qwen-native evaluation pool.

Selection is deliberately rank-based rather than score-based. Within the
pre-existing frozen prompt order, the report takes the first three detected
unsupported answer endpoints and first two quiet supported endpoints. The
external entity grader was completed before this CLI gallery analysis and never
saw detector scores. Three known false-positive examples are retained from the
existing gallery rather than requiring this pool to produce them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from llm_oscilloscope.channels import QwenResearchChannels


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _load_states(directory: Path) -> dict[int, np.ndarray]:
    states: dict[int, np.ndarray] = {}
    for path in sorted(directory.glob("batch_*.npz")):
        with np.load(path, allow_pickle=False) as stored:
            for row, idx in enumerate(stored["idx"]):
                length = int(stored["lengths"][row])
                key = int(idx)
                if key in states:
                    raise ValueError(f"Duplicate generation index {key}")
                states[key] = stored["hidden_L23"][row, :length].astype(np.float32)
    return states


def _score_states(
    channels: QwenResearchChannels,
    states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    entity = channels.entity.score(states)["entity_score"].astype(np.float64)
    support_margin = channels.support.score(states)["support_margin"]
    calibration = channels.config["support_calibration"]
    support = _sigmoid(
        support_margin * float(calibration["coefficient"])
        + float(calibration["intercept"])
    )
    return entity, support, entity * support


def _answer_endpoints(
    generation: dict[str, Any],
    grade: dict[str, Any],
    states: np.ndarray,
    channels: QwenResearchChannels,
) -> list[dict[str, Any]]:
    entity_scores, support_scores, combined_scores = _score_states(channels, states)
    rows = []
    for entity in grade.get("entities", []):
        if entity.get("role") != "ANSWER":
            continue
        support_label = entity.get("support")
        if support_label not in {"SUPPORTED", "UNSUPPORTED"}:
            continue
        alignment = entity.get("token_alignment", {})
        if alignment.get("status") not in {"EXACT_REENCODE", "EXACT_OFFSET"}:
            continue
        endpoint = int(alignment["token_end"]) - 1
        if not 0 <= endpoint < len(states):
            raise ValueError(f"Endpoint outside generation {generation['idx']}")
        entity_score = float(entity_scores[endpoint])
        support_score = float(support_scores[endpoint])
        combined_score = float(combined_scores[endpoint])
        detected = (
            entity_score >= channels.entity_threshold
            and combined_score >= channels.combined_threshold
        )
        category = {
            ("UNSUPPORTED", True): "detected_unsupported",
            ("UNSUPPORTED", False): "missed_unsupported",
            ("SUPPORTED", True): "detected_supported",
            ("SUPPORTED", False): "quiet_supported",
        }[(support_label, detected)]
        rows.append(
            {
                "idx": int(generation["idx"]),
                "ordinal": int(entity.get("ordinal", 0)),
                "source": generation["src"],
                "source_id": generation["source_id"],
                "question_sha256": generation["question_sha256"],
                "question": generation["q"],
                "model_prompt": generation["model_prompt"],
                "generated_text": generation["gen_text"],
                "gold_answers": generation["gold_answers"],
                "entity_text": entity["text"],
                "support_label": support_label,
                "overall_grade": grade.get("overall_grade"),
                "grader_reason": entity.get("reason"),
                "token_start": int(alignment["token_start"]),
                "token_end": int(alignment["token_end"]),
                "endpoint": endpoint,
                "entity_score": entity_score,
                "support_score": support_score,
                "combined_score": combined_score,
                "detected": detected,
                "category": category,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    directory = args.generation_dir.resolve()
    generations = {
        int(row["idx"]): row for row in _read_json(directory / "results.json")
    }
    grades = {
        int(idx): grade
        for idx, grade in _read_json(directory / "entity_spans_v2.json")[
            "grades"
        ].items()
    }
    states = _load_states(directory)
    channels = QwenResearchChannels(args.artifacts)
    if set(generations) != set(states):
        raise ValueError("Generation rows and saved state batches do not match")

    endpoints = []
    for idx in sorted(generations):
        if idx not in grades or grades[idx].get("status") != "OK":
            continue
        endpoints.extend(
            _answer_endpoints(generations[idx], grades[idx], states[idx], channels)
        )
    endpoints.sort(key=lambda row: (row["idx"], row["ordinal"]))

    categories = {
        name: [row for row in endpoints if row["category"] == name]
        for name in (
            "detected_unsupported",
            "missed_unsupported",
            "detected_supported",
            "quiet_supported",
        )
    }
    if len(categories["detected_unsupported"]) < 3:
        raise RuntimeError("Frozen pool contains fewer than three detected false entities")
    if len(categories["quiet_supported"]) < 2:
        raise RuntimeError("Frozen pool contains fewer than two quiet supported entities")

    selected = (
        categories["detected_unsupported"][:3]
        + categories["quiet_supported"][:2]
    )
    report = {
        "schema_version": 1,
        "selection_rule": (
            "First 3 detected unsupported and first 2 quiet supported endpoints "
            "in the pre-existing frozen idx/ordinal order; no sorting by detector "
            "score. Three known false positives remain from the prior CLI gallery."
        ),
        "thresholds": {
            "entity_end": channels.entity_threshold,
            "combined_candidate": channels.combined_threshold,
        },
        "counts": {name: len(rows) for name, rows in categories.items()},
        "n_scorable_answer_endpoints": len(endpoints),
        "selected": selected,
    }
    args.output.resolve().write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["counts"], indent=2))
    for row in selected:
        print(
            f"{row['category']:22s} idx={row['idx']:3d} "
            f"combined={row['combined_score']:.4f} {row['entity_text']!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
