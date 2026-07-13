"""Recompute the paired prompt bootstrap shipped with the evidence release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


def load_payload(path: Path, prompt_idx: np.ndarray, score_key: str) -> dict[str, np.ndarray]:
    with np.load(path) as stored:
        rows = stored["rows"].astype(np.int64)
        return {
            "labels": stored["labels"].astype(np.int8),
            "scores": stored[score_key].astype(np.float64),
            "prompts": prompt_idx[rows].astype(np.int32),
        }


def weighted_ap(payload, counts, lookup) -> float:
    indices = np.asarray([lookup[int(idx)] for idx in payload["prompts"]], dtype=np.int32)
    weights = counts[indices]
    keep = weights > 0
    return float(
        average_precision_score(
            payload["labels"][keep], payload["scores"][keep], sample_weight=weights[keep]
        )
    )


def recompute(root: Path, iterations: int, seed: int) -> dict[str, list[float] | float]:
    artifacts = root / "artifacts"
    with np.load(artifacts / "online_metadata.npz") as stored:
        prompt_idx = stored["prompt_idx"]
    new = load_payload(artifacts / "oof_random.npz", prompt_idx, "completion_product")
    official = load_payload(
        artifacts / "oof_random.npz", prompt_idx, "boundary_x_official"
    )
    old = load_payload(artifacts / "old_oof_random.npz", prompt_idx, "product")
    unique = np.unique(np.concatenate([new["prompts"], old["prompts"]]))
    lookup = {int(idx): position for position, idx in enumerate(unique)}
    rng = np.random.default_rng(seed)
    deltas = {"new_minus_old": [], "new_minus_official": []}
    for _ in range(iterations):
        sampled = rng.integers(0, len(unique), size=len(unique))
        counts = np.bincount(sampled, minlength=len(unique))
        new_ap = weighted_ap(new, counts, lookup)
        deltas["new_minus_old"].append(new_ap - weighted_ap(old, counts, lookup))
        deltas["new_minus_official"].append(
            new_ap - weighted_ap(official, counts, lookup)
        )
    return {
        name: {
            "mean": float(np.mean(values)),
            "ci95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
        }
        for name, values in deltas.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.root.resolve()
    actual = recompute(root, args.iterations, args.seed)
    expected_document = json.loads(
        (root / "artifacts" / "analysis.json").read_text(encoding="utf-8")
    )
    expected = expected_document["random_paired_prompt_bootstrap"]["metrics"]
    if args.iterations == expected_document["random_paired_prompt_bootstrap"]["iterations"]:
        for name, value in actual.items():
            if abs(value["mean"] - expected[name]["mean"]) > 1e-12:
                raise AssertionError(f"{name} bootstrap mean mismatch")
            if np.max(np.abs(np.asarray(value["ci95"]) - expected[name]["ci95"])) > 1e-12:
                raise AssertionError(f"{name} bootstrap CI mismatch")
    print(json.dumps(actual, indent=2, sort_keys=True))
    print(f"PASS: paired prompt bootstrap ({args.iterations} iterations)")


if __name__ == "__main__":
    main()

