"""CPU verification for the compact Qwen-native transfer evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


SOURCES = ("simpleqa", "granola")
CHANNELS = ("entity", "support")
SEED = 20260724


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(len(labels)),
        "positive": int(np.sum(labels == 1)),
        "negative": int(np.sum(labels == 0)),
        "prevalence": float(np.mean(labels)),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def _bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    prompts: np.ndarray,
    mask: np.ndarray,
    iterations: int,
    seed: int,
) -> list[float]:
    unique = np.unique(prompts[mask])
    rows = {int(prompt): np.flatnonzero(mask & (prompts == prompt)) for prompt in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([rows[int(prompt)] for prompt in sampled])
        if len(np.unique(labels[selected])) == 2:
            values.append(float(roc_auc_score(labels[selected], scores[selected])))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _close(actual: Any, expected: Any, label: str, tolerance: float = 1e-10) -> None:
    if isinstance(expected, (int, np.integer)):
        if int(actual) != int(expected):
            raise AssertionError(f"{label}: {actual} != {expected}")
    elif abs(float(actual) - float(expected)) > tolerance:
        raise AssertionError(f"{label}: {actual} != {expected}")


def _check_metrics(actual: dict[str, Any], expected: dict[str, Any], label: str) -> int:
    keys = ("n", "positive", "negative", "prevalence", "auroc", "average_precision")
    for key in keys:
        _close(actual[key], expected[key], f"{label}/{key}")
    return len(keys)


def _channel_pass(result: dict[str, Any], threshold: float) -> bool:
    primary = result["transport"]
    return bool(
        primary["auroc"] >= threshold
        and primary["auroc"] - result["random_pair"]["mean_auroc"] >= 0.10
        and primary["prompt_bootstrap_95_ci"][0] > 0.50
        and all(result["by_source"][source]["auroc"] > 0.50 for source in SOURCES)
    )


def _verify_adapter(path: Path) -> None:
    expected = {
        "target_pca_components": (64, 3584),
        "source_pca_components": (64, 4096),
        "ridge_coef": (64, 64),
        "random_ridge_coef": (20, 64, 64),
        "entity_coef": (4096,),
        "support_coef": (4096,),
    }
    with np.load(path, allow_pickle=False) as stored:
        for key, shape in expected.items():
            if key not in stored or stored[key].shape != shape:
                raise AssertionError(f"adapter {key}: unexpected shape")
            if not np.isfinite(stored[key]).all():
                raise AssertionError(f"adapter {key}: non-finite values")


def verify_qwen_native_transfer(root: Path) -> None:
    directory = root / "artifacts" / "qwen_native_transfer"
    result = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    with np.load(directory / "predictions.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    _verify_adapter(directory / "adapter.npz")

    checks = 0
    iterations = int(result["bootstrap_iterations"])
    for position, channel in enumerate(CHANNELS):
        labels = arrays[f"{channel}_label"]
        scores = arrays[f"{channel}_transport_score"]
        mask = labels >= 0
        checks += _check_metrics(
            _metrics(labels[mask], scores[mask]),
            result["channels"][channel]["transport"],
            f"{channel}/transport",
        )
        interval = _bootstrap(
            labels, scores, arrays["prompt_idx"], mask, iterations, SEED + position
        )
        for actual, expected in zip(
            interval, result["channels"][channel]["transport"]["prompt_bootstrap_95_ci"]
        ):
            _close(actual, expected, f"{channel}/bootstrap")
            checks += 1

        for source in SOURCES:
            source_mask = mask & (arrays["source"] == source)
            checks += _check_metrics(
                _metrics(labels[source_mask], scores[source_mask]),
                result["channels"][channel]["by_source"][source],
                f"{channel}/{source}",
            )

        random_values = [
            _metrics(labels[mask], values[mask])["auroc"]
            for values in arrays[f"{channel}_random_scores"]
        ]
        random_result = result["channels"][channel]["random_pair"]
        _close(float(np.mean(random_values)), random_result["mean_auroc"], f"{channel}/random mean")
        _close(float(np.std(random_values)), random_result["std_auroc"], f"{channel}/random std")
        checks += 2

        native_mask = mask & np.isfinite(arrays[f"{channel}_native_score"])
        checks += _check_metrics(
            _metrics(labels[native_mask], arrays[f"{channel}_native_score"][native_mask]),
            result["channels"][channel]["native_target_labeled_ceiling"],
            f"{channel}/native",
        )

    combined_mask = arrays["combined_label"] >= 0
    checks += _check_metrics(
        _metrics(
            arrays["combined_label"][combined_mask],
            arrays["combined_score"][combined_mask],
        ),
        result["combined_secondary"],
        "combined",
    )
    combined_interval = _bootstrap(
        arrays["combined_label"],
        arrays["combined_score"],
        arrays["prompt_idx"],
        combined_mask,
        iterations,
        SEED + 10,
    )
    for actual, expected in zip(
        combined_interval, result["combined_secondary"]["prompt_bootstrap_95_ci"]
    ):
        _close(actual, expected, "combined/bootstrap")
        checks += 1

    coverage = result["coverage"]
    coverage_pass = bool(
        coverage["successful_grades"] >= 350
        and coverage["aligned_answer_endpoints"] >= 250
        and coverage["supported_endpoints"] >= 60
        and coverage["unsupported_endpoints"] >= 60
    )
    entity_pass = _channel_pass(result["channels"]["entity"], 0.80)
    support_pass = _channel_pass(result["channels"]["support"], 0.65)
    status = (
        "INSUFFICIENT_NATIVE_COVERAGE"
        if not coverage_pass
        else "QWEN_NATIVE_TWO_CHANNEL_PASS"
        if entity_pass and support_pass
        else "QWEN_NATIVE_SINGLE_CHANNEL_ONLY"
        if entity_pass or support_pass
        else "NO_QWEN_NATIVE_CHANNEL_PASS"
    )
    expected = {
        "status": status,
        "coverage_pass": coverage_pass,
        "entity_transport_pass": entity_pass,
        "support_transport_pass": support_pass,
    }
    if result["decision"] != expected:
        raise AssertionError("Qwen-native decision does not follow its frozen rule")
    print(f"PASS: verified Qwen-native two-channel transfer ({checks} metric checks)")
