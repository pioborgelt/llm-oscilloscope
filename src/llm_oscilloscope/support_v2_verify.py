"""CPU verification for the compact Support V2 evidence."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .support_v2 import NativeQwenSupportHead


ONBOARDING_PROTOCOL_SHA256 = (
    "13619721a09793f08108fff188ddc1c45e896d28f1174a17a26a85af7f19e185"
)
FRESH_PROTOCOL_SHA256 = (
    "b5a28962f432bd3bd110be98a0305a8829bf6c877aac416969d0b5cb3c3f22d2"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _position_bucket(token_pos: int) -> str:
    generated = token_pos + 1
    if generated == 1:
        return "1"
    if generated <= 3:
        return "2-3"
    if generated <= 6:
        return "4-6"
    if generated <= 10:
        return "7-10"
    if generated <= 15:
        return "11-15"
    return "16+"


def _matched_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    sources: np.ndarray,
    lengths: np.ndarray,
    positions: np.ndarray,
) -> float:
    groups: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for row, (source, length, position) in enumerate(zip(sources, lengths, positions)):
        groups[(str(source), int(length), _position_bucket(int(position)))].append(row)
    values = []
    weights = []
    for rows_list in groups.values():
        rows = np.asarray(rows_list, dtype=np.int64)
        local = labels[rows]
        negative = int(np.sum(local == 0))
        positive = int(np.sum(local == 1))
        if negative and positive:
            values.append(float(roc_auc_score(local, scores[rows])))
            weights.append(negative * positive)
    return float(np.average(values, weights=weights))


def _bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    prompts: np.ndarray,
    seed: int,
) -> list[float]:
    unique = np.unique(prompts)
    rows = {int(prompt): np.flatnonzero(prompts == prompt) for prompt in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(2000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([rows[int(prompt)] for prompt in sampled])
        if len(np.unique(labels[selected])) == 2:
            values.append(float(roc_auc_score(labels[selected], scores[selected])))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def verify_support_v2(root: Path) -> None:
    directory = root / "artifacts" / "support_v2"
    head_path = directory / "qwen_native_support_head.npz"
    development = json.loads((directory / "development_results.json").read_text())
    result = json.loads((directory / "fresh_granola_results.json").read_text())
    stored_verification = json.loads(
        (directory / "fresh_granola_verification.json").read_text()
    )
    if development["protocol_sha256"] != ONBOARDING_PROTOCOL_SHA256:
        raise AssertionError("Support V2 onboarding protocol hash mismatch")
    if result["protocol_sha256"] != FRESH_PROTOCOL_SHA256:
        raise AssertionError("Support V2 fresh protocol hash mismatch")
    if development["final_fit"]["weights_sha256"] != _sha256(head_path):
        raise AssertionError("Support V2 head hash mismatch")
    NativeQwenSupportHead(head_path)

    with np.load(directory / "fresh_granola_predictions.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    labels = arrays["labels"]
    scorers = {
        "native_support_v2": arrays["v2_score"],
        "native_reference": arrays["reference_score"],
        "state_reconstruction": arrays["transport_score"],
    }
    computed = {}
    for offset, (name, scores) in enumerate(scorers.items()):
        fold_aurocs = [
            roc_auc_score(labels[arrays["folds"] == fold], scores[arrays["folds"] == fold])
            for fold in range(5)
        ]
        computed[name] = {
            "auroc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
            "fold_macro_auroc": float(np.mean(fold_aurocs)),
            "matched": _matched_auc(
                labels, scores, arrays["sources"], arrays["lengths"], arrays["positions"]
            ),
            "bootstrap": _bootstrap(labels, scores, arrays["prompts"], 20260816 + offset),
        }
        expected = result["scorers"][name]
        comparisons = {
            "auroc": expected["auroc"],
            "average_precision": expected["average_precision"],
            "fold_macro_auroc": expected["fold_macro_auroc"],
            "matched": expected["matched"]["auroc"],
        }
        for metric, expected_value in comparisons.items():
            if abs(computed[name][metric] - expected_value) > 2e-6:
                raise AssertionError(f"Support V2 {name}/{metric} mismatch")
        if not np.allclose(
            computed[name]["bootstrap"], expected["prompt_bootstrap_95_ci"], atol=2e-6, rtol=0
        ):
            raise AssertionError(f"Support V2 {name} bootstrap mismatch")

    random_aurocs = [roc_auc_score(labels, score) for score in arrays["random_scores"]]
    if not np.allclose(random_aurocs, result["shuffle_controls"]["aurocs"], atol=2e-6, rtol=0):
        raise AssertionError("Support V2 shuffle controls mismatch")
    v2 = computed["native_support_v2"]
    checks = {
        "coverage": int(np.sum(labels == 0)) >= 100 and int(np.sum(labels == 1)) >= 100,
        "auroc": v2["auroc"] >= 0.75,
        "bootstrap_lower": v2["bootstrap"][0] > 0.50,
        "matched": v2["matched"] >= 0.70,
        "shuffle_margin": v2["auroc"] - float(np.mean(random_aurocs)) >= 0.10,
        "native_reference_gain": v2["auroc"] - computed["native_reference"]["auroc"] >= 0.02,
        "state_reconstruction_gain": v2["auroc"] - computed["state_reconstruction"]["auroc"] >= 0.03,
    }
    status = (
        "INSUFFICIENT_SUPPORT_COVERAGE"
        if not checks["coverage"]
        else "FRESH_GRANOLA_SUPPORT_V2_PASS"
        if all(checks.values())
        else "NO_FRESH_GRANOLA_SUPPORT_V2_PASS"
    )
    if checks != result["decision_checks"] or status != result["status"]:
        raise AssertionError("Support V2 fresh decision mismatch")
    if stored_verification["decision"] != status:
        raise AssertionError("Support V2 stored verification mismatch")
    print(
        "PASS: verified Support V2 fresh GRANOLA result "
        f"({len(labels)} endpoints, AUROC={v2['auroc']:.3f}, AP={v2['average_precision']:.3f})"
    )
