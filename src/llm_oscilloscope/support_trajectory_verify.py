"""CPU verification for the optional high-recall support trajectory profile."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .support_v2 import QwenSupportChannel


TRAJECTORY_PROTOCOL_SHA256 = (
    "2dac3e5ce1223fa37f06a7ea914e7297983a7660151b40508ad710bd3118902a"
)
TRAJECTORY_AMENDMENT_SHA256 = (
    "5df5c0ae1059661cf5b7e9387bdc26633dc11582f73a1d2a5d45b99a0018c1f7"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _confusion(labels: np.ndarray, alerts: np.ndarray) -> dict[str, float | int]:
    positive = labels == 1
    negative = labels == 0
    tp = int(np.sum(alerts & positive))
    fp = int(np.sum(alerts & negative))
    fn = int(np.sum(~alerts & positive))
    tn = int(np.sum(~alerts & negative))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": tp / max(tp + fn, 1),
        "miss_rate": fn / max(tp + fn, 1),
        "false_alarms_per_correct_alarm": fp / max(tp, 1),
        "precision": tp / max(tp + fp, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
    }


def _summary(
    labels: np.ndarray, scores: np.ndarray, alerts: np.ndarray
) -> dict[str, float | int]:
    result = _confusion(labels, alerts)
    result.update(
        {
            "n": int(len(labels)),
            "supported": int(np.sum(labels == 0)),
            "unsupported": int(np.sum(labels == 1)),
            "auroc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
        }
    )
    return result


def _close(left: object, right: object) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _close(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _close(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, (float, np.floating)) or isinstance(
        right, (float, np.floating)
    ):
        return bool(np.isclose(left, right, rtol=1e-10, atol=1e-12))
    return left == right


def _bootstrap(
    labels: np.ndarray,
    standard_scores: np.ndarray,
    high_recall_scores: np.ndarray,
    standard_alerts: np.ndarray,
    high_recall_alerts: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    rng = np.random.default_rng(24042)
    unique = np.unique(groups[mask])
    recall_gain = []
    auroc_gain = []
    for _ in range(5000):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(mask & (groups == group)) for group in sampled]
        )
        recall_gain.append(
            _confusion(labels[indices], high_recall_alerts[indices])["recall"]
            - _confusion(labels[indices], standard_alerts[indices])["recall"]
        )
        if len(np.unique(labels[indices])) == 2:
            auroc_gain.append(
                roc_auc_score(labels[indices], high_recall_scores[indices])
                - roc_auc_score(labels[indices], standard_scores[indices])
            )
    return {
        "draws": 5000,
        "recall_gain_q025_median_q975": np.quantile(
            recall_gain, [0.025, 0.5, 0.975]
        ).tolist(),
        "auroc_gain_q025_median_q975": np.quantile(
            auroc_gain, [0.025, 0.5, 0.975]
        ).tolist(),
    }


def _verify_gate(directory: Path, kind: str) -> str:
    stored = json.loads((directory / f"{kind}_results.json").read_text())
    verification = json.loads(
        (directory / f"{kind}_verification.json").read_text()
    )
    with np.load(directory / f"{kind}_predictions.npz", allow_pickle=False) as data:
        labels = data["labels"].astype(np.int8)
        standard_scores = data["fast_score"].astype(float)
        high_recall_scores = data["candidate_score"].astype(float)
        standard_alerts = data["fast_alert"].astype(bool)
        high_recall_alerts = data["candidate_alert"].astype(bool)
        groups = data["groups"].astype(str)
        sources = data["sources"].astype(str)
        modes = data["modes"].astype(str)
        positions = data["positions"].astype(int)

    standard = _summary(labels, standard_scores, standard_alerts)
    high_recall = _summary(labels, high_recall_scores, high_recall_alerts)
    if not _close(standard, stored["fast_support"]):
        raise AssertionError(f"Support trajectory {kind} standard metrics mismatch")
    if not _close(high_recall, stored["candidate"]):
        raise AssertionError(f"Support trajectory {kind} high-recall metrics mismatch")

    by_source = {}
    for source in sorted(set(sources)):
        mask = sources == source
        by_source[source] = {
            "fast": _summary(labels[mask], standard_scores[mask], standard_alerts[mask]),
            "candidate": _summary(
                labels[mask], high_recall_scores[mask], high_recall_alerts[mask]
            ),
        }
    source_macro = {
        name: float(np.mean([value[name]["auroc"] for value in by_source.values()]))
        for name in ("fast", "candidate")
    }
    if not _close(by_source, stored["by_source"]):
        raise AssertionError(f"Support trajectory {kind} source metrics mismatch")
    if not _close(source_macro, stored["macro_auroc"]["source"]):
        raise AssertionError(f"Support trajectory {kind} source macro mismatch")

    overall_bootstrap = _bootstrap(
        labels,
        standard_scores,
        high_recall_scores,
        standard_alerts,
        high_recall_alerts,
        groups,
        np.ones(len(labels), dtype=bool),
    )
    if not _close(overall_bootstrap, stored["question_bootstrap"]["overall"]):
        raise AssertionError(f"Support trajectory {kind} bootstrap mismatch")

    if kind == "surface":
        by_mode = {}
        for mode in sorted(set(modes)):
            mask = modes == mode
            by_mode[mode] = {
                "fast": _summary(
                    labels[mask], standard_scores[mask], standard_alerts[mask]
                ),
                "candidate": _summary(
                    labels[mask], high_recall_scores[mask], high_recall_alerts[mask]
                ),
            }
        mode_macro = {
            name: float(np.mean([value[name]["auroc"] for value in by_mode.values()]))
            for name in ("fast", "candidate")
        }
        checks = {
            "coverage": standard["supported"] >= 100
            and standard["unsupported"] >= 100,
            "source_class_coverage": all(
                value["fast"]["supported"] > 0
                and value["fast"]["unsupported"] > 0
                for value in by_source.values()
            ),
            "mode_coverage": all(
                value["fast"]["supported"] >= 30
                and value["fast"]["unsupported"] >= 30
                for value in by_mode.values()
            ),
            "recall_gain": high_recall["recall"] - standard["recall"] >= 0.03,
            "bootstrap_recall_lower": overall_bootstrap[
                "recall_gain_q025_median_q975"
            ][0]
            > 0,
            "additional_false_alerts": high_recall["fp"] - standard["fp"] <= 2,
            "false_alarm_ratio": high_recall["false_alarms_per_correct_alarm"]
            - standard["false_alarms_per_correct_alarm"]
            <= 0.005,
            "auroc_gain": high_recall["auroc"] - standard["auroc"] >= 0.005,
            "source_macro_gain": source_macro["candidate"] - source_macro["fast"]
            >= 0.005,
            "mode_macro_gain": mode_macro["candidate"] - mode_macro["fast"]
            >= 0.005,
            "stratum_recall": all(
                value["candidate"]["recall"] >= value["fast"]["recall"] - 0.02
                for value in list(by_source.values()) + list(by_mode.values())
            ),
        }
        if not _close(by_mode, stored["by_mode"]):
            raise AssertionError("Support trajectory surface mode metrics mismatch")
        if not _close(mode_macro, stored["macro_auroc"]["mode"]):
            raise AssertionError("Support trajectory surface mode macro mismatch")
        status = (
            "FRESH_SURFACE_V2_PASS"
            if all(checks.values())
            else "NO_FRESH_SURFACE_V2_PASS"
        )
    else:
        late = positions >= 8
        late_standard = _summary(
            labels[late], standard_scores[late], standard_alerts[late]
        )
        late_high_recall = _summary(
            labels[late], high_recall_scores[late], high_recall_alerts[late]
        )
        late_bootstrap = _bootstrap(
            labels,
            standard_scores,
            high_recall_scores,
            standard_alerts,
            high_recall_alerts,
            groups,
            late,
        )
        checks = {
            "coverage": standard["supported"] >= 60
            and standard["unsupported"] >= 60,
            "source_class_coverage": all(
                value["fast"]["supported"] > 0
                and value["fast"]["unsupported"] > 0
                for value in by_source.values()
            ),
            "late_coverage": late_standard["n"] >= 100
            and late_standard["supported"] > 0
            and late_standard["unsupported"] > 0,
            "overall_recall_gain": high_recall["recall"] - standard["recall"]
            >= 0.03,
            "overall_bootstrap_recall_lower": overall_bootstrap[
                "recall_gain_q025_median_q975"
            ][0]
            > 0,
            "overall_additional_false_alerts": high_recall["fp"]
            - standard["fp"]
            <= 2,
            "overall_false_alarm_ratio": high_recall[
                "false_alarms_per_correct_alarm"
            ]
            - standard["false_alarms_per_correct_alarm"]
            <= 0.005,
            "overall_auroc_non_degradation": high_recall["auroc"]
            >= standard["auroc"] - 0.01,
            "late_recall_gain": late_high_recall["recall"]
            - late_standard["recall"]
            >= 0.03,
            "late_bootstrap_recall_lower": late_bootstrap[
                "recall_gain_q025_median_q975"
            ][0]
            > 0,
            "late_additional_false_alerts": late_high_recall["fp"]
            - late_standard["fp"]
            <= 2,
            "late_false_alarm_ratio": late_high_recall[
                "false_alarms_per_correct_alarm"
            ]
            - late_standard["false_alarms_per_correct_alarm"]
            <= 0.005,
            "late_auroc_non_degradation": late_high_recall["auroc"]
            >= late_standard["auroc"] - 0.01,
            "source_macro_non_degradation": source_macro["candidate"]
            >= source_macro["fast"] - 0.01,
            "source_recall": all(
                value["candidate"]["recall"] >= value["fast"]["recall"] - 0.02
                for value in by_source.values()
            ),
        }
        if not _close(
            {"fast": late_standard, "candidate": late_high_recall},
            stored["late_positions_8_plus"],
        ):
            raise AssertionError("Support trajectory delayed late metrics mismatch")
        if not _close(late_bootstrap, stored["question_bootstrap"]["late"]):
            raise AssertionError("Support trajectory delayed late bootstrap mismatch")
        status = (
            "FRESH_DELAYED_V2_PASS"
            if all(checks.values())
            else "NO_FRESH_DELAYED_V2_PASS"
        )

    if not _close(checks, stored["checks"]):
        raise AssertionError(f"Support trajectory {kind} frozen checks mismatch")
    if status != stored["status"]:
        raise AssertionError(f"Support trajectory {kind} frozen status mismatch")
    if verification["status"] != "SUPPORT_CONDITIONAL_SE_RELEASE_V2_VERIFICATION_PASS":
        raise AssertionError(f"Support trajectory {kind} stored verification failed")
    if verification["stored_status"] != status:
        raise AssertionError(f"Support trajectory {kind} verification status mismatch")
    return status


def verify_support_trajectory(root: Path) -> None:
    directory = root / "artifacts" / "support_v2" / "trajectory_high_recall"
    head = root / "artifacts" / "support_v2" / "qwen_native_support_head.npz"
    profile = directory / "profile.npz"
    calibration = json.loads((directory / "calibration.json").read_text())
    if calibration["output_artifact_sha256"] != _sha256(profile):
        raise AssertionError("Support trajectory profile hash mismatch")
    if calibration["protocol_sha256"] != TRAJECTORY_PROTOCOL_SHA256:
        raise AssertionError("Support trajectory protocol hash mismatch")
    QwenSupportChannel(head, profile)

    full_replay = json.loads((directory / "surface_full_replay_fail.json").read_text())
    if full_replay["status"] != "GENERATION_STATE_REPLAY_FAIL":
        raise AssertionError("Original support trajectory replay failure was not preserved")
    for kind in ("surface", "delayed"):
        replay = json.loads((directory / f"{kind}_cached_replay.json").read_text())
        if replay["protocol_amendment_sha256"] != TRAJECTORY_AMENDMENT_SHA256:
            raise AssertionError(f"Support trajectory {kind} amendment hash mismatch")
        if replay["status"] != "CACHED_GENERATION_STATE_REPLAY_PASS":
            raise AssertionError(f"Support trajectory {kind} cached replay did not pass")

    surface_status = _verify_gate(directory, "surface")
    delayed_status = _verify_gate(directory, "delayed")
    if surface_status != "NO_FRESH_SURFACE_V2_PASS":
        raise AssertionError("Support trajectory surface failure was not preserved")
    if delayed_status != "NO_FRESH_DELAYED_V2_PASS":
        raise AssertionError("Support trajectory delayed failure was not preserved")
    print(
        "PASS: verified optional high-recall support profile "
        "and reproduced both frozen false-alarm gate failures"
    )
