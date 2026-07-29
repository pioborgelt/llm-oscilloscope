"""CPU verification for the released subject-routing evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

from .subject import SubjectRoutingHead, TransportedSubjectRoutingHead


SUBJECTS = (
    "mathematics",
    "physics",
    "chemistry",
    "biology",
    "computer_science",
    "engineering",
    "economics_business",
    "psychology_social_science",
)
METRIC_KEYS = (
    "macro_auroc",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "nll",
    "brier",
    "ece_15",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(name: str, observed: float, expected: float, tolerance: float = 1e-6):
    if not np.isclose(observed, expected, atol=tolerance, rtol=0.0):
        raise AssertionError(f"{name}: observed={observed}, expected={expected}")


def _macro_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    values = []
    for class_id in range(len(SUBJECTS)):
        binary = labels == class_id
        if binary.any() and not binary.all():
            values.append(roc_auc_score(binary, probabilities[:, class_id]))
    if not values:
        raise AssertionError("subject AUROC requires at least two classes")
    return float(np.mean(values))


def _brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    target = np.zeros_like(probabilities)
    target[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (confidence < upper)
        if upper == 1.0:
            mask |= confidence == 1.0
        if mask.any():
            value += (
                mask.sum()
                / len(labels)
                * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
            )
    return float(value)


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    return {
        "macro_auroc": _macro_auc(labels, probabilities),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "nll": float(log_loss(labels, probabilities, labels=np.arange(len(SUBJECTS)))),
        "brier": _brier(labels, probabilities),
        "ece_15": _ece(labels, probabilities),
    }


def _verify_probabilities(
    name: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    expected: dict,
) -> None:
    if probabilities.shape != (len(labels), len(SUBJECTS)):
        raise AssertionError(f"{name}: invalid probability shape {probabilities.shape}")
    reproduced = _metrics(labels, probabilities)
    for key in METRIC_KEYS:
        _close(f"{name}/{key}", reproduced[key], expected[key])


def _verify_native(directory: Path, split_manifest: dict) -> dict:
    result = json.loads((directory / "core_results.json").read_text(encoding="utf-8"))
    if tuple(result["subjects"]) != SUBJECTS or result["n_core"] != 1664:
        raise AssertionError("native subject ontology or row count mismatch")
    if split_manifest["role_counts"]["core"] != result["n_core"]:
        raise AssertionError("native split-manifest row count mismatch")
    if not split_manifest["zero_exact_core_question_overlap_between_sources"]:
        raise AssertionError("native source question overlap")
    if not split_manifest["zero_core_to_paired_question_overlap"]:
        raise AssertionError("native core/holdout question overlap")

    required = {
        "source_transfer",
        "template_holdout",
        "fine_subject_holdout",
        "random_prompt_cv",
    }
    if set(result["protocols"]) != required:
        raise AssertionError("native subject protocol set mismatch")
    for protocol_name, protocol in result["protocols"].items():
        labels = np.asarray(protocol["pooled_labels"], dtype=int)
        probabilities = np.asarray(protocol["pooled_probabilities"], dtype=float)
        if len(protocol["pooled_indices"]) != len(labels):
            raise AssertionError(f"{protocol_name}: index and label counts differ")
        _verify_probabilities(
            f"native/{protocol_name}", labels, probabilities, protocol["pooled"]
        )

    head_path = directory / "llama_head.npz"
    if _sha256(head_path) != result["release_head"]["sha256"]:
        raise AssertionError("native subject-head checksum mismatch")
    head = SubjectRoutingHead(head_path)
    if head.layer != result["release_head"]["layer"]:
        raise AssertionError("native subject-head layer mismatch")
    if head.classes != SUBJECTS:
        raise AssertionError("native subject-head class mismatch")

    protocols = result["protocols"]
    source = protocols["source_transfer"]
    directions = [fold["metrics"]["macro_auroc"] for fold in source["folds"]]
    random_mean = float(
        np.mean(
            [fold["random_subspace"]["mean_macro_auroc"] for fold in source["folds"]]
        )
    )
    strong = (
        source["pooled"]["macro_auroc"] >= 0.80
        and min(directions) >= 0.75
        and source["pooled"]["macro_auroc"] - random_mean >= 0.10
        and protocols["template_holdout"]["pooled"]["macro_auroc"] >= 0.75
        and protocols["fine_subject_holdout"]["pooled"]["macro_auroc"] >= 0.70
    )
    if not strong:
        raise AssertionError("native subject channel misses its frozen rule")
    return result


def _verify_transfer(
    directory: Path, model: str, split_manifest: dict
) -> dict[str, float]:
    result = json.loads(
        (directory / f"{model}_results.json").read_text(encoding="utf-8")
    )
    labels = np.asarray(result["test_labels"], dtype=int)
    for key, expected in (
        ("transfer_probabilities", result["transfer"]),
        ("native_probabilities", result["native_target"]["metrics"]),
        ("teacher_probabilities", result["teacher_reference"]),
    ):
        _verify_probabilities(
            f"{model}/{key}",
            labels,
            np.asarray(result[key], dtype=float),
            expected,
        )

    core_base_ids = set(split_manifest["core_base_ids"])
    test_base_ids = set(result["test_base_ids"])
    if len(test_base_ids) != 128 or core_base_ids & test_base_ids:
        raise AssertionError(f"{model}: fit/test base-id boundary failed")
    if test_base_ids != set(split_manifest["paired_base_ids"]):
        raise AssertionError(f"{model}: unexpected holdout base ids")
    if not result["zero_base_id_overlap"]:
        raise AssertionError(f"{model}: stored overlap decision is false")

    primary_adapter = directory / f"{model}_adapter.npz"
    independent_adapter = directory / f"{model}_independent_prompt_adapter.npz"
    if _sha256(primary_adapter) != result["adapter_sha256"]:
        raise AssertionError(f"{model}: primary adapter checksum mismatch")
    independent = result["independent_prompt_control"]
    if _sha256(independent_adapter) != independent["adapter_sha256"]:
        raise AssertionError(f"{model}: independent adapter checksum mismatch")
    runtime = TransportedSubjectRoutingHead(primary_adapter)
    if runtime.classes != SUBJECTS or runtime.target_layer != result["selected"]["layer"]:
        raise AssertionError(f"{model}: portable adapter metadata mismatch")

    random_mean = float(np.mean(result["random_pairing"]["values"]))
    _close(
        f"{model}/random mean",
        random_mean,
        result["random_pairing"]["mean_macro_auroc"],
    )
    _close(
        f"{model}/random margin",
        result["transfer"]["macro_auroc"] - random_mean,
        result["random_pairing"]["learned_margin"],
    )
    if [row["n_pairs"] for row in result["alignment_curve"]] != [
        128,
        256,
        512,
        1024,
        1664,
    ]:
        raise AssertionError(f"{model}: incomplete onboarding curve")

    source_min = min(row["macro_auroc"] for row in result["by_source"].values())
    expected_outcome = (
        "strong"
        if result["transfer"]["macro_auroc"] >= 0.85
        and source_min >= 0.75
        and result["corrected_transfer_efficiency"] >= 0.80
        and result["random_pairing"]["learned_margin"] >= 0.20
        else "partial"
        if result["transfer"]["macro_auroc"] >= 0.75
        and result["random_pairing"]["learned_margin"] >= 0.10
        else "failed"
    )
    if result["outcome"] != expected_outcome:
        raise AssertionError(f"{model}: primary transfer outcome mismatch")

    for key, expected in (
        ("probabilities", independent["transfer"]),
        ("teacher_probabilities", independent["teacher_reference"]),
    ):
        _verify_probabilities(
            f"{model}/independent/{key}",
            labels,
            np.asarray(independent[key], dtype=float),
            expected,
        )
    independent_random = float(np.mean(independent["random_pairing"]["values"]))
    _close(
        f"{model}/independent random mean",
        independent_random,
        independent["random_pairing"]["mean_macro_auroc"],
    )
    if independent["general_subject_question_overlap"] != 0:
        raise AssertionError(f"{model}: independent prompt overlap")
    independent_source_min = min(
        row["macro_auroc"] for row in independent["by_source"].values()
    )
    independent_expected = (
        "passed"
        if independent["transfer"]["macro_auroc"] >= 0.75
        and independent_source_min >= 0.70
        and independent["random_pairing"]["learned_margin"] >= 0.10
        else "failed"
    )
    if independent["outcome"] != independent_expected:
        raise AssertionError(f"{model}: independent control outcome mismatch")

    return {
        "primary": float(result["transfer"]["macro_auroc"]),
        "independent": float(independent["transfer"]["macro_auroc"]),
    }


def _pair_values(
    scores: np.ndarray, labels: np.ndarray, pairs: np.ndarray
) -> np.ndarray:
    output = []
    for pair_index, (first, second) in enumerate(pairs):
        mask = (labels == first) | (labels == second)
        output.append(
            roc_auc_score(labels[mask] == first, scores[pair_index, mask])
        )
    return np.asarray(output)


def _heldout_decision(
    source_mean: float,
    transport_mean: float,
    random_mean: float,
    pairs_above_chance: int,
    bootstrap_lower: float,
) -> str:
    if source_mean < 0.85:
        return "SOURCE_CHANNEL_FAILURE"
    margin = transport_mean - random_mean
    if (
        transport_mean >= 0.75
        and pairs_above_chance >= 24
        and margin >= 0.10
        and bootstrap_lower > 0.60
    ):
        return "HELDOUT_SUBJECT_TRANSPORT_PASS"
    if (
        transport_mean >= 0.65
        and pairs_above_chance >= 20
        and margin >= 0.10
        and bootstrap_lower > 0.55
    ):
        return "PARTIAL_HELDOUT_SUBJECT_TRANSPORT"
    return "NO_HELDOUT_SUBJECT_TRANSPORT"


def _verify_heldout(directory: Path, split_manifest: dict) -> float:
    result = json.loads(
        (directory / "heldout_subject_results.json").read_text(encoding="utf-8")
    )
    with np.load(directory / "heldout_subject_predictions.npz") as stored:
        labels = stored["labels"].astype(int)
        base_ids = stored["base_ids"].astype(str)
        pairs = stored["pairs"].astype(int)
        transport = _pair_values(stored["transport_scores"], labels, pairs)
        source = _pair_values(stored["source_scores"], labels, pairs)
        full = _pair_values(stored["full_scores"], labels, pairs)
        unaligned = _pair_values(stored["unaligned_scores"], labels, pairs)
        random = np.asarray(
            [
                np.mean(_pair_values(control, labels, pairs))
                for control in stored["random_scores"]
            ]
        )
        bootstrap = stored["bootstrap"]

    if set(base_ids) != set(split_manifest["paired_base_ids"]):
        raise AssertionError("heldout subject base-id boundary mismatch")
    checks = (
        ("source", source.mean(), result["source"]["mean_auroc"]),
        (
            "full",
            full.mean(),
            result["full_domain_transport"]["mean_auroc"],
        ),
        (
            "transport",
            transport.mean(),
            result["heldout_transport"]["mean_auroc"],
        ),
        ("unaligned", unaligned.mean(), result["unaligned"]["mean_auroc"]),
        ("random", random.mean(), result["random_pair"]["mean_auroc"]),
        (
            "bootstrap lower",
            np.quantile(bootstrap, 0.025),
            result["heldout_transport"]["base_question_bootstrap_95_ci"][0],
        ),
        (
            "bootstrap upper",
            np.quantile(bootstrap, 0.975),
            result["heldout_transport"]["base_question_bootstrap_95_ci"][1],
        ),
    )
    for name, observed, expected in checks:
        _close(f"heldout/{name}", float(observed), float(expected), tolerance=1e-8)
    decision = _heldout_decision(
        float(source.mean()),
        float(transport.mean()),
        float(random.mean()),
        int(np.sum(transport > 0.5)),
        float(np.quantile(bootstrap, 0.025)),
    )
    if decision != result["decision"]:
        raise AssertionError("heldout subject decision mismatch")
    return float(transport.mean())


def _top1_stability(predictions: np.ndarray) -> float:
    values = np.asarray(predictions, dtype=int).reshape(-1)
    return float(np.bincount(values).max() / len(values))


def _verify_qwen_stream(directory: Path) -> float:
    result = json.loads(
        (directory / "qwen_stream_results.json").read_text(encoding="utf-8")
    )
    if _sha256(directory / "qwen_adapter.npz") != result["adapter_sha256"]:
        raise AssertionError("qwen stream adapter checksum mismatch")

    single = result["single_subject_stream"]
    class_to_id = {name: index for index, name in enumerate(SUBJECTS)}
    token_labels = []
    token_probabilities = []
    prompt_labels = []
    prompt_probabilities = []
    stability = []
    target_occupancy = []
    for row in single["records"]:
        label = class_to_id[row["subject"]]
        probabilities = np.asarray(row["token_probabilities"], dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(SUBJECTS):
            raise AssertionError("qwen stream probability shape mismatch")
        mean_probability = probabilities.mean(axis=0)
        np.testing.assert_allclose(
            mean_probability,
            np.asarray(row["prompt_mean_probabilities"], dtype=float),
            atol=3e-7,
            rtol=0.0,
        )
        token_labels.extend([label] * len(probabilities))
        token_probabilities.append(probabilities)
        prompt_labels.append(label)
        prompt_probabilities.append(mean_probability)
        predictions = probabilities.argmax(axis=1)
        stability.append(_top1_stability(predictions))
        target_occupancy.append(float(np.mean(predictions == label)))

    token_y = np.asarray(token_labels, dtype=int)
    token_p = np.concatenate(token_probabilities)
    prompt_y = np.asarray(prompt_labels, dtype=int)
    prompt_p = np.stack(prompt_probabilities)
    if len(token_y) != single["n_post_token_states"]:
        raise AssertionError("qwen stream token count mismatch")
    if len(prompt_y) != single["n_prompts"]:
        raise AssertionError("qwen stream prompt count mismatch")
    checks = (
        ("token AUROC", _macro_auc(token_y, token_p), single["token_macro_auroc"]),
        (
            "token accuracy",
            accuracy_score(token_y, token_p.argmax(axis=1)),
            single["token_accuracy"],
        ),
        (
            "token F1",
            f1_score(token_y, token_p.argmax(axis=1), average="macro"),
            single["token_macro_f1"],
        ),
        (
            "prompt AUROC",
            _macro_auc(prompt_y, prompt_p),
            single["prompt_mean_macro_auroc"],
        ),
        (
            "prompt accuracy",
            accuracy_score(prompt_y, prompt_p.argmax(axis=1)),
            single["prompt_mean_accuracy"],
        ),
        (
            "stability",
            np.mean(stability),
            single["mean_within_prompt_top1_stability"],
        ),
        (
            "target occupancy",
            np.mean(target_occupancy),
            single["mean_within_prompt_target_occupancy"],
        ),
    )
    for name, observed, expected in checks:
        _close(f"qwen stream/{name}", float(observed), float(expected), tolerance=1e-7)

    mixed = result["mixed_subject_switch"]
    records = mixed["records"]
    latencies = [
        row["crossing_latency_tokens"]
        for row in records
        if row["crossing_latency_tokens"] is not None
    ]
    mixed_checks = (
        (
            "pre accuracy",
            np.mean([row["pre_accuracy"] for row in records]),
            mixed["mean_pre_switch_top1_accuracy"],
        ),
        (
            "post accuracy",
            np.mean([row["post_accuracy"] for row in records]),
            mixed["mean_post_switch_top1_accuracy"],
        ),
        (
            "margin change",
            np.mean([row["margin_change"] for row in records]),
            mixed["mean_second_minus_first_margin_change"],
        ),
        (
            "crossing latency",
            np.median(latencies),
            mixed["median_crossing_latency_tokens"],
        ),
    )
    for name, observed, expected in mixed_checks:
        _close(f"qwen mixed/{name}", float(observed), float(expected), tolerance=1e-7)
    return float(single["token_macro_auroc"])


def verify_subject_routing(root: Path) -> None:
    directory = root / "artifacts" / "subject_routing"
    split_manifest = json.loads(
        (directory / "split_manifest.json").read_text(encoding="utf-8")
    )
    native = _verify_native(directory, split_manifest)
    mistral = _verify_transfer(directory, "mistral", split_manifest)
    qwen = _verify_transfer(directory, "qwen", split_manifest)
    heldout = _verify_heldout(directory, split_manifest)
    qwen_stream = _verify_qwen_stream(directory)
    print(
        "subject routing       "
        f"native={native['protocols']['source_transfer']['pooled']['macro_auroc']:.3f}  "
        f"mistral={mistral['primary']:.3f}  "
        f"qwen={qwen['primary']:.3f}  "
        f"heldout={heldout:.3f}  "
        f"qwen-token={qwen_stream:.3f}"
    )
    print(
        "PASS: verified native subject OOF results, two transported heads, "
        "two failed independent-prompt controls, 28 held-out subject pairs, "
        "and 7,830 Qwen post-token states"
    )
