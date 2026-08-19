"""CPU verification of released OOF evidence and weight artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .detector import DetectorHeads
from .qwen_verify import verify_qwen_native_transfer
from .subject_verify import verify_subject_routing
from .support_v2_verify import verify_support_v2
from .support_trajectory_verify import verify_support_trajectory


METHODS = (
    "completion_product",
    "boundary_only",
    "correctness_only",
    "official_only",
    "boundary_x_official",
    "surprisal",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> None:
    manifest = root / "MANIFEST.sha256"
    if not manifest.exists():
        raise AssertionError("MANIFEST.sha256 is missing")
    listed_paths: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if relative in listed_paths:
            raise AssertionError(f"duplicate release manifest entry: {relative}")
        listed_paths.add(relative)
        path = root / relative.removeprefix("./")
        if not path.is_file():
            raise AssertionError(f"release manifest path is missing: {relative}")
        if sha256(path) != expected:
            raise AssertionError(f"release manifest mismatch: {relative}")

    artifacts = root / "artifacts"
    results = json.loads((artifacts / "all_results.json").read_text(encoding="utf-8"))

    print("split                 AUROC       AP")
    for mode in ("random", "entity_disjoint", "loso"):
        with np.load(artifacts / f"oof_{mode}.npz") as stored:
            labels = stored["labels"]
            for method in METHODS:
                auc = float(roc_auc_score(labels, stored[method]))
                ap = float(average_precision_score(labels, stored[method]))
                expected = results["modes"][mode]["aggregate"][method]
                if abs(auc - expected["auroc"]) > 1e-10:
                    raise AssertionError(f"{mode}/{method} AUROC mismatch")
                if abs(ap - expected["average_precision"]) > 1e-10:
                    raise AssertionError(f"{mode}/{method} AP mismatch")
                if method == "completion_product":
                    print(f"{mode:20s} {auc:.6f}  {ap:.6f}")

    entity_analysis = json.loads(
        (artifacts / "entity_completion_analysis.json").read_text(encoding="utf-8")
    )
    with np.load(artifacts / "entity_completion_scores.npz") as stored:
        for mode in ("random", "entity_disjoint", "loso"):
            prefix = f"tokenwise__{mode}"
            labels = stored[f"{prefix}__label"]
            expected_mode = entity_analysis["nested_cv"]["tokenwise"][mode]["oof_metrics"]
            for position in ("first", "last", "max"):
                scores = stored[f"{prefix}__{position}"]
                auc = float(roc_auc_score(labels, scores))
                ap = float(average_precision_score(labels, scores))
                expected = expected_mode[position]
                if abs(auc - expected["auroc"]) > 1e-10:
                    raise AssertionError(
                        f"entity completion {mode}/{position} AUROC mismatch"
                    )
                if abs(ap - expected["average_precision"]) > 1e-10:
                    raise AssertionError(
                        f"entity completion {mode}/{position} AP mismatch"
                    )

    holdout = json.loads(
        (artifacts / "matched_holdout_v2_evaluation.json").read_text(encoding="utf-8")
    )
    if not holdout["decision_rule"]["passed"]:
        raise AssertionError("matched holdout decision rule is not marked passed")
    pooled = holdout["entity_end"]["pooled"]["correctness"]
    source_rows = [
        holdout["entity_end"][source]["correctness"]
        for source in ("simpleqa", "granola")
    ]
    for count_key in ("n", "n_positive", "n_negative"):
        if pooled[count_key] != sum(row[count_key] for row in source_rows):
            raise AssertionError(f"matched holdout pooled {count_key} mismatch")
    ci = holdout["entity_end"]["pooled"]["prompt_bootstrap"]["metrics"]["auroc"]["ci95"]
    per_source_auc = [row["auroc"] for row in source_rows]
    expected_pass = ci[0] > 0.5 and all(value >= 0.5 for value in per_source_auc)
    if bool(expected_pass) != bool(holdout["decision_rule"]["passed"]):
        raise AssertionError("matched holdout decision rule is internally inconsistent")

    split_manifest = json.loads(
        (artifacts / "entity_disjoint_manifest.json").read_text(encoding="utf-8")
    )
    if len(split_manifest["folds"]) != 5:
        raise AssertionError("expected five entity-disjoint split manifests")
    for fold in split_manifest["folds"]:
        if fold["overlap"] != 0:
            raise AssertionError(f"entity overlap in {fold['fold']}")
        if set(fold["train_key_sha256"]) & set(fold["test_key_sha256"]):
            raise AssertionError(f"hashed entity overlap in {fold['fold']}")

    weight_dir = artifacts / "weights"
    weight_files = sorted(weight_dir.glob("*.npz"))
    if len(weight_files) != 17:
        raise AssertionError(f"expected 17 weight artifacts, found {len(weight_files)}")
    for path in weight_files:
        heads = DetectorHeads(path)
        if heads.arrays["boundary_coef"].shape != (1, 4096):
            raise AssertionError(f"bad boundary shape in {path.name}")
        if heads.arrays["correctness_coef"].shape != (1, 4096):
            raise AssertionError(f"bad correctness shape in {path.name}")
    for mode in ("random", "entity_disjoint", "loso"):
        for fold in results["modes"][mode]["folds"]:
            roundtrip = fold["weight_artifact_roundtrip"]
            if max(
                roundtrip["boundary_max_abs_error"],
                roundtrip["correctness_max_abs_error"],
            ) > 1e-5:
                raise AssertionError(f"roundtrip error in {mode}/{fold['fold']}")
            path = weight_dir / f"{mode}_{fold['fold']}.npz"
            if sha256(path) != roundtrip["sha256"]:
                raise AssertionError(f"weight checksum mismatch: {path.name}")
    print(
        "PASS: verified 3 per-token OOF evaluations, "
        "3 entity-completion OOF evaluations, and "
        f"{len(weight_files)} detector artifacts"
    )
    verify_subject_routing(root)
    verify_qwen_native_transfer(root)
    verify_support_v2(root)
    verify_support_trajectory(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    verify(args.root.resolve())


if __name__ == "__main__":
    main()
