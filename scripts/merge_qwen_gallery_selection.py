#!/usr/bin/env python3
"""Replace five gallery false positives with three fixed-pool detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm_oscilloscope.recordings import canonical_json_sha256


REMOVE_IDS = {
    "biology_plant_food",
    "engineering_truss",
    "economics_inflation",
    "psychology_confirmation_bias",
    "psychology_peer_pressure",
}
KEEP_FALSE_POSITIVE_IDS = {
    "biology_chloroplasts",
    "biology_dna",
    "computer_ram",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--additions", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = _read(args.base.resolve())
    additions = _read(args.additions.resolve())
    selection = _read(args.selection.resolve())
    selection_by_id = {row["id"]: row for row in selection["samples"]}
    additions_by_id = {row["id"]: row for row in additions["samples"]}
    if set(additions_by_id) != set(selection_by_id):
        raise ValueError("Recorded additions do not match the fixed selection")

    base_ids = {row["id"] for row in base["samples"]}
    if REMOVE_IDS <= base_ids:
        replaced_ids = REMOVE_IDS
    elif REMOVE_IDS.isdisjoint(base_ids) and set(additions_by_id) <= base_ids:
        replaced_ids = set(additions_by_id)
    else:
        raise ValueError("Base archive is neither the original nor merged gallery")
    retained = [row for row in base["samples"] if row["id"] not in replaced_ids]

    new_rows = []
    for identifier in selection_by_id:
        sample = additions_by_id[identifier]
        audit = selection_by_id[identifier]["selection_provenance"]
        if sample["generated_text"] != audit["current_cli_output"]:
            raise ValueError(f"Current replay changed for {identifier}")
        endpoint = sample["tokens"][audit["detected_entity_endpoint"]]
        if not endpoint["entity_alert"] or not endpoint["combined_alert"]:
            raise ValueError(f"Selected entity endpoint no longer fires for {identifier}")
        sample["provenance"]["source_artifact"] = (
            "qwen_cli_gallery_v2_fixed_pool_addition"
        )
        sample["provenance"]["showcase_selection"] = {
            "source_pool": selection["selection"]["source_pool"],
            "frozen_idx": audit["frozen_idx"],
            "source": audit["source"],
            "source_id": audit["source_id"],
            "question_sha256": audit["question_sha256"],
            "category": "detected_unsupported_reference_mismatch",
            "detected_entity_endpoint": audit["detected_entity_endpoint"],
            "selection_rule": "first_three_in_frozen_pool_order",
        }
        new_rows.append(sample)

    first_natural = next(
        (
            index
            for index, row in enumerate(retained)
            if str(row["kind"]).startswith("natural_")
        ),
        len(retained),
    )
    samples = retained[:first_natural] + new_rows + retained[first_natural:]

    combined_samples = {
        row["id"]
        for row in samples
        if any(token["combined_alert"] for token in row["tokens"])
        and row["id"] not in additions_by_id
    }
    if combined_samples != KEEP_FALSE_POSITIVE_IDS:
        raise ValueError(
            "Retained prior Combined candidates differ from the three declared "
            f"false-positive examples: {sorted(combined_samples)}"
        )

    document = {
        "schema_version": 2,
        "description": (
            "Compact immutable Qwen2.5-7B-Instruct GPU recordings using the same "
            "pinned model, tokenizer, layers and measurement artifacts as live CLI "
            "generation. Three detected reference-inconsistent factual answers were "
            "selected by frozen pool order; three known false positives are retained."
        ),
        "source": {
            "artifact": "qwen_cli_gallery_v2_fixed_pool_showcase",
            "capture_status": "COMPLETE",
            "verification_status": "STRUCTURAL_AND_FIXED_POOL_SELECTION_PASS",
            "model_id": base["source"]["model_id"],
            "model_revision": base["source"]["model_revision"],
            "selection_sha256": canonical_json_sha256(selection),
            "recordings_sha256": canonical_json_sha256(samples),
            "n_samples": len(samples),
            "n_tokens": sum(len(sample["tokens"]) for sample in samples),
            "showcase_counts": {
                "detected_reference_inconsistent": len(new_rows),
                "retained_known_false_positive": len(KEEP_FALSE_POSITIVE_IDS),
            },
        },
        "samples": samples,
    }
    args.output.resolve().write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {document['source']['n_samples']} samples / "
        f"{document['source']['n_tokens']} tokens"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
