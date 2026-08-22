#!/usr/bin/env python3
"""Record the CLI prompt gallery with the pinned live Qwen runtime.

The run is resumable: every completed sample is written to ``--work-dir``
before the final embedded archive is assembled. The input archive supplies only
stable prompt-suite metadata; none of its Llama generations or scores are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm_oscilloscope.qwen_runtime import QwenInstrumentedRuntime, TokenReading
from llm_oscilloscope.recordings import canonical_json_sha256


CHANNELS = [
    "entity_end",
    "support_v2",
    "combined_candidate",
    "subject_routing",
]
TITLE_OVERRIDES = {
    "mathematics_fraction": "Mathematics · Adding fractions",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _prompt_suite(document: dict[str, Any]) -> list[dict[str, Any]]:
    samples = document.get("samples")
    if document.get("schema_version") != 2 or not isinstance(samples, list):
        raise ValueError("Input must be a schema-v2 recorded-sample archive")
    suite = []
    for sample in samples:
        suite.append(
            {
                "id": sample["id"],
                "title": TITLE_OVERRIDES.get(sample["id"], sample["title"]),
                "kind": sample["kind"],
                "fixed_showcase": bool(sample.get("fixed_showcase", False)),
                "prompt": sample["prompt"],
            }
        )
    identifiers = [row["id"] for row in suite]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Prompt-suite identifiers are not unique")
    return suite


def _is_short_factual(specification: dict[str, Any]) -> bool:
    return str(specification["kind"]).startswith("simple_")


def _recording(
    specification: dict[str, Any],
    rows: list[TokenReading],
    runtime_manifest: dict[str, Any],
    environment: dict[str, str],
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"Qwen generated no retained tokens for {specification['id']}")
    short_factual = _is_short_factual(specification)
    thresholds = runtime_manifest["thresholds"]
    entity_threshold = float(thresholds["entity_balanced"]["value"])
    combined_threshold = float(thresholds["combined_balanced"]["value"])

    token_rows = []
    for position, reading in enumerate(rows):
        values = reading.as_dict()
        if reading.position != position:
            raise ValueError(
                f"Non-contiguous token positions for {specification['id']}"
            )
        if not short_factual:
            values["entity_alert"] = False
            values["combined_alert"] = False
        values["switch_marker"] = False
        token_rows.append(values)

    layers = runtime_manifest["layers"]
    artifacts = runtime_manifest["artifact_sha256"]
    return {
        **specification,
        "generated_text": "".join(row.text for row in rows),
        "available_channels": CHANNELS,
        "provenance": {
            "gpu_recorded": True,
            "source_artifact": "qwen_cli_gallery_v1",
            "model_id": runtime_manifest["model_id"],
            "model_revision": runtime_manifest["model_revision"],
            "quantization": "bitsandbytes_int8",
            "decoding": "greedy",
            "seed": 0,
            **environment,
            "channel_layers": {
                "entity_end": int(layers["detector"]),
                "support_v2": int(layers["detector"]),
                "combined_candidate": int(layers["detector"]),
                "subject_routing": int(layers["subject_routing"]),
            },
            "token_alignment": "original_gpu_token_ids",
            "thresholds": {
                "entity_end": entity_threshold if short_factual else None,
                "combined_candidate": combined_threshold if short_factual else None,
            },
            "threshold_status": (
                "post_hoc_short_factual_cli_candidates"
                if short_factual
                else "score_only_outside_candidate_scope"
            ),
            "threshold_basis": (
                "Post-hoc Qwen CLI candidates from the opened short factual-QA "
                "development sample; not a released detector operating point."
                if short_factual
                else "Natural-text gallery row outside the evaluated short factual-QA "
                "scope; raw scores only and alert flags disabled."
            ),
            "entity_adapter_sha256": artifacts["entity_transport"],
            "support_head_sha256": artifacts["support_v2"],
            "subject_adapter_sha256": artifacts["subject_routing"],
            "token_rows_sha256": canonical_json_sha256(token_rows),
            "evidence_boundary": runtime_manifest["evidence_boundary"],
        },
        "tokens": token_rows,
    }


def _validate(samples: list[dict[str, Any]], model_revision: str) -> None:
    if not samples:
        raise ValueError("No Qwen recordings were captured")
    for sample in samples:
        provenance = sample["provenance"]
        if provenance["model_revision"] != model_revision:
            raise ValueError(f"Model revision mismatch in {sample['id']}")
        short_factual = _is_short_factual(sample)
        entity_threshold = provenance["thresholds"]["entity_end"]
        combined_threshold = provenance["thresholds"]["combined_candidate"]
        for position, row in enumerate(sample["tokens"]):
            if row["position"] != position or len(row["subject_scores"]) != 8:
                raise ValueError(f"Invalid token row in {sample['id']} at {position}")
            expected_product = row["entity_end_score"] * row["support_score"]
            if abs(row["combined_score"] - expected_product) > 1e-10:
                raise ValueError(
                    f"Combined score mismatch in {sample['id']} at {position}"
                )
            if short_factual:
                if row["entity_alert"] != (row["entity_end_score"] >= entity_threshold):
                    raise ValueError(f"Entity flag mismatch in {sample['id']}")
                if row["combined_alert"] != (
                    row["combined_score"] >= combined_threshold
                ):
                    raise ValueError(f"Combined flag mismatch in {sample['id']}")
            elif row["entity_alert"] or row["combined_alert"]:
                raise ValueError(f"OOD alert flag remained active in {sample['id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--simple-max-new-tokens", type=int, default=64)
    parser.add_argument("--natural-max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    suite = _prompt_suite(_read_json(args.input.resolve()))
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    samples_by_id: dict[str, dict[str, Any]] = {}
    pending = []
    for specification in suite:
        cached = work_dir / f"{specification['id']}.json"
        if cached.is_file():
            samples_by_id[specification["id"]] = _read_json(cached)
        else:
            pending.append(specification)

    if pending:
        import bitsandbytes
        import torch
        import transformers

        environment = {
            "torch_version": str(torch.__version__),
            "transformers_version": str(transformers.__version__),
            "bitsandbytes_version": str(bitsandbytes.__version__),
            "cuda_version": str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(0)),
        }
        with QwenInstrumentedRuntime(
            artifacts_root=args.artifacts,
            quantization="8bit",
            allow_download=args.allow_download,
        ) as runtime:
            runtime_manifest = runtime.trace_manifest("generate", "")["runtime"]
            for index, specification in enumerate(pending, start=1):
                limit = (
                    args.simple_max_new_tokens
                    if _is_short_factual(specification)
                    else args.natural_max_new_tokens
                )
                print(
                    f"[{index}/{len(pending)}] {specification['id']} "
                    f"(max {limit})",
                    flush=True,
                )
                rows = list(
                    runtime.generate(
                        specification["prompt"],
                        max_new_tokens=limit,
                        temperature=0.0,
                        seed=0,
                    )
                )
                sample = _recording(
                    specification,
                    rows,
                    runtime_manifest,
                    environment,
                )
                _write_json(work_dir / f"{specification['id']}.json", sample)
                samples_by_id[specification["id"]] = sample
                print(
                    f"  {len(rows)} tokens · {sample['generated_text']!r}",
                    flush=True,
                )

    samples = [samples_by_id[row["id"]] for row in suite]
    model_id = samples[0]["provenance"]["model_id"]
    model_revision = samples[0]["provenance"]["model_revision"]
    _validate(samples, model_revision)
    document = {
        "schema_version": 2,
        "description": (
            "Compact immutable Qwen2.5-7B-Instruct GPU recordings using the same "
            "pinned model, tokenizer, layers and measurement artifacts as live CLI "
            "generation. Hidden states, reference answers and manual labels are not "
            "distributed."
        ),
        "source": {
            "artifact": "qwen_cli_gallery_v1",
            "capture_status": "COMPLETE",
            "verification_status": "STRUCTURAL_PASS",
            "model_id": model_id,
            "model_revision": model_revision,
            "prompt_suite_sha256": canonical_json_sha256(suite),
            "recordings_sha256": canonical_json_sha256(samples),
            "n_samples": len(samples),
            "n_tokens": sum(len(sample["tokens"]) for sample in samples),
        },
        "samples": samples,
    }
    _write_json(args.output.resolve(), document)
    print(
        f"Wrote {len(samples)} Qwen samples / {document['source']['n_tokens']} tokens "
        f"to {args.output.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
