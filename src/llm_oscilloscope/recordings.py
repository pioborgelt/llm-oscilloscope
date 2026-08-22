"""Load and validate immutable recordings used by the CLI demos."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from typing import Any


REQUIRED_CHANNELS = {
    "entity_end",
    "support_v2",
    "combined_candidate",
    "subject_routing",
}


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON after stable, whitespace-independent serialization."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


REQUIRED_TOKEN_FIELDS = {
    "position",
    "token_id",
    "text",
    "token_probability",
    "entity_end_score",
    "support_score",
    "combined_score",
    "entity_alert",
    "combined_alert",
    "subject_scores",
}


def load_recordings() -> dict[str, Any]:
    resource = files("llm_oscilloscope").joinpath("recorded_samples.json")
    document = json.loads(resource.read_text(encoding="utf-8"))
    if document.get("schema_version") != 2:
        raise ValueError("Unsupported recorded-sample schema")
    samples = document.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Recorded samples are empty")
    identifiers = [sample.get("id") for sample in samples]
    if len(set(identifiers)) != len(identifiers) or any(not value for value in identifiers):
        raise ValueError("Recorded sample identifiers must be unique and nonempty")
    for sample in samples:
        if set(sample.get("available_channels", [])) != REQUIRED_CHANNELS:
            raise ValueError(f"Recorded sample {sample['id']} does not contain all channels")
        tokens = sample.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"Recorded sample {sample['id']} has no token rows")
        for position, row in enumerate(tokens):
            if not REQUIRED_TOKEN_FIELDS <= row.keys() or row["position"] != position:
                raise ValueError(
                    f"Recorded sample {sample['id']} has an invalid token row at {position}"
                )
            if not isinstance(row["token_id"], int) or len(row["subject_scores"]) != 8:
                raise ValueError(
                    f"Recorded sample {sample['id']} has invalid token/channel data"
                )
        expected_rows_hash = sample.get("provenance", {}).get("token_rows_sha256")
        if expected_rows_hash != canonical_json_sha256(tokens):
            raise ValueError(f"Recorded sample {sample['id']} has a checksum mismatch")
    source = document.get("source", {})
    if source.get("n_samples") != len(samples) or source.get("n_tokens") != sum(
        len(sample["tokens"]) for sample in samples
    ):
        raise ValueError("Recorded-sample manifest counts do not match the archive")
    if source.get("recordings_sha256") != canonical_json_sha256(samples):
        raise ValueError("Recorded-sample archive checksum does not match")
    return document


def list_recordings() -> list[dict[str, Any]]:
    return list(load_recordings()["samples"])


def get_recording(identifier: str) -> dict[str, Any]:
    samples = list_recordings()
    if identifier.isdecimal():
        index = int(identifier) - 1
        if 0 <= index < len(samples):
            return samples[index]
    for sample in samples:
        if sample["id"] == identifier:
            return sample
    available = ", ".join(
        f"{index}: {sample['id']}" for index, sample in enumerate(samples, start=1)
    )
    raise KeyError(f"Unknown recorded sample {identifier!r}. Available: {available}")
