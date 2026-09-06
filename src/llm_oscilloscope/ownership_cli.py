"""CLI for an explicit ownership comparison; never a default generation warning."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from rich.console import Console
from rich.table import Table

from .ownership import ChoiceReading, OwnershipCase, replay_assay


def _load_json(path: Path, maximum: int = 2_000_000):
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError(f"Input exceeds the {maximum}-byte limit")
    return json.loads(raw)


def packaged_samples() -> dict:
    return _load_json(Path(__file__).with_name("ownership_samples.json"))


def _display(recording: dict, measurement: dict) -> None:
    console = Console()
    console.print(
        "Sycophancy assay · experimental user-ownership comparison", style="bold"
    )
    console.print(recording["case"]["title"], markup=False)
    console.print(f"Model: {recording['runtime']['model_id']}", markup=False)
    console.print(
        "Active A/B comparison, not a harm detector or a passive chat warning."
    )
    console.print()
    console.print("A: " + recording["case"]["side_a"], markup=False)
    console.print("B: " + recording["case"]["side_b"], markup=False)
    console.print()
    table = Table("Measurement", "Value")
    table.add_row(
        "User-ownership shift", f"{measurement['ownership_shift_pp']:+.2f} pp"
    )
    table.add_row(
        "Person-X control shift", f"{measurement['person_x_shift_pp']:+.2f} pp"
    )
    table.add_row(
        "User minus Person-X (descriptive)",
        f"{measurement['ownership_minus_person_x_pp']:+.2f} pp",
    )
    table.add_row(
        "Difference between presentation orders",
        f"{measurement['ownership_order_spread_pp']:.2f} pp",
    )
    table.add_row(
        "Answers switch with the owner",
        f"{measurement['natural_owner_switch_pairs']}/2 order pairs",
    )
    table.add_row("Valid completed A/B answers", f"{measurement['valid_answers']}/10")
    table.add_row(
        "Minimum A/B vocabulary mass", f"{100 * measurement['min_choice_mass']:.2f}%"
    )
    console.print(table)
    console.print(
        "pp means percentage points of the conditional A/B readout, not the probability of sycophancy."
    )
    if measurement["quality"]["status"] != "ok":
        console.print(
            "OFF PROTOCOL: " + ", ".join(measurement["quality"]["flags"]),
            style="yellow",
            markup=False,
        )
    details = Table("Order", "Condition", "P(A | A/B)", "Answer")
    labels = {
        "neutral": "No owner",
        "user_a": "User owns A",
        "user_b": "User owns B",
        "person_x_a": "Person X owns A",
        "person_x_b": "Person X owns B",
    }
    for event in recording["events"]:
        reading = ChoiceReading(
            **{
                key: event[key]
                for key in (
                    "a_logit",
                    "b_logit",
                    "choice_mass",
                    "output_text",
                    "eos_complete",
                )
            }
        )
        details.add_row(
            event["order"].upper(),
            labels[event["arm"]],
            f"{100 * reading.a_probability:.2f}%",
            reading.answer or "unknown",
        )
    console.print(details)


def command_sycophancy(args) -> int:
    try:
        if args.live and args.input is None:
            raise ValueError("--live requires --input with two explicit accounts")
        if args.input is not None and not args.live:
            raise ValueError(
                "--input requires --live; use --recording to replay existing measurements"
            )
        if not args.live and (
            args.allow_download or args.cache_dir is not None or args.model is not None
        ):
            raise ValueError(
                "Model/profile options apply only to --live; recordings retain their original model"
            )
        if args.output is not None:
            if args.output.exists() or args.output.is_symlink():
                raise ValueError(
                    "Output already exists; choose a new path. No recording was overwritten."
                )
            if not args.output.parent.is_dir():
                raise ValueError(
                    "The output directory does not exist; choose an existing directory"
                )
        if args.live:
            raw = _load_json(args.input, maximum=100_000)
            if not isinstance(raw, dict) or set(raw) - {"side_a", "side_b", "title"}:
                raise ValueError(
                    "Input must be an object with side_a, side_b and an optional title"
                )
            case = OwnershipCase(**raw)
            from .ownership_runtime import OwnershipRuntime

            print(
                "Running ten controlled local evaluations (no harm-alarm threshold).",
                file=sys.stderr,
            )
            with OwnershipRuntime(
                model=args.model or "smollm3",
                cache_dir=args.cache_dir,
                allow_download=args.allow_download,
            ) as runtime:
                recording = runtime.measure(case)
        elif args.recording is not None:
            recording = _load_json(args.recording)
        else:
            samples = packaged_samples()
            key = args.sample or "workshop"
            if key not in samples:
                raise ValueError(
                    "Unknown sample. Available: " + ", ".join(sorted(samples))
                )
            recording = samples[key]
        if not isinstance(recording, dict):
            raise ValueError("Recording must be a JSON object")
        measurement = replay_assay(recording)
        # Recompute rather than trusting an imported summary field.
        recording = {**recording, "measurement": measurement}
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(
                    recording, handle, ensure_ascii=False, indent=2, allow_nan=False
                )
                handle.write("\n")
        if args.json:
            print(json.dumps(recording, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            _display(recording, measurement)
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def add_sycophancy_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "sycophancy",
        help="experimental active ownership assay (GPU-free recorded samples by default)",
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--sample",
        help="recorded sample: workshop, arithmetic, qwen3-workshop or qwen3-arithmetic",
    )
    inputs.add_argument(
        "--recording", type=Path, help="recompute a saved ownership recording"
    )
    inputs.add_argument(
        "--input",
        type=Path,
        help="JSON with side_a, side_b and optional title; requires --live",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="explicitly evaluate a new comparison on the local GPU",
    )
    parser.add_argument(
        "--model",
        choices=("smollm3", "qwen3"),
        help="live profile (default: SmolLM3-3B; alternative: Qwen3-1.7B)",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow download of the selected pinned model weights",
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="optional existing Hugging Face model cache"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="save a new JSON recording; never overwrite an existing file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="machine-readable recording including all ten conditions",
    )
    parser.set_defaults(func=command_sycophancy)
