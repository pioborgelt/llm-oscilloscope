"""Command-line research preview for instrumented Qwen generation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.padding import Padding
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from . import __version__
from .channels import QwenResearchChannels, find_artifacts_root, load_cli_config
from .qwen_runtime import (
    QwenInstrumentedRuntime,
    TokenReading,
    _llm_int8_hardware_issue,
    write_jsonl_trace,
)
from .recordings import get_recording, list_recordings
from .ownership_cli import add_sycophancy_parser
from .tui import (
    AMBER,
    GOLD,
    GOLD_SOFT,
    IVORY,
    NAVY,
    NAVY_BRIGHT,
    SUBJECT_LABELS,
    _meter,
    recording_view,
    run_recording_tui,
)


SUBJECT_SHORT = {
    "mathematics": "math",
    "physics": "physics",
    "chemistry": "chem",
    "biology": "biology",
    "computer_science": "cs",
    "engineering": "engineering",
    "economics_business": "econ",
    "psychology_social_science": "psych/social",
}


def _score(value: float | None) -> str:
    return "   - " if value is None else f"{value:5.3f}"


def _gated_score(value: float | None, active: bool) -> Text:
    text = Text(_score(value), style=None if active else "dim")
    return text


def _token_text(value: str | None, token_id: int | None) -> str:
    if value is None:
        return f"#{token_id}" if token_id is not None else "<recorded-position>"
    escaped = value.replace("\n", "\\n").replace("\t", "\\t")
    return repr(escaped)[:24]


def _top_subject(scores: dict[str, float] | None) -> str:
    if not scores:
        return "-"
    top = sorted(scores.items(), key=lambda row: row[1], reverse=True)[:2]
    return " ".join(f"{SUBJECT_SHORT.get(name, name)}:{value:.2f}" for name, value in top)


def command_samples(args: argparse.Namespace) -> int:
    samples = list_recordings()
    if args.json:
        print(json.dumps(samples, indent=2, ensure_ascii=False))
        return 0
    table = Table(
        title="Recorded GPU samples",
        box=box.SIMPLE_HEAVY,
        header_style=f"bold {GOLD}",
        border_style=NAVY_BRIGHT,
    )
    table.add_column("#", justify="right", style=GOLD, no_wrap=True)
    table.add_column("Sample ID", style="bold white", no_wrap=True)
    table.add_column("Recording")
    table.add_column("Channels", style=NAVY_BRIGHT)
    for index, sample in enumerate(samples, start=1):
        channels = ", ".join(sample["available_channels"])
        table.add_row(str(index), sample["id"], sample["title"], channels)
    Console().print(table)
    return 0


def _static_demo(sample: dict[str, Any]) -> None:
    console = Console()
    console.print(recording_view(sample, 0, width=console.size.width))
    table = Table(
        title="Complete recorded timeline",
        box=box.SIMPLE_HEAVY,
        header_style=f"bold {GOLD}",
    )
    table.add_column("pos", justify="right")
    table.add_column("token")
    channels = set(sample["available_channels"])
    detector = bool({"entity_end", "support_v2", "combined_candidate"} & channels)
    if detector:
        table.add_column("p(tok)", justify="right")
        table.add_column("entity", justify="right")
        table.add_column("support", justify="right")
        table.add_column("combined", justify="right")
        table.add_column("threshold state")
        for row in sample["tokens"]:
            downstream_active = bool(row.get("entity_alert"))
            flags = []
            if row.get("entity_alert"):
                flags.append("ENTITY CANDIDATE")
            if row.get("combined_alert"):
                flags.append("COMBINED CANDIDATE")
            threshold_state = " ".join(flags)
            style = f"bold {AMBER}" if row.get("combined_alert") else None
            table.add_row(
                str(row["position"]),
                _token_text(row.get("text"), row.get("token_id")),
                _score(row.get("token_probability")),
                _score(row.get("entity_end_score")),
                _gated_score(row.get("support_score"), downstream_active),
                _gated_score(row.get("combined_score"), downstream_active),
                Text(threshold_state, style=style),
            )
    else:
        table.add_column("top routing scores")
        table.add_column("state")
        for row in sample["tokens"]:
            table.add_row(
                str(row["position"]),
                _token_text(row.get("text"), row.get("token_id")),
                _top_subject(row.get("subject_scores")),
                "SWITCH" if row.get("switch_marker") else "scored token",
            )
    console.print(table)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        console.print(Rule(characters="─", style=GOLD_SOFT))
        console.print(
            Text.from_markup(
                "Interactive token navigation starts automatically in a terminal · "
                "use [bold]--interactive[/] to require it."
            )
        )


def command_demo(args: argparse.Namespace) -> int:
    try:
        sample = get_recording(args.sample)
    except KeyError as error:
        print(str(error), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(sample, indent=2, ensure_ascii=False))
        return 0
    terminal = sys.stdin.isatty() and sys.stdout.isatty()
    if args.interactive and not terminal:
        print("error: interactive mode requires a terminal (TTY)", file=sys.stderr)
        return 2
    if args.interactive or (terminal and not args.static):
        return run_recording_tui(sample)
    _static_demo(sample)
    return 0


def _runtime(args: argparse.Namespace) -> QwenInstrumentedRuntime:
    return QwenInstrumentedRuntime(
        artifacts_root=args.artifacts,
        quantization=args.quantization,
        device_map=args.device_map,
        max_gpu_memory=args.max_gpu_memory,
        max_cpu_memory=args.max_cpu_memory,
        allow_download=args.allow_download,
    )


def _live_scope(
    args: argparse.Namespace,
    mode: str,
    manifest: dict[str, Any],
    *,
    token_limit: int | None = None,
) -> tuple[str, str]:
    runtime = manifest.get("runtime", {})
    if runtime.get("synthetic"):
        if mode == "generate":
            return "SYNTHETIC", f"Token limit {token_limit}"
        return "SYNTHETIC", "User-supplied completion · score-only synthetic replay"
    if mode == "replay":
        return "", "User-supplied completion · score-only forced replay"
    if args.preset == "freeform":
        return "OUT OF DISTRIBUTION", f"Scores only · token limit {token_limit}"
    return "", f"Short factual QA · candidate thresholds · token limit {token_limit}"


def _candidate_thresholds(
    manifest: dict[str, Any], *, active: bool
) -> dict[str, float | None]:
    if not active:
        return {"entity_end": None, "combined_candidate": None}
    configured = manifest.get("runtime", {}).get("thresholds")
    if not configured:
        configured = load_cli_config()["thresholds"]
    return {
        "entity_end": float(configured["entity_balanced"]["value"]),
        "combined_candidate": float(configured["combined_balanced"]["value"]),
    }


def _live_signal_table(
    row: TokenReading,
    *,
    width: int,
    thresholds: dict[str, float | None],
) -> Table:
    values = row.as_dict()
    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim", no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column(justify="right", no_wrap=True)
    channels = (
        ("entity", values["entity_end_score"], "entity_end", NAVY_BRIGHT),
        ("support", values["support_score"], None, NAVY_BRIGHT),
        ("combined", values["combined_score"], "combined_candidate", AMBER),
    )
    meter_width = 10 if width < 56 else 18
    for label, value, threshold_key, color in channels:
        threshold = thresholds.get(threshold_key) if threshold_key else None
        state = f"{value:.3f}"
        if threshold is not None and value >= threshold:
            state += "  candidate"
        elif threshold_key is not None and threshold is None:
            state += "  score only"
        table.add_row(
            label,
            _meter(value, color, threshold=threshold, width=meter_width),
            state,
        )

    scores = values.get("subject_scores") or {}
    if scores:
        subject, score = max(scores.items(), key=lambda item: item[1])
        table.add_row(
            "subject",
            Text(SUBJECT_LABELS.get(subject, subject), style=NAVY_BRIGHT),
            f"{score:.3f}",
        )
    return table


def _live_view(
    *,
    mode: str,
    scope: str,
    scope_detail: str,
    prompt: str,
    text: str,
    row: TokenReading | None,
    complete: bool,
    width: int,
    thresholds: dict[str, float | None],
) -> RenderableType:
    title = Text("LLM OSCILLOSCOPE", style=f"bold {GOLD}")
    title.append(f"  v{__version__}", style="dim")
    mode_text = mode.upper()
    if scope:
        mode_text += f"  ·  {scope}"
    mode_label = Text(mode_text, style=f"bold {GOLD_SOFT}")
    if width < 56:
        masthead: RenderableType = Group(title, mode_label)
    else:
        masthead_table = Table.grid(expand=True, padding=0)
        masthead_table.add_column(ratio=1)
        masthead_table.add_column(justify="right")
        masthead_table.add_row(title, mode_label)
        masthead = masthead_table

    context = Table.grid(padding=(0, 2))
    context.add_column(style="dim", no_wrap=True)
    context.add_column()
    context.add_row("PROMPT", Text(prompt, style=IVORY))

    response = Text("OUTPUT  ", style="dim")
    response.append(text or "…", style=f"bold {IVORY}")
    if not complete:
        response.append("▌", style=GOLD)

    content: list[RenderableType] = [
        masthead,
        Rule(characters="─", style=NAVY),
        Text("Research preview: Read the README before testing this.", style=GOLD_SOFT),
        Text(scope_detail, style="dim"),
        context,
        response,
    ]
    if row is not None:
        token = row.text.replace(" ", "␠").replace("\n", "↵") or "∅"
        if width < 56:
            token_detail = (
                f"TOKEN {row.position:02}  {token!r} · id {row.token_id} · "
                f"p {row.token_probability:.3f}"
            )
        else:
            token_detail = (
                f"TOKEN   {row.position:02}  {token!r}   ·   id {row.token_id}   ·   "
                f"p(tok) {row.token_probability:.3f}"
            )
        detail = Text(token_detail, style="dim")
        content.extend(
            (
                detail,
                Padding(
                    _live_signal_table(
                        row,
                        width=width,
                        thresholds=thresholds,
                    ),
                    (0, 0, 0, 2 if width < 56 else 8),
                ),
            )
        )
    if complete:
        count = 0 if row is None else row.position + 1
        content.append(Text(f"{count} tokens generated", style="dim"))
    return Group(*content)


def _live_recording(
    mode: str,
    prompt: str,
    rows: list[TokenReading],
    manifest: dict[str, Any],
    *,
    thresholds: dict[str, float | None],
) -> dict[str, Any]:
    synthetic = bool(manifest.get("runtime", {}).get("synthetic"))
    thresholds_active = thresholds["entity_end"] is not None
    scoped_rows = _rows_for_scope(rows, thresholds_active=thresholds_active)
    return {
        "id": f"{'dev-fake-' if synthetic else 'live-'}{mode}",
        "title": "Generated token stream" if mode == "generate" else "Forced replay token stream",
        "kind": "detector",
        "prompt": prompt,
        "generated_text": "".join(row.text for row in rows),
        "available_channels": [
            "entity_end",
            "support_v2",
            "combined_candidate",
            "subject_routing",
        ],
        "provenance": {
            "thresholds": dict(thresholds),
            "threshold_status": (
                "post_hoc_short_factual_cli_candidates"
                if thresholds_active
                else "score_only_outside_candidate_scope"
            ),
        },
        "tokens": [row.as_dict() for row in scoped_rows],
    }


def _rows_for_scope(
    rows: list[TokenReading], *, thresholds_active: bool
) -> list[TokenReading]:
    if thresholds_active:
        return rows
    scoped = []
    for row in rows:
        channels = dict(row.channels)
        channels["entity_alert"] = False
        channels["combined_alert"] = False
        scoped.append(
            TokenReading(
                position=row.position,
                token_id=row.token_id,
                text=row.text,
                token_probability=row.token_probability,
                channels=channels,
            )
        )
    return scoped


def _run_live(
    args: argparse.Namespace,
    mode: str,
    prompt: str,
    completion: str | None = None,
) -> int:
    console = Console()
    terminal = sys.stdin.isatty() and sys.stdout.isatty()
    rows: list[TokenReading] = []
    try:
        with _runtime(args) as runtime:
            manifest = runtime.trace_manifest(mode, prompt)
            maximum: int | None = None
            if mode == "generate":
                maximum = args.max_new_tokens
                if maximum is None:
                    maximum = 32 if args.preset == "short-factual" else 256
            scope, scope_detail = _live_scope(
                args,
                mode,
                manifest,
                token_limit=maximum,
            )
            thresholds_active = mode == "generate" and args.preset == "short-factual"
            thresholds = _candidate_thresholds(manifest, active=thresholds_active)
            manifest["runtime"]["cli_thresholds_active"] = thresholds_active
            manifest["runtime"]["cli_threshold_status"] = (
                "post_hoc_short_factual_cli_candidates"
                if thresholds_active
                else "score_only_outside_candidate_scope"
            )
            if mode == "generate":
                assert maximum is not None
                stream = runtime.generate(
                    prompt,
                    max_new_tokens=maximum,
                    temperature=args.temperature,
                    seed=args.seed,
                )
            else:
                assert completion is not None
                stream = runtime.replay(prompt, completion)

            text = ""
            last: TokenReading | None = None
            initial = _live_view(
                mode=mode,
                scope=scope,
                scope_detail=scope_detail,
                prompt=prompt,
                text=text,
                row=last,
                complete=False,
                width=console.size.width,
                thresholds=thresholds,
            )
            if terminal:
                with Live(
                    initial,
                    console=console,
                    auto_refresh=False,
                    transient=False,
                ) as live:
                    live.refresh()
                    for row in stream:
                        rows.append(row)
                        last = row
                        text += row.text
                        live.update(
                            _live_view(
                                mode=mode,
                                scope=scope,
                                scope_detail=scope_detail,
                                prompt=prompt,
                                text=text,
                                row=last,
                                complete=False,
                                width=console.size.width,
                                thresholds=thresholds,
                            ),
                            refresh=True,
                        )
                    live.update(
                        _live_view(
                            mode=mode,
                            scope=scope,
                            scope_detail=scope_detail,
                            prompt=prompt,
                            text=text,
                            row=last,
                            complete=True,
                            width=console.size.width,
                            thresholds=thresholds,
                        ),
                        refresh=True,
                    )
            else:
                for row in stream:
                    rows.append(row)
                    last = row
                    text += row.text
                console.print(
                    _live_view(
                        mode=mode,
                        scope=scope,
                        scope_detail=scope_detail,
                        prompt=prompt,
                        text=text,
                        row=last,
                        complete=True,
                        width=console.size.width,
                        thresholds=thresholds,
                    )
                )
            if args.trace:
                write_jsonl_trace(
                    args.trace,
                    manifest,
                    _rows_for_scope(rows, thresholds_active=thresholds_active),
                )
                console.print(f"trace  {Path(args.trace).resolve()}", style="dim")
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if terminal and rows:
        console.print(
            Text.from_markup(
                f"[bold {GOLD}]←/→[/] inspect tokens   "
                f"[bold {GOLD}]q/Esc[/] close"
            )
        )
        return run_recording_tui(
            _live_recording(
                mode,
                prompt,
                rows,
                manifest,
                thresholds=thresholds,
            ),
            console=console,
        )
    return 0


def command_generate(args: argparse.Namespace) -> int:
    return _run_live(args, "generate", args.prompt)


def command_replay(args: argparse.Namespace) -> int:
    return _run_live(args, "replay", args.prompt, args.completion)


def command_doctor(args: argparse.Namespace) -> int:
    status = 0
    print("LLM-Oscilloscope doctor")
    print(f"  platform:     {platform.system()} {platform.machine()}")
    config = load_cli_config()
    print(f"  profile:      {config['profile']}")
    try:
        root = find_artifacts_root(args.artifacts)
        QwenResearchChannels(root)
        print(f"  artifacts:    OK ({root})")
    except (FileNotFoundError, ValueError) as error:
        print(f"  artifacts:    FAIL ({error})")
        status = 1

    for module in ("torch", "transformers", "accelerate", "bitsandbytes"):
        found = importlib.util.find_spec(module) is not None
        print(f"  {module:<12} {'OK' if found else 'missing'}")
        if not found:
            status = 1
    if importlib.util.find_spec("torch") is not None:
        import torch

        print(f"  CUDA:         {'available' if torch.cuda.is_available() else 'not available'}")
        if torch.cuda.is_available():
            capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
            print(f"  GPU:          {torch.cuda.get_device_name(0)}")
            print(f"  capability:   {capability[0]}.{capability[1]}")
            print(f"  torch CUDA:   {torch.version.cuda or 'unknown'}")
        else:
            capability = None
        issue = _llm_int8_hardware_issue(bool(torch.cuda.is_available()), capability)
        print(f"  LLM.int8:     {'OK' if issue is None else 'FAIL (' + issue + ')'}")
        if issue is not None:
            status = 1
    if importlib.util.find_spec("bitsandbytes") is not None:
        try:
            import bitsandbytes

            print(f"  bnb version:  {getattr(bitsandbytes, '__version__', 'unknown')}")
        except (ImportError, OSError, RuntimeError) as error:
            print(f"  bnb import:   FAIL ({error})")
            status = 1
    print("  demos:        OK" if list_recordings() else "  demos:        FAIL")
    return status


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifacts", type=Path, help="override artifacts directory")
    parser.add_argument("--quantization", choices=("8bit", "none"), default="8bit")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-gpu-memory", default="6200MiB")
    parser.add_argument("--max-cpu-memory", default="48GiB")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--trace", type=Path, help="write a JSONL trace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-oscilloscope",
        description="Research-preview CLI for tokenwise internal Qwen measurements.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    samples = subparsers.add_parser("samples", help="list recorded GPU demos")
    samples.add_argument("--json", action="store_true")
    samples.set_defaults(func=command_samples)

    demo = subparsers.add_parser("demo", help="replay a compact recorded GPU trace")
    demo.add_argument("sample")
    demo.add_argument("--json", action="store_true")
    display = demo.add_mutually_exclusive_group()
    display.add_argument("--interactive", action="store_true", help="require token TUI")
    display.add_argument("--static", action="store_true", help="print without token TUI")
    demo.set_defaults(func=command_demo)

    generate = subparsers.add_parser("generate", help="instrument a free Qwen generation")
    generate.add_argument("prompt")
    generate.add_argument(
        "--preset", choices=("short-factual", "freeform"), default="short-factual"
    )
    generate.add_argument("--max-new-tokens", type=int)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--seed", type=int, default=0)
    _add_runtime_arguments(generate)
    generate.set_defaults(func=command_generate)

    replay = subparsers.add_parser("replay", help="force and instrument a supplied completion")
    replay.add_argument("--prompt", required=True)
    replay.add_argument("--completion", required=True)
    _add_runtime_arguments(replay)
    replay.set_defaults(func=command_replay)

    doctor = subparsers.add_parser("doctor", help="check artifacts and optional runtime")
    doctor.add_argument("--artifacts", type=Path)
    doctor.set_defaults(func=command_doctor)
    add_sycophancy_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        return int(args.func(args))
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
