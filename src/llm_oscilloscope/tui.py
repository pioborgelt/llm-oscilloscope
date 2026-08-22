"""Rich terminal explorer for compact recorded GPU traces."""

from __future__ import annotations

import os
import select
import sys
from contextlib import contextmanager
from typing import Any, Iterator, TextIO

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.padding import Padding
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from . import __version__


NAVY = "#294A70"
NAVY_BRIGHT = "#6E9BC3"
NAVY_DARK = "#18324D"
GOLD = "#D6B45A"
GOLD_SOFT = "#B9903D"
IVORY = "#F0E5CE"
AMBER = "#D98245"


SUBJECT_LABELS = {
    "mathematics": "Mathematics",
    "physics": "Physics",
    "chemistry": "Chemistry",
    "biology": "Biology",
    "computer_science": "Computer Science",
    "engineering": "Engineering",
    "economics_business": "Economics / Business",
    "psychology_social_science": "Psychology / Social Science",
}


def _piece(row: dict[str, Any]) -> str:
    value = row.get("text")
    if value is None:
        return f"position {row['position']}"
    return value.replace(" ", "␠").replace("\n", "↵").replace("\t", "⇥") or "∅"


def _score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _meter(
    value: float,
    color: str,
    *,
    threshold: float | None = None,
    width: int = 22,
) -> Text:
    """Render a measurement marker without implying a percentage bar."""

    bounded = min(1.0, max(0.0, float(value)))
    value_at = round(bounded * (width - 1))
    characters = ["─"] * width
    threshold_at = None
    if threshold is not None:
        threshold_at = round(min(1.0, max(0.0, threshold)) * (width - 1))
        characters[threshold_at] = "┆"
    characters[value_at] = "◆" if value_at == threshold_at else "●"
    meter = Text("".join(characters), style="dim")
    if threshold_at is not None and threshold_at != value_at:
        meter.stylize(GOLD_SOFT, threshold_at, threshold_at + 1)
    meter.stylize(f"bold {color}", value_at, value_at + 1)
    return meter


def _timeline(sample: dict[str, Any], selected: int, *, window: int = 15) -> Text:
    timeline = Text()
    tokens = sample["tokens"]
    start = max(0, selected - window // 2)
    end = min(len(tokens), start + window)
    start = max(0, end - window)
    if start:
        timeline.append("…  ", style="dim")
    for index in range(start, end):
        row = tokens[index]
        if index > start:
            timeline.append(" ── ", style=NAVY)
        piece = _piece(row)
        label = f"{index:02}·{piece}"
        style = IVORY
        if row.get("combined_alert"):
            style = f"bold {AMBER}"
        elif row.get("entity_alert"):
            style = NAVY_BRIGHT
        elif row.get("switch_marker"):
            style = GOLD_SOFT
        if index == selected:
            style = f"bold #101820 on {GOLD}"
            label = f" {label} "
        timeline.append(label, style=style)
    if end < len(tokens):
        timeline.append("  …", style="dim")
    return timeline


def _section(label: str, content: RenderableType, color: str) -> Group:
    return Group(
        Text(label.upper(), style=f"bold {color}"),
        Padding(content, (0, 0, 0, 2)),
    )


def _detector_channels(row: dict[str, Any], sample: dict[str, Any]) -> Group:
    table = Table.grid(padding=(0, 1))
    table.add_column(no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column(justify="right", no_wrap=True)
    thresholds = sample.get("provenance", {}).get("thresholds", {})
    threshold_status = sample.get("provenance", {}).get("threshold_status")
    downstream_active = bool(row.get("entity_alert"))
    channel_rows = (
        (
            "Entity completion",
            row.get("entity_end_score"),
            thresholds.get("entity_end"),
            NAVY_BRIGHT,
            True,
        ),
        (
            "Support risk",
            row.get("support_score"),
            None,
            NAVY_BRIGHT,
            downstream_active,
        ),
        (
            "Combined candidate",
            row.get("combined_score"),
            thresholds.get("combined_candidate"),
            AMBER,
            downstream_active,
        ),
    )
    for label, value, threshold, color, active in channel_rows:
        label_cell = Text(label, style=None if active else "dim")
        if value is None:
            table.add_row(label_cell, Text("n/a", style="dim"), "")
            continue
        crossed = threshold is not None and value >= threshold
        suffix = Text(f"{value:.4f}", style=None if active else "dim")
        if threshold is not None:
            relation = "≥" if crossed else "<"
            relation_style = f"bold {AMBER}" if crossed else "dim"
            suffix.append(f"  {relation} {threshold:.3f}", style=relation_style)
        if not active:
            inactive_label = (
                "score only"
                if threshold_status == "score_only_outside_candidate_scope"
                else "inactive"
            )
            suffix.append(f"  {inactive_label}", style="dim italic")
        table.add_row(
            label_cell,
            _meter(value, color if active else "grey50", threshold=threshold),
            suffix,
        )
    contents: list[RenderableType] = [table]
    if threshold_status == "post_hoc_short_factual_cli_candidates":
        contents.append(
            Text(
                "short factual · post-hoc candidate thresholds",
                style="dim",
            )
        )
    elif threshold_status == "score_only_outside_candidate_scope":
        contents.append(
            Text("score only · candidate thresholds disabled", style="dim")
        )
    return _section("Signals", Group(*contents), NAVY_BRIGHT)


def _subject_channels(row: dict[str, Any]) -> Group:
    scores = row.get("subject_scores") or {}
    table = Table.grid(padding=(0, 1))
    table.add_column(no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column(justify="right", no_wrap=True)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
    for index, (name, value) in enumerate(ordered):
        style = NAVY_BRIGHT if index == 0 else NAVY
        label = SUBJECT_LABELS.get(name, name)
        table.add_row(label, _meter(value, style, width=18), f"{value:.4f}")
    return _section("Subject routing", table, NAVY_BRIGHT)


def _threshold_panel(row: dict[str, Any]) -> Group:
    flags = []
    if row.get("entity_alert"):
        flags.append(("entity candidate", NAVY_BRIGHT))
    if row.get("combined_alert"):
        flags.append(("combined candidate", AMBER))
    line = Text()
    for index, (label, color) in enumerate(flags):
        if index:
            line.append("   /   ", style="dim")
        line.append(label, style=f"bold {color}")
    return _section("Threshold state", line, GOLD_SOFT)


def _selected_line(sample: dict[str, Any], selected: int) -> Text:
    row = sample["tokens"][selected]
    token = repr(row.get("text")) if row.get("text") is not None else "not retained"
    line = Text("SELECTED  ", style=f"bold {NAVY_BRIGHT}")
    line.append(f"{selected:02}  {token}", style=f"bold {GOLD}")
    if row.get("token_id") is not None:
        line.append(f"   ·   id {row['token_id']}", style="dim")
    if row.get("token_probability") is not None:
        line.append(f"   ·   p(tok) {row['token_probability']:.4f}", style="dim")
    return line


def _meaningful_threshold_state(row: dict[str, Any]) -> RenderableType | None:
    if not (row.get("entity_alert") or row.get("combined_alert")):
        return None
    return _threshold_panel(row)


def recording_view(
    sample: dict[str, Any], selected: int, *, width: int = 120
) -> RenderableType:
    """Build a complete Rich view for one selected recording row."""

    row = sample["tokens"][selected]
    masthead = Table.grid(expand=True)
    masthead.add_column()
    masthead.add_column(justify="right")
    title = Text("LLM OSCILLOSCOPE", style=f"bold {GOLD}")
    title.append(f"  v{__version__}", style="dim")
    masthead.add_row(title, Text(sample["id"], style="dim"))

    context = Table.grid(padding=(0, 2))
    context.add_column(style="dim", no_wrap=True)
    context.add_column()
    context.add_row("PROMPT", Text(sample["prompt"]))
    context.add_row("OUTPUT", Text(sample["generated_text"], style=f"bold {IVORY}"))

    token_rail = Group(
        Text("TOKENS", style=f"bold {GOLD_SOFT}"),
        Padding(_timeline(sample, selected), (0, 0, 0, 2)),
    )

    available = set(sample.get("available_channels", []))
    channel_panels: list[RenderableType] = []
    if {"entity_end", "support_v2", "combined_candidate"} & available:
        channel_panels.append(_detector_channels(row, sample))
    if "subject_routing" in available:
        channel_panels.append(_subject_channels(row))
    channel_panel = Group(*channel_panels)
    footer = Text.from_markup(
        f"[bold {GOLD}]←/→[/] token   "
        f"[bold {NAVY_BRIGHT}]Home/End[/] jump   "
        f"[bold {GOLD}]q/Esc[/] quit"
    )
    content: list[RenderableType] = [
        masthead,
        Rule(characters="─", style=NAVY),
        context,
        Text(""),
        token_rail,
        Text(""),
        _selected_line(sample, selected),
        channel_panel,
    ]
    threshold_state = _meaningful_threshold_state(row)
    if threshold_state is not None:
        content.append(threshold_state)
    content.extend((Rule(characters="─", style=NAVY), footer))
    return Group(*content)


def move_selection(selected: int, count: int, key: str) -> tuple[int, bool]:
    """Apply one normalized navigation key and return (selection, should_quit)."""

    if key in {"q", "Q", "escape", "ctrl-c"}:
        return selected, True
    if key in {"left", "h"}:
        return max(0, selected - 1), False
    if key in {"right", "l"}:
        return min(count - 1, selected + 1), False
    if key == "home":
        return 0, False
    if key == "end":
        return count - 1, False
    return selected, False


def _normalize_key(data: bytes) -> str:
    mapping = {
        b"\x1b[D": "left",
        b"\x1bOD": "left",
        b"\x1b[C": "right",
        b"\x1bOC": "right",
        b"\x1b[H": "home",
        b"\x1bOH": "home",
        b"\x1b[1~": "home",
        b"\x1b[F": "end",
        b"\x1bOF": "end",
        b"\x1b[4~": "end",
        b"\x1b": "escape",
        b"\x03": "ctrl-c",
    }
    if data in mapping:
        return mapping[data]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return "unknown"


@contextmanager
def _terminal_input(stream: TextIO) -> Iterator[None]:
    if os.name == "nt":
        yield
        return
    import termios
    import tty

    descriptor = stream.fileno()
    previous = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        yield
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def _read_key(stream: TextIO) -> str:
    if os.name == "nt":
        import msvcrt

        first = msvcrt.getwch()
        if first in {"\x00", "\xe0"}:
            return {"K": "left", "M": "right", "G": "home", "O": "end"}.get(
                msvcrt.getwch(), "unknown"
            )
        return _normalize_key(first.encode("utf-8"))

    descriptor = stream.fileno()
    data = os.read(descriptor, 1)
    if data == b"\x1b":
        while len(data) < 8 and select.select([descriptor], [], [], 0.025)[0]:
            data += os.read(descriptor, 1)
    return _normalize_key(data)


def run_recording_tui(
    sample: dict[str, Any],
    *,
    console: Console | None = None,
    input_stream: TextIO | None = None,
) -> int:
    """Run the full-screen token explorer until the user quits."""

    console = console or Console()
    input_stream = input_stream or sys.stdin
    selected = 0
    with _terminal_input(input_stream):
        with Live(
            recording_view(sample, selected, width=console.size.width),
            console=console,
            screen=True,
            auto_refresh=False,
            transient=True,
        ) as live:
            live.refresh()
            while True:
                selected, should_quit = move_selection(
                    selected, len(sample["tokens"]), _read_key(input_stream)
                )
                if should_quit:
                    return 0
                live.update(
                    recording_view(sample, selected, width=console.size.width),
                    refresh=True,
                )
