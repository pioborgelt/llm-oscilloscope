"""Compact token histories drawn from existing readings, without rescoring."""

from __future__ import annotations

import math
from typing import Any, Sequence

from rich.console import Group
from rich.text import Text


SIGNALS = (
    ("combined_score", "Combined"),
    ("entity_end_score", "Entity completion"),
    ("support_score", "Support risk"),
    ("appraisal_mean4", "Appraisal · 4-token mean"),
)
BLUE = "#6E9BC3"
GOLD = "#D6B45A"
# Connected edges: up=1, right=2, down=4, left=8.
LINE_CHARS = " │─╰││╭├─╯─┴╮┤┬┼"


def _connect(grid: list[list[int]], x0: int, y0: int, x1: int, y1: int) -> None:
    """Rasterize a segment with joined terminal strokes between readings."""
    y = y0
    for x in range(x0, x1):
        next_y = round(y0 + (y1 - y0) * (x + 1 - x0) / (x1 - x0))
        grid[y][x] |= 2
        grid[y][x + 1] |= 8
        while y != next_y:
            step = 1 if next_y > y else -1
            grid[y][x + 1] |= 1 if step > 0 else 4
            grid[y + step][x + 1] |= 4 if step > 0 else 1
            y += step


def signal_options(rows: Sequence[dict[str, Any]]) -> list[tuple[str, str]]:
    options = [(key, label) for key, label in SIGNALS if any(key in r for r in rows)]
    subjects = sorted({name for r in rows for name in (r.get("subject_scores") or {})})
    options.extend((f"subject:{name}", name.replace("_", " ").title()) for name in subjects)
    return options


def signal_value(row: dict[str, Any], signal: str) -> float | None:
    if signal.startswith("subject:"):
        value = (row.get("subject_scores") or {}).get(signal.split(":", 1)[1])
    elif signal == "appraisal_mean4" and not row.get("appraisal_window_ready"):
        return None  # Never fill the opening positions with future readings.
    else:
        value = row.get(signal)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def move_signal(signal: str, options: Sequence[tuple[str, str]], key: str) -> str:
    keys = [item[0] for item in options]
    if not keys:
        return signal
    index = keys.index(signal) if signal in keys else 0
    if key in {"up", "down"}:
        index = (index + (1 if key == "down" else -1)) % len(keys)
    return keys[index]


def signal_history(
    rows: Sequence[dict[str, Any]],
    selected: int,
    *,
    signal: str = "combined_score",
    width: int = 120,
    height: int = 5,
) -> Group:
    """Join readings across the available width, leaving missing values as gaps.

    Long traces retain each column's min/max envelope.
    Detector and subject scores keep their 0–1 scale. Appraisal uses an
    explicitly labelled automatic range in its original score units.
    """
    options = signal_options(rows)
    if not options:
        return Group(Text("SIGNAL HISTORY · no readings", style="dim"))
    signal = move_signal(signal, options, "")
    label = dict(options)[signal]
    selected = max(0, min(selected, len(rows) - 1))
    values = [signal_value(row, signal) for row in rows]
    value = values[selected]
    reading = "n/a" if value is None else f"{value:.4f}"
    title = Text("SIGNAL HISTORY  ", style=f"bold {BLUE}")
    title.append(f"{label} · {reading}", style=f"bold {GOLD}")
    finite = [v for v in values if v is not None]
    if not finite:
        return Group(title, Text("No complete readings yet.", style="dim"))

    auto_scale = signal == "appraisal_mean4"
    low, high = (min(finite), max(finite)) if auto_scale else (0.0, 1.0)
    if low == high:
        low, high = low - 0.5, high + 0.5
    height = max(2, height)
    columns = max(1, width - 10)
    compressed = len(rows) > columns
    bins: list[tuple[int, int, int]] = []
    selected_column = 0
    for index in range(min(len(rows), columns)):
        if compressed:
            column = index
            start, end = index * len(rows) // columns, (index + 1) * len(rows) // columns
        else:
            column = round(index * (columns - 1) / max(1, len(rows) - 1))
            start, end = index, index + 1
        bins.append((column, start, end))
        if start <= selected < end:
            selected_column = column

    def level(v: float) -> int:
        return round((min(high, max(low, v)) - low) / (high - low) * (height - 1))

    grid = [[0] * columns for _ in range(height)]
    for (x0, _, end), (x1, start, _) in zip(bins, bins[1:]):
        # Only adjacent, available readings may be connected, even in a
        # compressed trace. A missing boundary reading breaks the line.
        left, right = values[end - 1], values[start]
        if left is not None and right is not None:
            _connect(grid, x0, level(left), x1, level(right))
    marks = [[LINE_CHARS[edges] for edges in row] for row in grid]
    for column, start, end in bins:
        bucket = [v for v in values[start:end] if v is not None]
        if bucket:
            bottom, top = level(min(bucket)), level(max(bucket))
            for y in range(bottom, top + 1):
                marks[y][column] = "●" if bottom == top else "│"

    plot = Text(no_wrap=True, overflow="crop")
    for y in reversed(range(height)):
        tick = low + (high - low) * y / (height - 1)
        plot.append(f"{tick:6.2f} │", style="dim")
        for column, mark in enumerate(marks[y]):
            plot.append(mark, style=f"bold {GOLD}" if column == selected_column else BLUE)
        plot.append("\n")
    plot.append("       └", style="dim")
    for column in range(columns):
        plot.append("▲" if column == selected_column else "─", style=GOLD if column == selected_column else "dim")

    first, last = rows[0].get("position", 0), rows[-1].get("position", len(rows) - 1)
    position = rows[selected].get("position", selected)
    caption = f"tokens {first}–{last} · ▲ {position}"
    if compressed:
        caption += " · min/max per column"
    if auto_scale:
        caption += " · auto scale"
    if len(finite) < len(rows):
        caption += f" · {len(rows) - len(finite)} unavailable"
    return Group(title, plot, Text(caption, style="dim"))
