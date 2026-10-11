from copy import deepcopy
from io import StringIO
import math
import unittest

from rich.console import Console

from llm_oscilloscope.signal_plot import (
    move_signal, signal_history, signal_options, signal_value,
)


def render(rows, selected=0, *, width=80, **kwargs):
    output = StringIO()
    Console(file=output, width=width, color_system=None).print(
        signal_history(rows, selected, width=width, **kwargs)
    )
    return output.getvalue()


class SignalHistoryTests(unittest.TestCase):
    def test_short_history_fills_terminal_width_with_a_connected_line(self):
        rows = [dict(combined_score=0.5), dict(combined_score=0.5)]
        for width in (40, 80, 160):
            with self.subTest(width=width):
                plotted = render(rows, 1, width=width)
                line = next(line for line in plotted.splitlines() if line.startswith("  0.50"))
                curve = line.split("│", 1)[1]
                self.assertTrue(curve.startswith("●─"))
                self.assertTrue(curve.endswith("─●"))
                self.assertEqual(set(curve[1:-1]), {"─"})
                self.assertGreaterEqual(len(line), width - 2)
                axis = next(line for line in plotted.splitlines() if "└" in line)
                self.assertEqual(axis.index("▲"), line.rindex("●"))
                self.assertTrue(all(len(line) <= width for line in plotted.splitlines()))

    def test_rising_and_falling_segments_join_the_measured_points(self):
        rows = [dict(combined_score=v) for v in (0.0, 1.0, 0.0)]
        plotted = render(rows)
        curves = [line.split("│", 1)[1] for line in plotted.splitlines() if "│" in line]
        self.assertEqual(plotted.count("●"), 3)
        for corner in "╭╮╰╯":
            self.assertIn(corner, plotted)
        self.assertTrue(all(any(char != " " for char in column) for column in zip(*curves)))

    def test_missing_middle_readings_break_the_line_at_any_density(self):
        for missing_count in (1, 200):
            with self.subTest(missing_count=missing_count):
                values = [0.0, 0.0] + [None] * missing_count + [1.0, 1.0]
                plotted = render([dict(combined_score=v) for v in values])
                curves = [line.split("│", 1)[1] for line in plotted.splitlines() if "│" in line]
                self.assertTrue(all(line[len(line) // 2] == " " for line in curves))
                self.assertIn(f"{missing_count} unavailable", plotted)

    def test_compression_keeps_isolated_peaks_and_selected_token(self):
        rows = [dict(position=i, combined_score=0.0) for i in range(200)]
        rows[101]["combined_score"] = 1.0
        original = deepcopy(rows)
        plotted = render(rows, 101, width=40)
        top = next(line for line in plotted.splitlines() if line.startswith("  1.00"))
        self.assertIn("│", top.split("│", 1)[1])
        self.assertIn("1.0000", plotted)
        self.assertIn("▲ 101", plotted)
        self.assertIn("min/max per column", " ".join(plotted.split()))
        self.assertEqual(rows, original)
        self.assertTrue(all(len(line) <= 40 for line in plotted.splitlines()))

    def test_subject_curve_keeps_one_subject_when_the_winner_changes(self):
        rows = [
            dict(subject_scores={"biology": 0.1, "physics": 0.9}),
            dict(subject_scores={"biology": 0.8, "physics": 0.2}),
        ]
        self.assertEqual(signal_value(rows[0], "subject:biology"), 0.1)
        plotted = render(rows, signal="subject:biology")
        self.assertIn("Biology · 0.1000", plotted)
        self.assertNotIn("Physics", plotted)
        self.assertIn("  1.00", plotted)
        self.assertIn("  0.00", plotted)

    def test_missing_and_nonfinite_values_are_gaps_not_zeroes(self):
        rows = [dict(combined_score=v) for v in [None, math.nan, math.inf, 0.0]]
        plotted = render(rows)
        self.assertIn("Combined · n/a", plotted)
        self.assertIn("3 unavailable", plotted)
        self.assertEqual(plotted.count("●"), 1)

    def test_appraisal_preserves_signed_units_and_never_looks_ahead(self):
        rows = [dict(appraisal_mean4=-4.0, appraisal_window_ready=i >= 3)
                for i in range(4)]
        rows.append(dict(appraisal_mean4=2.0, appraisal_window_ready=True))
        plotted = render(rows, signal="appraisal_mean4")
        self.assertIn("n/a", plotted)
        self.assertIn(" -4.00", plotted)
        self.assertIn("  2.00", plotted)
        self.assertIn("auto scale", plotted)
        self.assertIn("3 unavailable", plotted)
        self.assertIn("No complete readings yet", render(rows[:3], signal="appraisal_mean4"))

    def test_navigation_wraps_and_ignores_token_navigation(self):
        rows = [dict(combined_score=0.4, subject_scores={"physics": 0.7})]
        options = signal_options(rows)
        self.assertEqual(move_signal("combined_score", options, "down"), "subject:physics")
        self.assertEqual(move_signal("subject:physics", options, "down"), "combined_score")
        self.assertEqual(move_signal("combined_score", options, "up"), "subject:physics")
        self.assertEqual(move_signal("combined_score", options, "right"), "combined_score")

    def test_empty_constant_and_single_token_histories(self):
        self.assertIn("no readings", render([]))
        self.assertIn("tokens 0–0", render([dict(combined_score=0.5)]))
        self.assertIn("  1.00", render([dict(combined_score=0.5)]))
        constant = render([dict(appraisal_mean4=-2.0, appraisal_window_ready=True)],
                          signal="appraisal_mean4")
        self.assertIn("-2.0000", constant)
        self.assertIn("auto scale", constant)


if __name__ == "__main__":
    unittest.main()
