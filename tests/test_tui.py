from io import StringIO
from contextlib import nullcontext
import os
import unittest
from unittest.mock import patch

from rich.console import Console

from llm_oscilloscope.recordings import get_recording
from llm_oscilloscope.tui import _normalize_key, _read_key, move_selection, recording_view, run_recording_tui


class RecordingTuiTests(unittest.TestCase):
    def test_arrow_navigation_clamps_and_quits(self):
        self.assertEqual(move_selection(0, 3, "left"), (0, False))
        self.assertEqual(move_selection(0, 3, "right"), (1, False))
        self.assertEqual(move_selection(2, 3, "right"), (2, False))
        self.assertEqual(move_selection(1, 3, "home"), (0, False))
        self.assertEqual(move_selection(1, 3, "end"), (2, False))
        self.assertEqual(move_selection(1, 3, "q"), (1, True))
        self.assertEqual(move_selection(1, 3, "right", step=5), (2, False))
        self.assertEqual(move_selection(1, 3, "left", step=5), (0, False))

    @unittest.skipIf(os.name == "nt", "POSIX terminal byte stream")
    def test_queued_arrow_repeats_and_following_keys_are_read_separately(self):
        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd) as stream:
            try:
                os.write(write_fd, b"\x1b[C" * 20 + b"\x1bOD\x1b[1~\x1b[4~\x1b[Bq")
            finally:
                os.close(write_fd)
            expected = ["right"] * 20 + ["left", "home", "end", "down", "q", "escape"]
            self.assertEqual([_read_key(stream) for _ in expected], expected)

    def test_held_navigation_accelerates_and_resets_after_pause_or_other_key(self):
        keys = ["right"] * 14 + ["right", "left", "up", "right", "q"]
        # Include the keyboard's initial repeat delay, then regular repeats.
        times = [0.0, 0.5] + [0.5 + i * 0.04 for i in range(1, 13)]
        times += [1.5, 1.54, 1.58, 1.62, 1.66]
        console = Console(file=StringIO())
        with patch("llm_oscilloscope.tui._terminal_input", return_value=nullcontext()), \
             patch("llm_oscilloscope.tui._read_key", side_effect=keys), \
             patch("llm_oscilloscope.tui.monotonic", side_effect=times), \
             patch("llm_oscilloscope.tui.Live"), \
             patch("llm_oscilloscope.tui.recording_view", return_value="") as view:
            result = run_recording_tui({"tokens": [{}] * 200}, console=console, input_stream=StringIO())
        self.assertEqual(result, 0)
        positions = [call.args[1] for call in view.call_args_list]
        steps = [end - start for start, end in zip(positions, positions[1:])]
        self.assertEqual(steps[:5], [1] * 5)
        self.assertIn(3, steps[:14])
        self.assertEqual(steps[13], 5)
        self.assertEqual(steps[-4:], [1, -1, 0, 1])

    def test_common_terminal_arrow_sequences_are_normalized(self):
        self.assertEqual(_normalize_key(b"\x1b[D"), "left")
        self.assertEqual(_normalize_key(b"\x1b[C"), "right")
        self.assertEqual(_normalize_key(b"\x1b[H"), "home")
        self.assertEqual(_normalize_key(b"\x1b[F"), "end")
        self.assertEqual(_normalize_key(b"\x1b[A"), "up")
        self.assertEqual(_normalize_key(b"\x1bOB"), "down")

    def test_compact_explorer_keeps_graph_selection_and_controls_visible(self):
        sample = get_recording("appraisal_native_study_brief")
        output = StringIO()
        Console(file=output, width=80, height=24, color_system=None).print(
            recording_view(sample, 12, width=80, height=24, signal="subject:biology")
        )
        rendered = output.getvalue()
        self.assertIn("SIGNAL HISTORY  Biology", rendered)
        self.assertIn("▲ 12", rendered)
        self.assertIn("↑/↓ signal", rendered)
        self.assertIn("q/Esc quit", rendered)
        self.assertLessEqual(len(rendered.splitlines()), 24)

    def test_interactive_keys_change_the_curve_and_then_the_token(self):
        sample = get_recording("mathematics_fraction")
        output = StringIO()
        console = Console(file=output, width=80, height=24, force_terminal=True,
                          force_interactive=True, color_system=None,
                          _environ={"TERM": "xterm-256color"})
        with patch("llm_oscilloscope.tui._terminal_input", return_value=nullcontext()), \
             patch("llm_oscilloscope.tui._read_key", side_effect=["down", "right", "q"]):
            result = run_recording_tui(sample, console=console, input_stream=StringIO())
        self.assertEqual(result, 0)
        self.assertIn("SIGNAL HISTORY  Entity completion", output.getvalue())
        self.assertIn("▲ 1", output.getvalue())

    def test_threshold_candidate_renders_all_channels_without_manual_grade(self):
        sample = get_recording("mathematics_fraction")
        output = StringIO()
        console = Console(file=output, width=120, color_system=None)
        console.print(recording_view(sample, 2, width=120))
        rendered = output.getvalue()
        self.assertIn("entity candidate", rendered)
        self.assertIn("0.2198", rendered)
        self.assertIn("SUBJECT ROUTING", rendered)
        self.assertIn("≥ 0.350", rendered)
        self.assertIn("short factual · post-hoc candidate thresholds", rendered)
        self.assertNotIn("inactive", rendered.split("SUBJECT ROUTING", 1)[0])
        self.assertNotIn("ANNOTATION", rendered)
        self.assertNotIn("unsupported entity", rendered)
        self.assertIn("one half plus one quarter", rendered)
        self.assertIn("3/4", rendered)

    def test_subject_view_also_shows_detector_readings(self):
        sample = get_recording("biology_chloroplasts")
        output = StringIO()
        console = Console(file=output, width=120, color_system=None)
        console.print(recording_view(sample, 12, width=120))
        rendered = output.getvalue()
        self.assertIn("id 82", rendered)
        self.assertIn("SUBJECT ROUTING", rendered)
        self.assertIn("Biology", rendered)
        self.assertNotIn("expected", rendered)
        self.assertIn("Entity completion", rendered)
        self.assertIn("Support risk", rendered)

    def test_support_and_combined_are_dimmed_below_entity_threshold(self):
        sample = get_recording("mathematics_fraction")
        output = StringIO()
        console = Console(file=output, width=120, color_system=None)
        console.print(recording_view(sample, 0, width=120))
        rendered = output.getvalue()
        self.assertIn("Entity completion", rendered)
        self.assertEqual(rendered.count("inactive"), 2)

    def test_live_support_v2_is_rendered_but_replay_is_score_only(self):
        sample = {
            "id": "live-replay",
            "prompt": "What is the capital of Australia?",
            "generated_text": "Canberra",
            "available_channels": [
                "entity_end",
                "support_v2",
                "combined_candidate",
                "subject_routing",
            ],
            "provenance": {
                "thresholds": {
                    "entity_end": None,
                    "combined_candidate": None,
                },
                "threshold_status": "score_only_outside_candidate_scope",
            },
            "tokens": [
                {
                    "position": 0,
                    "token_id": 1,
                    "text": "Canberra",
                    "token_probability": 1.0,
                    "entity_end_score": 0.8,
                    "support_score": 0.7,
                    "combined_score": 0.56,
                    "entity_alert": False,
                    "combined_alert": False,
                    "subject_scores": {"biology": 0.1},
                }
            ],
        }
        output = StringIO()
        Console(file=output, width=120, color_system=None).print(
            recording_view(sample, 0, width=120)
        )
        rendered = output.getvalue()
        self.assertIn("Support risk", rendered)
        self.assertIn("score only · candidate thresholds disabled", rendered)
        self.assertEqual(rendered.count("score only"), 3)
        self.assertNotIn("gated", rendered)
        self.assertNotIn("Threshold state", rendered)


if __name__ == "__main__":
    unittest.main()
