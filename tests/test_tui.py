from io import StringIO
import unittest

from rich.console import Console

from llm_oscilloscope.recordings import get_recording
from llm_oscilloscope.tui import _normalize_key, move_selection, recording_view


class RecordingTuiTests(unittest.TestCase):
    def test_arrow_navigation_clamps_and_quits(self):
        self.assertEqual(move_selection(0, 3, "left"), (0, False))
        self.assertEqual(move_selection(0, 3, "right"), (1, False))
        self.assertEqual(move_selection(2, 3, "right"), (2, False))
        self.assertEqual(move_selection(1, 3, "home"), (0, False))
        self.assertEqual(move_selection(1, 3, "end"), (2, False))
        self.assertEqual(move_selection(1, 3, "q"), (1, True))

    def test_common_terminal_arrow_sequences_are_normalized(self):
        self.assertEqual(_normalize_key(b"\x1b[D"), "left")
        self.assertEqual(_normalize_key(b"\x1b[C"), "right")
        self.assertEqual(_normalize_key(b"\x1b[H"), "home")
        self.assertEqual(_normalize_key(b"\x1b[F"), "end")

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
