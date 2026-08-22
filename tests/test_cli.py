from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

from llm_oscilloscope.cli import _rows_for_scope, main
from llm_oscilloscope.qwen_runtime import TokenReading
from llm_oscilloscope.recordings import get_recording, list_recordings, load_recordings


EXPECTED_CHANNELS = {
    "entity_end",
    "support_v2",
    "combined_candidate",
    "subject_routing",
}


class _SyntheticRuntime:
    def __init__(self, token):
        self.token = token

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def trace_manifest(self, mode, prompt):
        return {
            "schema_version": 1,
            "mode": mode,
            "prompt": prompt,
            "runtime": {"synthetic": True},
        }

    def generate(self, *_args, **_kwargs):
        yield self.token

    def replay(self, *_args, **_kwargs):
        yield self.token


class RecordedCliTests(unittest.TestCase):
    def test_score_only_scope_disables_alert_flags_without_changing_scores(self):
        row = TokenReading(
            position=0,
            token_id=123,
            text="Cloud",
            token_probability=0.9,
            channels={
                "entity_end_score": 0.8,
                "support_score": 0.9,
                "combined_score": 0.72,
                "entity_alert": True,
                "combined_alert": True,
                "subject_scores": {"computer_science": 0.7},
            },
        )
        scoped = _rows_for_scope([row], thresholds_active=False)[0]
        self.assertFalse(scoped.channels["entity_alert"])
        self.assertFalse(scoped.channels["combined_alert"])
        self.assertEqual(scoped.channels["combined_score"], 0.72)
        self.assertTrue(row.channels["combined_alert"])

    def test_generate_uses_compact_stream_view(self):
        token = TokenReading(
            position=0,
            token_id=123,
            text="Fleming",
            token_probability=0.8,
            channels={
                "entity_end_score": 0.6,
                "support_score": 0.2,
                "combined_score": 0.12,
                "entity_alert": True,
                "combined_alert": False,
                "subject_scores": {"biology": 0.9},
            },
        )

        output = StringIO()
        with patch(
            "llm_oscilloscope.cli._runtime",
            return_value=_SyntheticRuntime(token),
        ), redirect_stdout(output):
            status = main(["generate", "Who discovered penicillin?"])

        rendered = output.getvalue()
        self.assertEqual(status, 0)
        self.assertIn("LLM OSCILLOSCOPE", rendered)
        self.assertIn("GENERATE  ·  SYNTHETIC", rendered)
        self.assertIn("Token limit 32", rendered)
        self.assertIn("OUTPUT  Fleming", rendered)
        self.assertIn("TOKEN   00", rendered)
        self.assertNotIn("Closest available evidence match", rendered)
        self.assertIn("Research preview: Read the README before testing this.", rendered)
        self.assertIn("candidate", rendered)
        self.assertNotIn(" pos  token", rendered)

    def test_replay_header_is_not_duplicated_and_thresholds_are_score_only(self):
        token = TokenReading(
            position=0,
            token_id=123,
            text="Canberra",
            token_probability=0.8,
            channels={
                "entity_end_score": 0.6,
                "support_score": 0.8,
                "combined_score": 0.48,
                "entity_alert": True,
                "combined_alert": True,
                "subject_scores": {"biology": 0.1},
            },
        )

        output = StringIO()
        with patch(
            "llm_oscilloscope.cli._runtime",
            return_value=_SyntheticRuntime(token),
        ), redirect_stdout(output):
            status = main(
                [
                    "replay",
                    "--prompt",
                    "What is the capital of Australia?",
                    "--completion",
                    "Canberra",
                ]
            )

        rendered = output.getvalue()
        self.assertEqual(status, 0)
        self.assertIn("REPLAY", rendered)
        self.assertNotIn("REPLAY  ·  REPLAY", rendered)
        self.assertIn("score-only synthetic replay", rendered)
        self.assertIn("score only", rendered)

    def test_gallery_contains_complete_verified_qwen_recordings(self):
        samples = list_recordings()
        self.assertEqual(len(samples), 26)
        self.assertEqual(sum(len(sample["tokens"]) for sample in samples), 742)
        identifiers = [sample["id"] for sample in samples]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertNotIn("unsupported-entity-alert", identifiers)
        self.assertTrue(all(sample["provenance"]["gpu_recorded"] for sample in samples))
        self.assertTrue(
            all(
                sample["provenance"]["model_id"]
                == "Qwen/Qwen2.5-7B-Instruct"
                for sample in samples
            )
        )
        self.assertTrue(
            all(
                sample["provenance"]["model_revision"]
                == "a09a35458c702b33eeacc393d103063234e8bc28"
                for sample in samples
            )
        )
        source = load_recordings()["source"]
        self.assertEqual(
            source["showcase_counts"],
            {
                "detected_reference_inconsistent": 3,
                "retained_known_false_positive": 3,
            },
        )

    def test_showcase_has_three_fixed_pool_detections_and_three_known_failures(self):
        selected_ids = {
            "factual_patent_date",
            "factual_elkay_client",
            "factual_escape_date",
        }
        removed_ids = {
            "biology_plant_food",
            "engineering_truss",
            "economics_inflation",
            "psychology_confirmation_bias",
            "psychology_peer_pressure",
        }
        samples = {sample["id"]: sample for sample in list_recordings()}
        self.assertTrue(selected_ids <= samples.keys())
        self.assertTrue(removed_ids.isdisjoint(samples))
        for identifier in selected_ids:
            sample = samples[identifier]
            audit = sample["provenance"]["showcase_selection"]
            self.assertEqual(
                audit["category"], "detected_unsupported_reference_mismatch"
            )
            endpoint = sample["tokens"][audit["detected_entity_endpoint"]]
            self.assertTrue(endpoint["entity_alert"])
            self.assertTrue(endpoint["combined_alert"])

        retained_alerts = {
            identifier
            for identifier, sample in samples.items()
            if identifier not in selected_ids
            and any(row["combined_alert"] for row in sample["tokens"])
        }
        self.assertEqual(
            retained_alerts,
            {"biology_chloroplasts", "biology_dna", "computer_ram"},
        )
    def test_every_sample_has_original_ids_and_all_channels_per_token(self):
        for sample in list_recordings():
            self.assertEqual(set(sample["available_channels"]), EXPECTED_CHANNELS)
            self.assertEqual(
                sample["provenance"]["token_alignment"], "original_gpu_token_ids"
            )
            for row in sample["tokens"]:
                self.assertIsInstance(row["token_id"], int)
                self.assertIsInstance(row["entity_end_score"], float)
                self.assertIsInstance(row["support_score"], float)
                self.assertIsInstance(row["combined_score"], float)
                self.assertEqual(len(row["subject_scores"]), 8)
                self.assertNotIn("labels", row)
            self.assertNotIn("annotation", sample)
            self.assertNotIn("reference_answers", sample)

    def test_fraction_sample_is_the_actual_qwen_generation(self):
        sample = get_recording("mathematics_fraction")
        endpoint = sample["tokens"][-1]
        self.assertEqual(sample["generated_text"], "3/4")
        self.assertEqual(endpoint["token_id"], 19)
        self.assertEqual(endpoint["text"], "4")
        self.assertAlmostEqual(endpoint["combined_score"], 0.2198415053791531)

    def test_gallery_threshold_scope_matches_recording_kind(self):
        for sample in list_recordings():
            thresholds = sample["provenance"]["thresholds"]
            if sample["kind"].startswith("simple_"):
                self.assertEqual(
                    sample["provenance"]["threshold_status"],
                    "post_hoc_short_factual_cli_candidates",
                )
                self.assertEqual(thresholds["entity_end"], 0.35)
                self.assertAlmostEqual(
                    thresholds["combined_candidate"], 0.34419047831653277
                )
                for row in sample["tokens"]:
                    self.assertEqual(
                        row["entity_alert"],
                        row["entity_end_score"] >= thresholds["entity_end"],
                    )
                    self.assertEqual(
                        row["combined_alert"],
                        row["combined_score"] >= thresholds["combined_candidate"],
                    )
            else:
                self.assertEqual(
                    sample["provenance"]["threshold_status"],
                    "score_only_outside_candidate_scope",
                )
                self.assertIsNone(thresholds["entity_end"])
                self.assertIsNone(thresholds["combined_candidate"])
                self.assertTrue(
                    all(
                        not row["entity_alert"] and not row["combined_alert"]
                        for row in sample["tokens"]
                    )
                )

    def test_dna_tokens_are_colored_by_threshold_not_manual_labels(self):
        sample = get_recording("biology_dna")
        threshold = sample["provenance"]["thresholds"]["entity_end"]
        self.assertTrue(any(row["entity_alert"] for row in sample["tokens"]))
        self.assertTrue(
            all(
                row["entity_alert"] == (row["entity_end_score"] >= threshold)
                for row in sample["tokens"]
            )
        )
        self.assertTrue(all("labels" not in row for row in sample["tokens"]))

    def test_simple_biology_sample_uses_real_ids_and_routing(self):
        biology = get_recording("biology_chloroplasts")
        endpoint = biology["tokens"][12]
        self.assertEqual(endpoint["token_id"], 82)
        self.assertNotIn("expected_subject", endpoint)
        self.assertEqual(
            max(endpoint["subject_scores"], key=endpoint["subject_scores"].get),
            "biology",
        )

    def test_samples_and_demo_commands(self):
        output = StringIO()
        with redirect_stdout(output):
            status = main(["samples"])
        self.assertEqual(status, 0)
        self.assertIn("biology_mitochondria", output.getvalue())
        self.assertNotIn("unsupported-entity-alert", output.getvalue())

        output = StringIO()
        with redirect_stdout(output):
            status = main(["demo", "mathematics_fraction"])
        self.assertEqual(status, 0)
        self.assertIn("ENTITY CANDIDATE", output.getvalue())
        self.assertNotIn("gold:", output.getvalue())
        self.assertIn("one half plus one", output.getvalue())
        self.assertIn("quarter?", output.getvalue())
        self.assertNotIn("COMBINED CANDIDATE", output.getvalue())

    def test_demo_accepts_numbered_sample_alias(self):
        self.assertEqual(get_recording("1")["id"], "biology_mitochondria")

        output = StringIO()
        with redirect_stdout(output):
            status = main(["demo", "1", "--static"])
        self.assertEqual(status, 0)
        self.assertIn("biology_mitochondria", output.getvalue())

    def test_unknown_demo_is_a_clean_cli_error(self):
        errors = StringIO()
        with redirect_stderr(errors):
            status = main(["demo", "missing-sample"])
        self.assertEqual(status, 2)
        self.assertIn("Unknown recorded sample", errors.getvalue())

    def test_ctrl_c_exits_without_a_traceback(self):
        output = StringIO()
        with patch(
            "llm_oscilloscope.cli.command_samples",
            side_effect=KeyboardInterrupt,
        ), redirect_stdout(output):
            status = main(["samples"])
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue(), "\n")


if __name__ == "__main__":
    unittest.main()
