from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_oscilloscope import __version__
from llm_oscilloscope.cli import main
from llm_oscilloscope.ownership import ChoiceReading, run_assay
from llm_oscilloscope.ownership_cli import packaged_samples


class SyntheticRuntime:
    def __init__(self, **kwargs):
        self.options = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def measure(self, case):
        return run_assay(
            case,
            lambda _: ChoiceReading(0.0, 0.0, 1.0, "A", True),
            runtime={"model_id": "synthetic-test-only"},
        )


class OwnershipCliTests(unittest.TestCase):
    def test_cli_version_tracks_package_version(self):
        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit) as stopped:
            main(["--version"])
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn(__version__, out.getvalue())

    def invoke(self, arguments):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            status = main(["sycophancy", *arguments])
        return status, out.getvalue(), err.getvalue()

    def test_default_is_recorded_and_never_loads_a_model(self):
        with patch(
            "llm_oscilloscope.ownership_runtime.OwnershipRuntime",
            side_effect=AssertionError("No model allowed"),
        ):
            status, out, err = self.invoke([])
        self.assertEqual(status, 0, err)
        self.assertIn("experimental", out)
        self.assertIn("not a harm detector", out)
        self.assertIn("0/2 order pairs", out)
        self.assertEqual(err, "")

    def test_json_samples_are_complete_and_preselected(self):
        self.assertEqual(
            set(packaged_samples()),
            {"workshop", "arithmetic", "qwen3-workshop", "qwen3-arithmetic"},
        )
        for name, sid in (
            ("workshop", 3001),
            ("arithmetic", 2001),
            ("qwen3-workshop", 3001),
            ("qwen3-arithmetic", 2001),
        ):
            status, out, err = self.invoke(["--sample", name, "--json"])
            self.assertEqual(status, 0, err)
            record = json.loads(out)
            self.assertEqual(record["research_case"]["id"], sid)
            self.assertEqual(len(record["events"]), 10)
            self.assertIsNone(record["measurement"]["harm_alarm"])
            if name.startswith("qwen3-"):
                self.assertEqual(record["runtime"]["model_id"], "Qwen/Qwen3-1.7B")

    def test_replay_recomputes_imported_summary_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path, output = (
                Path(directory) / "input.json",
                Path(directory) / "saved.json",
            )
            value = deepcopy(packaged_samples()["arithmetic"])
            value["measurement"]["ownership_shift_pp"] = 999
            path.write_text(json.dumps(value))
            status, out, err = self.invoke(
                ["--recording", str(path), "--output", str(output), "--json"]
            )
            self.assertEqual(status, 0, err)
            self.assertLess(json.loads(out)["measurement"]["ownership_shift_pp"], 100)
            original = output.read_bytes()
            status, out, err = self.invoke(
                ["--recording", str(path), "--output", str(output)]
            )
            self.assertEqual(status, 2)
            self.assertIn("already exists", err)
            self.assertEqual(output.read_bytes(), original)

    def test_explicit_live_case_uses_injected_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            path.write_text(
                json.dumps(
                    {"side_a": "A fixed schedule.", "side_b": "A rotating schedule."}
                )
            )
            with patch(
                "llm_oscilloscope.ownership_runtime.OwnershipRuntime", SyntheticRuntime
            ):
                status, out, err = self.invoke(
                    ["--live", "--input", str(path), "--json"]
                )
            self.assertEqual(status, 0, err)
            self.assertEqual(json.loads(out)["measurement"]["ownership_shift_pp"], 0.0)
            self.assertIn("ten controlled", err)

    def test_invalid_options_do_not_start_gpu_work(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_parent = Path(directory) / "absent" / "result.json"
            for args in (
                ["--live"],
                ["--input", "missing.json"],
                ["--allow-download"],
                ["--model", "qwen3"],
                ["--live", "--input", "missing.json", "--output", str(missing_parent)],
            ):
                with patch(
                    "llm_oscilloscope.ownership_runtime.OwnershipRuntime",
                    side_effect=AssertionError("No model allowed"),
                ):
                    status, _, _ = self.invoke(args)
                self.assertEqual(status, 2)

    def test_bad_recordings_fail_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for value in (
                [],
                {"schema_version": 99},
                {"schema_version": True},
                {"runtime": {}},
            ):
                path.write_text(json.dumps(value))
                status, out, err = self.invoke(["--recording", str(path), "--json"])
                self.assertEqual(status, 2)
                self.assertEqual(out, "")
                self.assertTrue(err.startswith("error:"))


if __name__ == "__main__":
    unittest.main()
