from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from llm_oscilloscope.verify import main
from scripts.update_manifest import ROOT, release_files


class VerifyCliTests(unittest.TestCase):
    def test_release_manifest_contains_only_tracked_release_files(self):
        listed = {
            line.split("  ", 1)[1]
            for line in (ROOT / "MANIFEST.sha256")
            .read_text(encoding="utf-8")
            .splitlines()
        }
        expected = {
            f"./{path.relative_to(ROOT).as_posix()}" for path in release_files()
        }
        self.assertEqual(listed, expected)

    def test_missing_evidence_root_is_a_clean_cli_error(self):
        errors = StringIO()
        with TemporaryDirectory() as directory, redirect_stderr(errors):
            with self.assertRaises(SystemExit) as raised:
                main(["--root", directory])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("does not contain MANIFEST.sha256", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
