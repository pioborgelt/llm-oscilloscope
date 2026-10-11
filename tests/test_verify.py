from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from llm_oscilloscope.verify import main
from scripts.update_manifest import ROOT, included


class VerifyCliTests(unittest.TestCase):
    def test_private_workspace_files_are_excluded_from_release_manifest(self):
        self.assertFalse(included(ROOT / "AGENTS.md"))
        self.assertFalse(included(ROOT / "admin" / "notes.md"))

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
