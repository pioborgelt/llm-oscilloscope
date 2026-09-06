#!/usr/bin/env python3
"""Exercise an installed wheel outside its checkout, without Torch or a GPU."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import llm_oscilloscope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path)
    args = parser.parse_args()
    package = Path(llm_oscilloscope.__file__).resolve()
    assert "site-packages" in package.parts, (
        "Use a clean wheel installation, not an editable checkout"
    )
    expected_version = llm_oscilloscope.__version__
    assert version("llm-oscilloscope") == expected_version
    assert importlib.util.find_spec("torch") is None, (
        "The CPU-only smoke environment must not contain Torch"
    )
    environment = dict(os.environ)
    environment.pop("LLM_OSCILLOSCOPE_ARTIFACTS", None)
    environment.pop("PYTHONPATH", None)

    def call(module, *arguments, expected=0):
        result = subprocess.run(
            [sys.executable, "-m", module, *arguments],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == expected, (arguments, result.stdout, result.stderr)
        assert "Traceback" not in result.stderr
        return result.stdout, result.stderr

    # No repository artifacts or editable source may satisfy these checks.
    with tempfile.TemporaryDirectory(prefix="llmosci-wheel-smoke-") as directory:
        cli = "llm_oscilloscope.cli"
        out, _ = call(cli, "--version")
        assert expected_version in out
        out, _ = call(cli, "samples", "--json")
        assert len(json.loads(out)) == 26
        out, _ = call(cli, "demo", "biology_mitochondria", "--static")
        assert "LLM OSCILLOSCOPE" in out
        for sample, sid in (
            ("workshop", 3001),
            ("arithmetic", 2001),
            ("qwen3-workshop", 3001),
            ("qwen3-arithmetic", 2001),
        ):
            out, _ = call(cli, "sycophancy", "--sample", sample, "--json")
            record = json.loads(out)
            assert record["research_case"]["id"] == sid
            assert record["measurement"]["valid_answers"] == 10
            assert record["measurement"]["harm_alarm"] is None
        out, _ = call(cli, "sycophancy", "--output", "saved.json", "--json")
        saved = json.loads(out)
        out, _ = call(cli, "sycophancy", "--recording", "saved.json", "--json")
        assert saved == json.loads(out)
        call(cli, "sycophancy", "--output", "saved.json", expected=2)
        _, err = call("llm_oscilloscope.verify", expected=2)
        assert "could not locate the evidence checkout" in err
        # Resolve bundled heads in the isolated interpreter, not this script's checkout.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from llm_oscilloscope.channels import QwenResearchChannels; QwenResearchChannels()",
            ],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stderr
        if args.evidence_root is not None:
            call("llm_oscilloscope.verify", "--root", str(args.evidence_root.resolve()))
    print(
        json.dumps(
            {
                "status": "PASS",
                "version": version("llm-oscilloscope"),
                "package": str(package),
                "torch_installed": False,
                "recorded_qwen_samples": 26,
                "ownership_samples": 4,
                "explicit_evidence_check": args.evidence_root is not None,
            }
        )
    )


if __name__ == "__main__":
    main()
