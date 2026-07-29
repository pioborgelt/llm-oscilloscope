#!/usr/bin/env python3

import argparse
from pathlib import Path

from llm_oscilloscope.subject_verify import verify_subject_routing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    verify_subject_routing(args.root.resolve())


if __name__ == "__main__":
    main()
