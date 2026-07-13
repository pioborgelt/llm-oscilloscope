#!/usr/bin/env python3
"""Rebuild MANIFEST.sha256 from distributable repository files."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(
        part in EXCLUDED_DIRECTORIES or part.endswith(".egg-info")
        for part in relative.parts
    ):
        return False
    if path.name == "MANIFEST.sha256" or path.suffix == ".pyc":
        return False
    return path.is_file()


def main() -> None:
    files = sorted(
        (path for path in ROOT.rglob("*") if included(path)),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    lines = [
        f"{sha256(path)}  ./{path.relative_to(ROOT).as_posix()}"
        for path in files
    ]
    (ROOT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote MANIFEST.sha256 with {len(files)} files")


if __name__ == "__main__":
    main()
