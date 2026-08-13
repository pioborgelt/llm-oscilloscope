#!/usr/bin/env python3
"""Verify the released Qwen-native two-channel transfer."""

from pathlib import Path

from llm_oscilloscope.qwen_verify import verify_qwen_native_transfer


if __name__ == "__main__":
    verify_qwen_native_transfer(Path(__file__).resolve().parents[1])
