"""Portable detector heads and evidence verification."""

from .detector import DetectorHeads
from .qwen import TransportedQwenDetector
from .support_v2 import NativeQwenSupportHead, QwenSupportChannel
from .subject import SubjectRoutingHead, TransportedSubjectRoutingHead

__all__ = [
    "DetectorHeads",
    "TransportedQwenDetector",
    "NativeQwenSupportHead",
    "QwenSupportChannel",
    "SubjectRoutingHead",
    "TransportedSubjectRoutingHead",
]
