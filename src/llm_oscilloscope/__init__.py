"""Portable detector heads and evidence verification."""

from .detector import DetectorHeads
from .qwen import TransportedQwenDetector
from .subject import SubjectRoutingHead, TransportedSubjectRoutingHead

__all__ = [
    "DetectorHeads",
    "TransportedQwenDetector",
    "SubjectRoutingHead",
    "TransportedSubjectRoutingHead",
]
