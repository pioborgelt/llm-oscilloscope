"""Portable detector heads and evidence verification."""

__version__ = "0.6.0a0"

from .detector import DetectorHeads
from .channels import ChannelReading, QwenResearchChannels
from .qwen import TransportedQwenDetector
from .support_v2 import NativeQwenSupportHead, QwenSupportChannel
from .subject import SubjectRoutingHead, TransportedSubjectRoutingHead

__all__ = [
    "__version__",
    "DetectorHeads",
    "ChannelReading",
    "QwenResearchChannels",
    "TransportedQwenDetector",
    "NativeQwenSupportHead",
    "QwenSupportChannel",
    "SubjectRoutingHead",
    "TransportedSubjectRoutingHead",
]
