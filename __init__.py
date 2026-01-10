"""
Clip Extraction System

A multimodal video clip extraction system for viral content detection.
"""

__version__ = "1.0.0"
__author__ = "Clip Extraction Project"

from .audio import ClipAudio
from .video import ClipVideo
from .transcribe import Transcriber
from .llm import LLM
from .main import ClipExtractor

__all__ = [
    'ClipAudio',
    'ClipVideo',
    'Transcriber',
    'LLM',
    'ClipExtractor',
]