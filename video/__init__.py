"""Reusable, offline video-file frame-by-frame detection for AdaptiveAOI.

This package deliberately reuses the current PatchCore image inferencer.  It
does not claim or implement a live industrial camera service.
"""

from .video_inference import VideoDetectionOptions, run_video_detection
from .video_reader import VideoInputError, VideoReader

__all__ = ["VideoDetectionOptions", "VideoInputError", "VideoReader", "run_video_detection"]
