"""WebRTC glue kept separate so non-camera analysis remains easy to test."""

from __future__ import annotations

import cv2

from visionvoice.core import LiveState, MobilityReasoningPipeline
from visionvoice.pipeline import Yolo26Detector, analyze_frame, annotate_frame


def make_video_processor(
    detector: Yolo26Detector,
    reasoning: MobilityReasoningPipeline,
    state: LiveState,
):
    """Return a processor class accepted by streamlit-webrtc."""

    class VisionVoiceVideoProcessor:
        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")
            assessment = analyze_frame(image, detector, reasoning)
            annotated = annotate_frame(image, assessment)
            encoded, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            state.update(assessment, jpeg.tobytes() if encoded else None)
            import av

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    return VisionVoiceVideoProcessor
