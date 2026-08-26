"""VisionVoice AI Streamlit application."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from streamlit.components.v1 import html as component_html

from visionvoice.core import FrameAssessment, LiveState, MobilityReasoningPipeline, format_distance
from visionvoice.pipeline import Yolo26Detector, analyze_frame, annotate_frame
from visionvoice.voice_agent import BrowserVoiceAgent


st.set_page_config(page_title="VisionVoice AI", page_icon="👁️", layout="wide")


@st.cache_resource(show_spinner="Loading YOLO26 detector…")
def load_detector(weights: str, confidence: float, image_size: int, device: str) -> Yolo26Detector:
    return Yolo26Detector(weights, confidence, image_size, device)


@st.cache_resource
def get_shared_state() -> LiveState:
    """One latest-frame store for every browser connected to this app server."""
    return LiveState()


def get_reasoning(
    cooldown_seconds: float,
    horizontal_fov_degrees: float,
    narrate_all: bool,
    max_voice_objects: int,
) -> MobilityReasoningPipeline:
    key = "reasoning_pipeline"
    config = (cooldown_seconds, horizontal_fov_degrees, narrate_all, max_voice_objects)
    if key not in st.session_state or st.session_state.get("reasoning_config") != config:
        st.session_state[key] = MobilityReasoningPipeline(
            cooldown_seconds=cooldown_seconds,
            horizontal_fov_degrees=horizontal_fov_degrees,
            narrate_all=narrate_all,
            max_voice_objects=max_voice_objects,
        )
        st.session_state.reasoning_config = config
    return st.session_state[key]


def local_link() -> str:
    """Build the LAN URL displayed to the broadcaster and nearby viewers."""
    port = os.environ.get("STREAMLIT_SERVER_PORT", "8501")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        try:
            connection.connect(("10.255.255.255", 1))
            host = connection.getsockname()[0]
        except OSError:
            host = "127.0.0.1"
    return f"http://{host}:{port}"


def global_link() -> Optional[str]:
    """Read the temporary public URL created by share.py, if available."""
    configured_url = os.environ.get("VISIONVOICE_PUBLIC_URL", "").strip()
    if configured_url:
        return configured_url
    url_file = Path("runtime/public-url.txt")
    if url_file.is_file():
        value = url_file.read_text(encoding="utf-8").strip()
        return value or None
    return None


def show_share_links() -> None:
    with st.expander("Share links", expanded=False):
        st.caption("Local network link")
        st.code(local_link(), language=None)




def show_dashboard(assessment: Optional[FrameAssessment], alerts) -> None:
    st.subheader("System activity")
    if assessment is None:
        st.info("Start a broadcaster or upload an image to populate the shared live feed.")
        return

    hazards = sum(event.is_hazard for event in assessment.risks)
    critical = sum(event.severity == "critical" for event in assessment.risks)
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Tracked objects", len(assessment.detections))
    metric_b.metric("Hazards", hazards)
    metric_c.metric("Critical risks", critical)

    if assessment.primary_alert:
        st.error(f"Latest spoken alert: {assessment.primary_alert}")
    else:
        st.success("No new spoken update for this frame.")
    st.caption(f"Scene: {assessment.scene_summary}")

    risk_rows = [
        {
            "Object": event.label,
            "Hazard": "Yes" if event.is_hazard else "No",
            "ID": event.track_id if event.track_id is not None else "—",
            "Position": event.direction,
            "Distance": format_distance(event.distance_m),
            "Proximity": event.proximity,
            "Risk": event.severity,
            "Approaching": "Yes" if event.approaching else "No",
        }
        for event in assessment.risks
    ]
    if risk_rows:
        st.dataframe(risk_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No objects are currently detected.")

    if alerts:
        with st.expander("Recent spoken alerts", expanded=False):
            for _, timestamp, message in alerts:
                st.write(f"{timestamp.astimezone().strftime('%H:%M:%S')} — {message}")


def analyze_uploaded_frame(detector: Yolo26Detector, reasoning: MobilityReasoningPipeline, state: LiveState) -> None:
    upload = st.file_uploader("Upload a street or indoor scene", type=["jpg", "jpeg", "png"], key="analysis_image")
    if upload is None:
        return
    bytes_array = np.frombuffer(upload.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(bytes_array, cv2.IMREAD_COLOR)
    if image is None:
        st.error("The image could not be decoded.")
        return
    if st.button("Analyze image", type="primary", use_container_width=True):
        try:
            with st.spinner("Detecting and assessing risk…"):
                assessment = analyze_frame(image, detector, reasoning)
            annotated = annotate_frame(image, assessment)
            encoded, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            state.update(assessment, jpeg.tobytes() if encoded else None)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        except Exception as exc:
            st.error(f"Unable to run YOLO26: {exc}")


def run_live_camera(detector: Yolo26Detector, reasoning: MobilityReasoningPipeline, state: LiveState) -> None:
    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
        from visionvoice.live import make_video_processor
    except ImportError:
        st.error("Live webcam support is unavailable. Install the dependencies in requirements.txt.")
        return

    st.caption("Grant camera permission, then press START. This browser becomes the broadcaster; other devices can open Viewer mode to receive its processed feed.")
    webrtc_streamer(
        key="visionvoice-live-camera",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=make_video_processor(detector, reasoning, state),
        async_processing=True,
    )


def show_remote_view(state: LiveState) -> None:
    _, _, preview_jpeg = state.snapshot()
    if preview_jpeg is None:
        st.info("Waiting for a broadcaster. Open this app on the camera device and choose Live broadcaster.")
        return
    st.image(preview_jpeg, caption="Shared processed live feed", use_container_width=True)


def main() -> None:
    st.title("VisionVoice AI")
    st.caption("Where AI becomes your eyes — shared real-time scene awareness with on-device voice alerts.")
    show_share_links()

    with st.sidebar:
        st.header("Runtime settings")
        weights = st.text_input(
            "YOLO26 weights",
            value="yolo26x.pt",
            help="Use a local custom checkpoint or an official Ultralytics YOLO26 model name.",
        )
        confidence = st.slider("Detection confidence", 0.20, 0.85, 0.25, 0.05)
        image_size = st.select_slider("Inference size", options=[416, 512, 640, 768], value=768)
        device = st.selectbox("Compute device", options=["", "cpu", "mps", "0"], index=2, format_func=lambda value: value or "Auto")
        horizontal_fov = st.slider(
            "Camera horizontal field of view (degrees)",
            35,
            120,
            70,
            1,
            help="Set this to your camera's measured horizontal FOV for better distance estimates.",
        )
        repeat_interval = st.slider("Critical hazard repeat (seconds)", 0.5, 5.0, 1.0, 0.5)
        narration_mode = st.radio(
            "Voice guidance",
            ["All detected objects (hazards first)", "Priority hazards"],
            index=0,
            help="All-object mode gives buses, bikes, vehicles, and harmful tools first priority, then reports normal objects.",
        )
        report_every_object = st.toggle(
            "Report every detected object",
            value=True,
            help="Hazards are spoken first. A long object list can delay the next spoken update.",
        )
        max_voice_objects = 0 if report_every_object else st.slider("Objects per spoken update", 1, 12, 6)
        
        st.divider()
        st.subheader("Voice configuration")
        browser_voice_enabled = st.toggle("Speak alerts on this device", value=True)
        preferred_voice = st.selectbox(
            "Preferred Voice", 
            options=["", "Google US English", "Google UK English Male", "Google UK English Female", "Samantha", "Alex", "Daniel"],
            index=1,
            help="Select a specific voice. If unavailable, it will fall back to the system default. Leave blank for default."
        )
        voice_rate = st.slider("Voice speed", 0.5, 2.0, 1.15, 0.05)
        st.divider()
        st.caption("Distance is approximate, based on camera FOV and typical object dimensions. Never treat it as a safety measurement.")
        st.caption("For safety, this complements—not replaces—a cane, guide dog, traffic signals, local orientation, or user judgment.")

    detector = load_detector(weights, confidence, image_size, device)
    reasoning = get_reasoning(
        repeat_interval,
        horizontal_fov,
        narration_mode == "All detected objects (hazards first)",
        max_voice_objects,
    )
    state = get_shared_state()

    mode = st.radio("Mode", ["Live broadcaster", "Remote viewer", "Upload a test image"], horizontal=True)
    left, right = st.columns([1.5, 1])
    with left:
        if mode == "Live broadcaster":
            run_live_camera(detector, reasoning, state)
        elif mode == "Remote viewer":
            show_remote_view(state)
        else:
            analyze_uploaded_frame(detector, reasoning, state)
    with right:
        assessment, alerts, _ = state.snapshot()
        show_dashboard(assessment, alerts)
        if st.button("Clear shared activity", use_container_width=True):
            state.clear()
            st.rerun()

    voice_agent = BrowserVoiceAgent(voice_preference=preferred_voice, rate=voice_rate)
    voice_agent.speak(alerts, browser_voice_enabled)

    # Poll the shared in-memory broadcast for the dashboard, remote preview, and on-device speech.
    if mode in {"Live broadcaster", "Remote viewer"}:
        try:
            from streamlit_autorefresh import st_autorefresh

            st_autorefresh(interval=500, key="visionvoice-live-refresh")
        except ImportError:
            st.caption("Install streamlit-autorefresh for automatic updates during live sharing.")


if __name__ == "__main__":
    main()
