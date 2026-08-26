"""Risk reasoning and alert planning independent from the UI or YOLO runtime."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from math import radians, tan
from threading import Lock
from time import monotonic
from typing import Deque, Iterable, Optional


VEHICLES = {"car", "motorcycle", "bus", "truck", "bicycle", "skateboard"}
SIGNS = {"traffic light", "stop sign"}
HARMFUL_TOOLS = {"knife", "scissors", "baseball bat"}
HAZARDS = VEHICLES | HARMFUL_TOOLS

# These values prioritise possible collision hazards. They are heuristics, not a
# substitute for a certified safety model or a measurement of distance.
BASE_RISK = {
    "car": 0.70,
    "motorcycle": 0.75,
    "bus": 0.80,
    "truck": 0.85,
    "bicycle": 0.65,
    "skateboard": 0.55,
    "person": 0.45,
    "dog": 0.50,
    "bench": 0.45,
    "chair": 0.50,
    "potted plant": 0.40,
    "suitcase": 0.55,
    "backpack": 0.35,
    "traffic light": 0.12,
    "stop sign": 0.20,
    "knife": 0.95,
    "scissors": 0.75,
    "baseball bat": 0.70,
}

# Approximate physical dimensions used by the monocular distance estimator.
# The tuple is (typical dimension in metres, matching bounding-box axis). These
# are only priors: pose, object variants, occlusion, and camera calibration can
# make a real object substantially different from the estimate.
OBJECT_SIZE_PRIORS = {
    "person": (1.70, "height"), "bicycle": (0.60, "height"), "car": (1.80, "width"),
    "motorcycle": (0.75, "height"), "airplane": (35.0, "width"), "bus": (2.55, "width"),
    "train": (3.00, "width"), "truck": (2.50, "width"), "boat": (3.00, "width"),
    "traffic light": (0.30, "width"), "fire hydrant": (0.30, "height"), "stop sign": (0.75, "width"),
    "parking meter": (0.30, "width"), "bench": (1.50, "width"), "bird": (0.30, "width"),
    "cat": (0.35, "height"), "dog": (0.55, "height"), "horse": (0.60, "height"),
    "sheep": (0.70, "height"), "cow": (0.80, "height"), "elephant": (2.50, "height"),
    "bear": (1.10, "height"), "zebra": (0.60, "height"), "giraffe": (0.80, "width"),
    "backpack": (0.35, "width"), "umbrella": (1.00, "width"), "handbag": (0.35, "width"),
    "tie": (0.10, "width"), "suitcase": (0.45, "width"), "frisbee": (0.27, "width"),
    "skis": (0.10, "width"), "snowboard": (0.30, "width"), "sports ball": (0.22, "width"),
    "kite": (1.00, "width"), "baseball bat": (0.07, "width"), "baseball glove": (0.20, "width"),
    "skateboard": (0.20, "width"), "surfboard": (0.50, "width"), "tennis racket": (0.25, "width"),
    "bottle": (0.07, "width"), "wine glass": (0.08, "width"), "cup": (0.08, "width"),
    "fork": (0.025, "width"), "knife": (0.025, "width"), "spoon": (0.04, "width"),
    "bowl": (0.20, "width"), "banana": (0.04, "width"), "apple": (0.08, "width"),
    "sandwich": (0.10, "width"), "orange": (0.08, "width"), "broccoli": (0.10, "width"),
    "carrot": (0.03, "width"), "hot dog": (0.04, "width"), "pizza": (0.30, "width"),
    "donut": (0.10, "width"), "cake": (0.25, "width"), "chair": (0.45, "width"),
    "couch": (2.00, "width"), "potted plant": (0.35, "width"), "bed": (1.50, "width"),
    "dining table": (1.20, "width"), "toilet": (0.40, "width"), "tv": (0.80, "width"),
    "laptop": (0.32, "width"), "mouse": (0.06, "width"), "remote": (0.05, "width"),
    "keyboard": (0.45, "width"), "cell phone": (0.07, "width"), "microwave": (0.50, "width"),
    "oven": (0.60, "width"), "toaster": (0.30, "width"), "sink": (0.50, "width"),
    "refrigerator": (0.80, "width"), "book": (0.15, "width"), "clock": (0.25, "width"),
    "vase": (0.15, "width"), "scissors": (0.08, "width"), "teddy bear": (0.25, "width"),
    "hair drier": (0.08, "width"), "toothbrush": (0.02, "width"),
}


def format_distance(distance_m: Optional[float]) -> str:
    """Return a stable, speech-friendly range label."""
    if distance_m is None:
        return "range unavailable"
    rounded = round(distance_m * 2) / 2
    if rounded >= 100:
        return "over 100 metres"
    unit = "metre" if rounded == 1 else "metres"
    return f"about {rounded:g} {unit}"


@dataclass(frozen=True)
class Detection:
    """One tracked detection in pixel coordinates (x1, y1, x2, y2)."""

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    track_id: Optional[int] = None


@dataclass(frozen=True)
class RiskEvent:
    label: str
    track_id: Optional[int]
    direction: str
    proximity: str
    severity: str
    score: float
    approaching: bool
    distance_m: Optional[float]
    bbox: tuple[float, float, float, float]

    @property
    def is_vehicle(self) -> bool:
        return self.label in VEHICLES

    @property
    def is_hazard(self) -> bool:
        return self.label in HAZARDS


class DistanceEstimator:
    """Pinhole-camera range estimate using a calibrated horizontal field of view."""

    def __init__(self, horizontal_fov_degrees: float = 70.0) -> None:
        if not 35.0 <= horizontal_fov_degrees <= 120.0:
            raise ValueError("horizontal_fov_degrees must be between 35 and 120")
        self.horizontal_fov_degrees = horizontal_fov_degrees

    def estimate(self, detection: Detection, frame_size: tuple[int, int]) -> Optional[float]:
        prior = OBJECT_SIZE_PRIORS.get(detection.label)
        if prior is None:
            return None
        real_dimension_m, axis = prior
        width, _ = frame_size
        x1, y1, x2, y2 = detection.bbox
        pixels = abs(x2 - x1) if axis == "width" else abs(y2 - y1)
        if pixels < 2:
            return None
        focal_length_px = (width / 2) / tan(radians(self.horizontal_fov_degrees) / 2)
        distance_m = real_dimension_m * focal_length_px / pixels
        # Avoid presenting implausible numerical precision or an unhelpful huge value.
        return round(min(max(distance_m, 0.1), 100.0), 2)


@dataclass(frozen=True)
class FrameAssessment:
    timestamp: datetime
    detections: tuple[Detection, ...]
    risks: tuple[RiskEvent, ...]
    scene_summary: str
    primary_alert: Optional[str]


class RiskAssessmentAgent:
    """Turns detections into explainable, conservative risk events."""

    def __init__(self, horizontal_fov_degrees: float = 70.0) -> None:
        self._areas: dict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=5))
        self.distance_estimator = DistanceEstimator(horizontal_fov_degrees)

    @staticmethod
    def _direction(box: tuple[float, float, float, float], width: int) -> str:
        midpoint = (box[0] + box[2]) / 2
        relative_x = midpoint / max(width, 1)
        if relative_x < 0.20:
            return "far left"
        if relative_x < 0.40:
            return "left"
        if relative_x > 0.80:
            return "far right"
        if relative_x > 0.60:
            return "right"
        return "ahead"

    @staticmethod
    def _proximity(
        box: tuple[float, float, float, float],
        width: int,
        height: int,
        distance_m: Optional[float],
    ) -> tuple[str, float]:
        if distance_m is not None:
            if distance_m <= 2.0:
                return "immediate", 1.0
            if distance_m <= 4.0:
                return "nearby", 0.72
            if distance_m <= 12.0:
                return "in view", 0.38
            return "distant", 0.18
        x1, y1, x2, y2 = box
        relative_area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(width * height, 1)
        bottom = y2 / max(height, 1)
        if relative_area >= 0.18 or (relative_area >= 0.08 and bottom >= 0.86):
            return "immediate", 1.0
        if relative_area >= 0.045 or bottom >= 0.76:
            return "nearby", 0.72
        return "in view", 0.38

    def _is_approaching(self, detection: Detection, width: int, height: int) -> bool:
        if detection.track_id is None or detection.label not in VEHICLES:
            return False
        x1, y1, x2, y2 = detection.bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(width * height, 1)
        history = self._areas[detection.track_id]
        previous = sum(history) / len(history) if history else 0.0
        history.append(area)
        # Require several observations and meaningful scale growth before saying
        # "approaching". A single frame cannot establish object motion.
        return len(history) >= 3 and area > previous * 1.18

    def analyze(self, detections: Iterable[Detection], frame_size: tuple[int, int]) -> tuple[RiskEvent, ...]:
        width, height = frame_size
        events: list[RiskEvent] = []
        for detection in detections:
            distance_m = self.distance_estimator.estimate(detection, frame_size)
            proximity, proximity_factor = self._proximity(detection.bbox, width, height, distance_m)
            score = min(
                1.0,
                BASE_RISK.get(detection.label, 0.24) * (0.35 + 0.65 * proximity_factor)
                + 0.10 * max(0.0, min(1.0, detection.confidence)),
            )
            severity = "critical" if score >= 0.76 else "caution" if score >= 0.52 else "info"
            events.append(
                RiskEvent(
                    label=detection.label,
                    track_id=detection.track_id,
                    direction=self._direction(detection.bbox, width),
                    proximity=proximity,
                    severity=severity,
                    score=round(score, 2),
                    approaching=self._is_approaching(detection, width, height),
                    distance_m=distance_m,
                    bbox=detection.bbox,
                )
            )
        return tuple(sorted(events, key=lambda event: event.score, reverse=True))


class SceneUnderstandingAgent:
    """Produces a compact scene statement from mobility-relevant objects."""

    @staticmethod
    def summarize(events: Iterable[RiskEvent]) -> str:
        event_list = list(events)
        if not event_list:
            return "No mobility-relevant objects detected. Keep using primary mobility aids."

        counts: dict[str, int] = defaultdict(int)
        for event in event_list:
            counts[event.label] += 1
        items = [f"{count} {label}{'' if count == 1 else 's'}" for label, count in sorted(counts.items())]
        summary = ", ".join(items[:4])
        if any(event.label in SIGNS for event in event_list):
            summary += "; traffic control detected"
        return summary.capitalize() + "."


class AlertPlanningAgent:
    """Chooses the single most useful alert and suppresses repeated speech."""

    def __init__(self, cooldown_seconds: float = 3.0) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._last_alert = ""
        self._last_sent_at = float("-inf")
        self._last_was_normal = False

    @staticmethod
    def _message(event: RiskEvent) -> str:
        article = "An" if event.label[0].lower() in "aeiou" else "A"
        object_name = event.label
        if event.is_vehicle and event.approaching:
            return f"{article} {object_name} is approaching {event.direction}, {format_distance(event.distance_m)}."
        if event.severity == "critical":
            return f"Warning: {article} {object_name} is very close {event.direction}, {format_distance(event.distance_m)}."
        if event.severity == "caution":
            return f"{article} {object_name} is nearby {event.direction}, {format_distance(event.distance_m)}."
        return f"{article} {object_name} is present {event.direction}, {format_distance(event.distance_m)}."

    def plan(
        self,
        events: Iterable[RiskEvent],
        now: Optional[float] = None,
        narrate_all: bool = False,
        max_objects: int = 3,
    ) -> Optional[str]:
        all_events = list(events)
        candidates = all_events if narrate_all else [event for event in all_events if event.severity in {"critical", "caution"}]
        
        is_normal_scene = False
        if not candidates:
            ranked_all = sorted(all_events, key=lambda event: -event.score)
            selected_all = ranked_all if max_objects <= 0 else ranked_all[:max_objects]
            if selected_all:
                message = " ".join(self._message(event) for event in selected_all)
                message = f"Normal scene. {message} Go."
            else:
                message = "Normal scene. No objects detected. Go."
            is_normal_scene = True
        else:
            ranked = sorted(candidates, key=lambda event: (not event.is_hazard, -event.score))
            selected = ranked if max_objects <= 0 else ranked[:max_objects]
            if narrate_all:
                has_hazard = any(event.is_hazard for event in selected)
                message = " ".join(self._message(event) for event in selected)
                if not has_hazard:
                    message = f"Normal scene. {message} Go."
                    is_normal_scene = True
            else:
                message = " ".join(self._message(event) for event in selected)
                
        timestamp = monotonic() if now is None else now
        
        if is_normal_scene:
            if self._last_was_normal and (timestamp - self._last_sent_at < 3.0):
                return None
            else:
                self._last_alert = message
                self._last_sent_at = timestamp
                self._last_was_normal = True
                return message

        is_new = message != self._last_alert
        selected_is_critical = bool(candidates) and selected[0].severity == "critical"
        critical_override = selected_is_critical and timestamp - self._last_sent_at >= self.cooldown_seconds
        
        # A changed scene needs immediate narration. Repeating an unchanged
        # critical hazard is intentionally allowed after the configured interval.
        if is_new or critical_override:
            self._last_alert = message
            self._last_sent_at = timestamp
            self._last_was_normal = False
            return message
        return None


class MobilityReasoningPipeline:
    """Coordinates scene, risk, and alert agents for one video stream."""

    def __init__(
        self,
        cooldown_seconds: float = 3.0,
        horizontal_fov_degrees: float = 70.0,
        narrate_all: bool = False,
        max_voice_objects: int = 3,
    ) -> None:
        self.risk_agent = RiskAssessmentAgent(horizontal_fov_degrees)
        self.scene_agent = SceneUnderstandingAgent()
        self.alert_agent = AlertPlanningAgent(cooldown_seconds)
        self.narrate_all = narrate_all
        self.max_voice_objects = max_voice_objects

    def assess(self, detections: Iterable[Detection], frame_size: tuple[int, int]) -> FrameAssessment:
        detection_list = tuple(detections)
        risks = self.risk_agent.analyze(detection_list, frame_size)
        return FrameAssessment(
            timestamp=datetime.now(timezone.utc),
            detections=detection_list,
            risks=risks,
            scene_summary=self.scene_agent.summarize(risks),
            primary_alert=self.alert_agent.plan(
                risks,
                narrate_all=self.narrate_all,
                max_objects=self.max_voice_objects,
            ),
        )


class LiveState:
    """Thread-safe live feed shared by the broadcaster and remote viewers."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._latest: Optional[FrameAssessment] = None
        self._alerts: Deque[tuple[int, datetime, str]] = deque(maxlen=12)
        self._alert_sequence = 0
        self._preview_jpeg: Optional[bytes] = None

    def update(self, assessment: FrameAssessment, preview_jpeg: Optional[bytes] = None) -> None:
        with self._lock:
            self._latest = assessment
            if preview_jpeg is not None:
                self._preview_jpeg = preview_jpeg
            if assessment.primary_alert:
                self._alert_sequence += 1
                self._alerts.appendleft((self._alert_sequence, assessment.timestamp, assessment.primary_alert))

    def snapshot(
        self,
    ) -> tuple[Optional[FrameAssessment], tuple[tuple[int, datetime, str], ...], Optional[bytes]]:
        with self._lock:
            return self._latest, tuple(self._alerts), self._preview_jpeg

    def clear(self) -> None:
        with self._lock:
            self._latest = None
            self._alerts.clear()
            self._preview_jpeg = None
