"""YOLO26 adapter, frame annotation, and local text-to-speech delivery."""

from __future__ import annotations

from threading import Lock

import cv2
import numpy as np

from visionvoice.core import Detection, FrameAssessment, MobilityReasoningPipeline, format_distance


class Yolo26Detector:
    """Lazy Ultralytics YOLO26 loader that keeps model import out of unit tests."""

    def __init__(self, weights: str, confidence: float = 0.45, image_size: int = 640, device: str = "") -> None:
        self.weights = weights
        self.confidence = confidence
        self.image_size = image_size
        self.device = device or None
        self._model = None
        self._lock = Lock()

    def _model_instance(self):
        with self._lock:
            if self._model is None:
                from ultralytics import YOLO

                self._model = YOLO(self.weights)
            return self._model

    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        model = self._model_instance()
        result = model.track(
            frame,
            persist=True,
            verbose=False,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return ()

        names = result.names
        ids = result.boxes.id
        track_ids = ids.int().cpu().tolist() if ids is not None else [None] * len(result.boxes)
        boxes = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()

        detections = []
        for bbox, confidence, class_id, track_id in zip(boxes, confidences, class_ids, track_ids):
            label = str(names[int(class_id)]).lower()
            detections.append(
                Detection(
                    label=label,
                    confidence=float(confidence),
                    bbox=tuple(float(value) for value in bbox),
                    track_id=int(track_id) if track_id is not None else None,
                )
            )
        return tuple(detections)


def analyze_frame(
    frame: np.ndarray, detector: Yolo26Detector, reasoning: MobilityReasoningPipeline
) -> FrameAssessment:
    height, width = frame.shape[:2]
    return reasoning.assess(detector.detect(frame), (width, height))


def annotate_frame(frame: np.ndarray, assessment: FrameAssessment) -> np.ndarray:
    """Add high-contrast labels for the operator dashboard, not audio guidance."""
    image = frame.copy()
    color_for = {"critical": (45, 45, 235), "caution": (20, 165, 245), "info": (70, 190, 80)}
    event_by_key = {(event.label, event.track_id): event for event in assessment.risks}
    for detection in assessment.detections:
        event = event_by_key.get((detection.label, detection.track_id))
        if event is None:
            continue
        x1, y1, x2, y2 = (int(value) for value in detection.bbox)
        color = color_for[event.severity]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.label} | {format_distance(event.distance_m)} | {event.direction}"
        cv2.putText(image, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    if assessment.primary_alert:
        cv2.rectangle(image, (0, 0), (image.shape[1], 38), (20, 20, 20), -1)
        cv2.putText(image, assessment.primary_alert, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return image

