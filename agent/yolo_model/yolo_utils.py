"""
YOLO utilities and simple rule engine
-------------------------------------

Used by workers to:
- load YOLO model by name/path (init_yolo_model)
- run a simple rule engine on detected class names (check_event_match)
- infer object targets from task_name (infer_object_targets)
- provide a draw_boxes placeholder for future overlays
"""
import os
import re
from typing import Optional

# Optional YOLO (used if installed)
try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # noqa: BLE001
    YOLO = None  # type: ignore[assignment]

from .model_registry import load_model


def init_yolo_model(task: dict):
    """
    Initialize a YOLO model using the simple model registry.

    Returns the model instance or None if unavailable.
    """
    model_id = (task.get("yolo_model_path") or task.get("model_id") or "").strip()
    model = load_model(model_id if model_id else None)
    if model is not None:
        print(f"✅ YOLO model loaded: {model_id or 'default'}")
    else:
        print(f"❌ Model not found in registry: '{model_id or 'default'}'")
    return model


def infer_object_targets(task: dict) -> set[str]:
    """
    Infer intended target classes from the task name using simple keyword heuristics.
    Returns a set of COCO-ish class names to match against detections.
    """
    task_name = (task.get("task_name") or "").lower()
    if not task_name:
        return set()

    tokens = re.findall(r"[a-zA-Z]+", task_name)
    mapping: dict[str, set[str]] = {
        "person": {"person"},
        "intruder": {"person"},
        "car": {"car"},
        "truck": {"truck"},
        "bus": {"bus"},
        "bicycle": {"bicycle"},
        "bike": {"bicycle", "motorcycle"},
        "motorcycle": {"motorcycle"},
        "knife": {"knife"},
        "scissor": {"scissors"},
        "scissors": {"scissors"},
        "phone": {"cell phone"},
        "mobile": {"cell phone"},
        "cell": {"cell phone"},
        "dog": {"dog"},
        "cat": {"cat"},
    }

    targets: set[str] = set()
    for token in tokens:
        if token in mapping:
            targets |= mapping[token]
    return {t.lower() for t in targets}


def check_event_match(task: dict, class_names: list[str]) -> Optional[str]:
    """
    Map task_type to a simplistic event rule using detected class names.

    Returns a human-readable event label if matched, else None.
    """
    task_type = (task.get("task_type") or "").lower()
    if not task_type:
        return None
    lower_classes = [c.lower() for c in class_names]

    if task_type == "person_activity_detection":
        if "person" in lower_classes:
            return "Person activity detected"

    if task_type == "fight_detection":
        # Naive heuristic: multiple persons in frame might indicate potential altercation
        if lower_classes.count("person") >= 2:
            return "Potential fight activity (multiple persons)"

    if task_type == "object_detection":
        # Only trigger if the detected classes include a target inferred from task_name
        targets = infer_object_targets(task)
        if not targets:
            return None
        matched = sorted(t for t in targets if t in lower_classes)
        if matched:
            return f"Object detected: {', '.join(matched)}"

    return None


def draw_boxes(frame, detections) -> "any":
    """
    Placeholder for drawing overlays on frames.
    detections is expected to carry boxes/classes/scores in the future.
    Currently returns the frame unchanged.
    """
    return frame


