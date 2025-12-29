"""
Rule: accident_presence (Human Fall Detection - KEYPOINT BASED)
---------------------------------------------------------------

Fall detection logic (robust & simple):

1. Hip sudden downward movement
2. Body height collapse
3. Lying posture confirmation
4. Must persist for N consecutive frames

Designed for:
- YOLOv8 Pose models
- Live stream / RTSP
- FPS >= 5
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import math

from agent.rule_engine.registry import register_rule


# =============================
# INTERNAL CONSTANTS
# =============================

LYING_ANGLE_THRESHOLD = 50        # degrees from vertical
HIP_DROP_THRESHOLD = 15           # pixels (sudden downward motion)
HEIGHT_DROP_RATIO = 0.25          # 25% height collapse
CONFIRM_FRAMES = 3                # frames to confirm fall
KP_CONF_THRESHOLD = 0.3           # keypoint confidence
MIN_HEIGHT_FOR_STANDING = 300     # minimum height in pixels to be considered standing


# =============================
# KEYPOINT HELPERS
# =============================

def _kp(person, idx) -> Optional[Tuple[float, float]]:
    if idx >= len(person):
        return None
    kp = person[idx]
    if kp is None or len(kp) < 2:
        return None
    if len(kp) >= 3 and kp[2] < KP_CONF_THRESHOLD:
        return None
    return float(kp[0]), float(kp[1])


def _mid(a, b):
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _angle_from_vertical(p1, p2) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return 0.0
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def _bbox_height(person) -> float:
    ys = [kp[1] for kp in person if kp and len(kp) >= 2]
    return max(ys) - min(ys) if ys else 0.0


# =============================
# CORE FALL ANALYSIS
# =============================

def _analyze(person, prev_metrics=None):
    """
    Returns:
        falling (bool) - detected falling motion
        lying (bool) - detected lying posture
        metrics (dict)
    """
    # COCO keypoints
    # 5,6 = shoulders | 11,12 = hips
    ls = _kp(person, 5)
    rs = _kp(person, 6)
    lh = _kp(person, 11)
    rh = _kp(person, 12)

    if not (ls and rs and lh and rh):
        return False, False, None

    shoulder_mid = _mid(ls, rs)
    hip_mid = _mid(lh, rh)

    height = _bbox_height(person)
    angle = _angle_from_vertical(shoulder_mid, hip_mid)

    # Lying detection: angle-based OR height-based
    # If angle is high (body horizontal) OR height is very small, person is lying
    lying = angle > LYING_ANGLE_THRESHOLD or height < MIN_HEIGHT_FOR_STANDING
    
    falling = False

    if prev_metrics:
        hip_drop = hip_mid[1] - prev_metrics["hip_y"]
        height_drop = prev_metrics["height"] - height
        prev_angle = prev_metrics.get("angle", 0)
        
        # Detect falling motion: sudden hip drop + height collapse
        if hip_drop > HIP_DROP_THRESHOLD and height_drop > (HEIGHT_DROP_RATIO * prev_metrics["height"]):
            falling = True
        
        # Also detect transition from standing to lying
        # If was standing (angle < threshold) and now lying (angle > threshold)
        if prev_angle <= LYING_ANGLE_THRESHOLD and angle > LYING_ANGLE_THRESHOLD:
            # And there's significant height drop
            if height_drop > (HEIGHT_DROP_RATIO * prev_metrics["height"]):
                falling = True

    metrics = {
        "hip_y": hip_mid[1],
        "height": height,
        "angle": angle,
    }

    return falling, lying, metrics


# =============================
# RULE ENTRY POINT
# =============================

@register_rule("accident_presence")
def evaluate_accident_presence(
    rule: Dict[str, Any],
    detections: Dict[str, Any],
    task: Dict[str, Any],
    rule_state: Dict[str, Any],
    now: datetime,
) -> Optional[Dict[str, Any]]:

    print("[accident_presence] Evaluating fall detection rule")

    # -----------------------------
    # 1. Class filter
    # -----------------------------
    target_class = (rule.get("class") or "person").lower()
    classes = [str(c).lower() for c in (detections.get("classes") or [])]

    if target_class not in classes:
        rule_state.clear()
        return None

    keypoints = detections.get("keypoints") or []
    if not keypoints:
        rule_state.clear()
        return None

    # -----------------------------
    # 2. Rule state init
    # -----------------------------
    history = rule_state.setdefault("history", {})
    fall_counter = rule_state.setdefault("fall_counter", {})

    fallen_ids = []

    # -----------------------------
    # 3. Per-person analysis
    # -----------------------------
    for idx, person in enumerate(keypoints):
        prev = history.get(idx)

        falling, lying, metrics = _analyze(person, prev)

        if metrics:
            history[idx] = metrics

        print(
            f"[accident_presence] Person {idx}: "
            f"falling={falling}, lying={lying}, metrics={metrics}"
        )

        # Fall detection logic:
        # 1. If detected falling motion AND lying -> increment counter
        # 2. OR if lying for consecutive frames (even without motion) -> increment counter
        # This handles cases where person is already lying when detection starts
        if lying:
            # If lying posture detected, increment counter
            # If also detected falling motion, increment faster
            if falling:
                fall_counter[idx] = fall_counter.get(idx, 0) + 2  # Faster confirmation if motion detected
            else:
                fall_counter[idx] = fall_counter.get(idx, 0) + 1  # Slower confirmation for static lying
        else:
            # Not lying, reset counter
            fall_counter[idx] = 0

        if fall_counter[idx] >= CONFIRM_FRAMES:
            fallen_ids.append(idx)

    # -----------------------------
    # 4. Final decision
    # -----------------------------
    if not fallen_ids:
        return None

    print("[accident_presence] 🚨 FALL CONFIRMED")

    return {
        "label": "🚨 Human fall detected",
        "fallen_count": len(fallen_ids),
        "fallen_indices": fallen_ids,
    }
