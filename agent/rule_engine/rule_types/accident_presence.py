from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from agent.rule_engine.registry import register_rule


def _get_point(person: List[List[float]], idx: int) -> Optional[Tuple[float, float]]:
    if idx < 0 or idx >= len(person):
        return None
    pt = person[idx]
    if pt is None or len(pt) < 2:
        return None
    x = float(pt[0])
    y = float(pt[1])
    return (x, y)


def _midpoint(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _angle_from_vertical(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return 0.0
    import math
    angle_rad = math.atan2(dx, dy)
    angle_deg = abs(angle_rad * 180.0 / math.pi)
    if angle_deg > 90.0:
        angle_deg = 180.0 - angle_deg
    return angle_deg


def _is_person_fallen(person: List[List[float]], angle_threshold: float, ratio_threshold: float, min_kps: int) -> bool:
    pts: List[Tuple[float, float]] = []
    for kp in person:
        if kp is None or len(kp) < 2:
            continue
        x = float(kp[0])
        y = float(kp[1])
        pts.append((x, y))
    if len(pts) < max(1, min_kps):
        return False

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    width = max(xs) - min(xs) if xs else 0.0
    height = max(ys) - min(ys) if ys else 0.0
    if width <= 0.0 or height <= 0.0:
        ratio_flag = False
    else:
        ratio_flag = (width / max(height, 1e-6)) >= ratio_threshold

    ls = _get_point(person, 5)
    rs = _get_point(person, 6)
    lh = _get_point(person, 11)
    rh = _get_point(person, 12)
    angle_flag = False
    if ls and rs and lh and rh:
        shoulder_mid = _midpoint(ls, rs)
        hip_mid = _midpoint(lh, rh)
        angle = _angle_from_vertical(shoulder_mid, hip_mid)
        angle_flag = angle >= angle_threshold
    return bool(ratio_flag or angle_flag)


@register_rule("accident_presence")
def evaluate_accident_presence(rule: Dict[str, Any], detections: Dict[str, Any], task: Dict[str, Any], rule_state: Dict[str, Any], now: datetime) -> Optional[Dict[str, Any]]:
    keypoints: List[List[List[float]]] = detections.get("keypoints") or []
    if not keypoints:
        rule_state["last_fall_since"] = None
        return None

    angle_threshold = float(rule.get("angle_threshold", 50.0) or 50.0)
    ratio_threshold = float(rule.get("ratio_threshold", 1.2) or 1.2)
    min_kps = int(rule.get("min_keypoints", 5) or 5)

    fallen_indices: List[int] = []
    for idx, person in enumerate(keypoints):
        try:
            if _is_person_fallen(person, angle_threshold, ratio_threshold, min_kps):
                fallen_indices.append(idx)
        except Exception:
            continue

    if not fallen_indices:
        rule_state["last_fall_since"] = None
        return None

    duration_seconds = int(rule.get("duration_seconds", 0) or 0)
    last_since: Optional[datetime] = rule_state.get("last_fall_since")
    if duration_seconds <= 0:
        rule_state["last_fall_since"] = now
    else:
        if last_since is None:
            rule_state["last_fall_since"] = now
            return None
        if (now - last_since).total_seconds() < duration_seconds:
            return None

    label = rule.get("label") or "Possible fall detected"
    return {"label": label, "fallen_count": len(fallen_indices)}

