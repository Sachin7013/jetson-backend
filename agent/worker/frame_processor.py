"""
Frame Processor
===============

Unified utilities for processing frames with detections.
This module provides a single, generic drawing function that works for all rule types.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

# Import visualization config helper
from agent.worker.visualization import (
    analyze_rules_for_visualization,
    get_class_color,
    should_draw_box
)

# ========================================================================
# Helper Functions
# ========================================================================

def _draw_label_with_background(frame, x1, y1, label_text, color_bgr, font_scale=0.6):
    """
    Helper function to draw text label with a colored background box.
    This makes the code cleaner by avoiding repetition.
    
    Args:
        frame: The image frame to draw on
        x1, y1: Top-left corner position for the label
        label_text: The text to display
        color_bgr: Background color as (B, G, R) tuple
        font_scale: Size of the text (default 0.6)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    
    # Calculate how big the text will be
    (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
    
    # Calculate position for label (make sure it's not above the frame)
    label_y_position = max(y1, text_height + 10)
    
    # Draw colored rectangle behind the text
    cv2.rectangle(
        frame,
        (x1, label_y_position - text_height - 10),  # Top-left of background
        (x1 + text_width + 5, label_y_position + 5),  # Bottom-right of background
        color_bgr,
        -1  # -1 means filled rectangle
    )
    
    # Draw the text on top (white color)
    cv2.putText(
        frame,
        label_text,
        (x1, label_y_position - 5),  # Text position
        font,
        font_scale,
        (255, 255, 255),  # White text color
        thickness
    )

# ========================================================================
# Check boxes overlap
# ========================================================================

def _check_boxes_overlap(box1, box2):
    """
    Simple function to check if two bounding boxes overlap significantly.
    
    This uses IoU (Intersection over Union) - a common way to measure box overlap.
    IoU = (Area of overlap) / (Area of union)
    
    Example: If IoU = 0.5, it means boxes overlap 50% of their combined area.
    
    Args:
        box1: First box as [x1, y1, x2, y2] (top-left and bottom-right corners)
        box2: Second box as [x1, y1, x2, y2]
    
    Returns:
        True if boxes overlap enough (IoU > 0.1), False otherwise
    """
    # Extract coordinates from both boxes
    box1_x1, box1_y1, box1_x2, box1_y2 = box1
    box2_x1, box2_y1, box2_x2, box2_y2 = box2
    
    # Step 1: Find the overlapping rectangle (intersection)
    # The overlap starts at the maximum of the left/top edges
    # and ends at the minimum of the right/bottom edges
    overlap_left = max(box1_x1, box2_x1)
    overlap_top = max(box1_y1, box2_y1)
    overlap_right = min(box1_x2, box2_x2)
    overlap_bottom = min(box1_y2, box2_y2)
    
    # Step 2: Check if there's actually an overlap
    # If right edge is to the left of left edge, or bottom is above top, no overlap
    if overlap_right <= overlap_left or overlap_bottom <= overlap_top:
        return False
    
    # Step 3: Calculate the area of overlap
    overlap_area = (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
    
    # Step 4: Calculate area of each box
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    
    # Step 5: Calculate total area (union)
    # Union = box1_area + box2_area - overlap_area
    # (We subtract overlap_area because it's counted twice)
    total_union_area = box1_area + box2_area - overlap_area
    
    # Step 6: Avoid division by zero
    if total_union_area == 0:
        return False
    
    # Step 7: Calculate IoU (Intersection over Union)
    iou_ratio = overlap_area / total_union_area
    
    # Step 8: Return True if overlap is more than 10%
    # This means the weapon and person boxes overlap at least 10%
    return iou_ratio > 0.1


# ========================================================================
# Unified Drawing Pipeline
# ========================================================================

def draw_detections_unified(
    frame: np.ndarray,
    detections: Dict[str, Any],
    rules: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Unified drawing function that handles ALL types of detections.
    
    This replaces the old rule-specific methods:
    - draw_weapon_detections
    - draw_pose_keypoints
    - draw_bounding_boxes
    
    How it works:
    1. Analyze rules to determine what to visualize
    2. Draw boxes for relevant classes
    3. Draw keypoints if needed
    4. Apply special overlays (weapon masks, etc.)
    
    Args:
        frame: Input frame (numpy array in BGR format)
        detections: Detection dictionary with boxes, classes, scores, keypoints
        rules: List of rule dictionaries
    
    Returns:
        Frame with all visualizations drawn
    """
    if cv2 is None:
        return frame
    
    if not rules:
        return frame
    
    # Step 1: Analyze rules to get visualization configuration
    viz_config = analyze_rules_for_visualization(rules)
    
    # Step 2: Make a copy of the frame (don't modify original)
    processed_frame = frame.copy()
    height, width = processed_frame.shape[:2]
    
    # Step 3: Draw weapon overlays first (if needed)
    # This goes first so boxes and keypoints can be drawn on top
    if viz_config.get("draw_weapon_overlays", False):
        processed_frame = _draw_weapon_overlays(
            processed_frame, 
            detections, 
            viz_config
        )
    
    # Step 4: Draw bounding boxes for relevant classes
    if viz_config.get("draw_boxes", False):
        processed_frame = _draw_boxes_unified(
            processed_frame,
            detections,
            viz_config
        )
    
    # Step 5: Draw keypoints (if needed)
    # Keypoints go last so they're visible on top
    if viz_config.get("draw_keypoints", False):
        processed_frame = _draw_keypoints_unified(
            processed_frame,
            detections,
            viz_config
        )
    
    return processed_frame

################################################################################
# Draw boxes
################################################################################

def _draw_boxes_unified(
    frame: np.ndarray,
    detections: Dict[str, Any],
    viz_config: Dict[str, Any]
) -> np.ndarray:
    """
    Draw bounding boxes based on visualization config.
    
    Args:
        frame: Frame to draw on
        detections: Detection data
        viz_config: Visualization configuration
    
    Returns:
        Frame with boxes drawn
    """
    boxes = detections.get("boxes", [])
    classes = detections.get("classes", [])
    scores = detections.get("scores", [])
    
    if not boxes or not classes:
        return frame
    
    height, width = frame.shape[:2]
    
    # Draw each box if it matches the config
    for box, cls, score in zip(boxes, classes, scores):
        if not should_draw_box(cls, viz_config):
            continue
        
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box)
        
        # Clamp to frame boundaries
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Get color for this class
        color = get_class_color(cls, viz_config)
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{cls}"
        if score:
            label += f" {score:.2f}"
        
        _draw_label_with_background(frame, x1, y1, label, color, font_scale=0.5)
    
    return frame

################################################################################
# Draw keypoints
################################################################################

def _draw_keypoints_unified(
    frame: np.ndarray,
    detections: Dict[str, Any],
    viz_config: Dict[str, Any]
) -> np.ndarray:
    """
    Draw pose keypoints based on visualization config.
    
    Args:
        frame: Frame to draw on
        detections: Detection data
        viz_config: Visualization configuration
    
    Returns:
        Frame with keypoints drawn
    """
    keypoints = detections.get("keypoints", [])
    if not keypoints:
        return frame
    
    height, width = frame.shape[:2]
    colors = viz_config.get("colors", {})
    
    # Keypoint skeleton connections (COCO format)
    skeleton = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)
    ]
    
    keypoint_color = colors.get("keypoint", (0, 255, 255))
    line_color = colors.get("keypoint_line", (0, 255, 0))
    
    # Draw keypoints for each person
    for person in keypoints:
        # Convert keypoints to pixel coordinates
        pts: List[Optional[Tuple[int, int]]] = []
        for kp in person:
            if kp is None or len(kp) < 2:
                pts.append(None)
                continue
            x = int(max(0, min(int(kp[0]), width - 1)))
            y = int(max(0, min(int(kp[1]), height - 1)))
            pts.append((x, y))
        
        # Draw keypoint circles
        for p in pts:
            if p is None:
                continue
            cv2.circle(frame, p, 3, keypoint_color, -1)
        
        # Draw skeleton lines
        for a, b in skeleton:
            pa = pts[a] if a < len(pts) else None
            pb = pts[b] if b < len(pts) else None
            if pa is None or pb is None:
                continue
            cv2.line(frame, pa, pb, line_color, 2)
    
    return frame

################################################################################
# Draw weapon overlays
################################################################################

def _draw_weapon_overlays(
    frame: np.ndarray,
    detections: Dict[str, Any],
    viz_config: Dict[str, Any]
) -> np.ndarray:
    """
    Draw special weapon detection overlays (red masks on armed persons).
    
    This is the complex weapon detection logic that:
    1. Finds weapons and people
    2. Matches weapons to people (by overlap)
    3. Draws red overlay on armed persons
    4. Draws weapon boxes for unmatched weapons
    
    Args:
        frame: Frame to draw on
        detections: Detection data
        viz_config: Visualization configuration
    
    Returns:
        Frame with weapon overlays drawn
    """
    boxes = detections.get("boxes", [])
    classes = detections.get("classes", [])
    scores = detections.get("scores", [])
    
    if not boxes or not classes:
        return frame
    
    weapon_overlay_classes = viz_config.get("weapon_overlay_classes", set())
    if not weapon_overlay_classes:
        return frame
    
    height, width = frame.shape[:2]
    
    # Separate weapons and people
    weapons_found = []
    people_found = []
    
    for box, cls, score in zip(boxes, classes, scores):
        cls_lower = str(cls).lower()
        
        # Check if weapon
        is_weapon = any(weapon in cls_lower for weapon in weapon_overlay_classes)
        if is_weapon:
            weapons_found.append((box, cls_lower, score))
        elif cls_lower == "person":
            people_found.append((box, cls_lower, score))
    
    # Match weapons to people
    armed_people_indices = set()
    matched_weapons = set()
    
    for weapon_idx, (weapon_box, weapon_cls, weapon_score) in enumerate(weapons_found):
        for person_idx, (person_box, person_cls, person_score) in enumerate(people_found):
            if _check_boxes_overlap(weapon_box, person_box):
                armed_people_indices.add(person_idx)
                matched_weapons.add(weapon_idx)
                break
    
    # Draw red overlay on armed people
    for person_idx in armed_people_indices:
        person_box, person_cls, person_score = people_found[person_idx]
        x1, y1, x2, y2 = map(int, person_box)
        
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        if x2 > x1 and y2 > y1:
            # Apply red overlay
            person_region = frame[y1:y2, x1:x2]
            if person_region.size > 0:
                red_mask = person_region.copy()
                red_mask[:, :] = (0, 0, 255)  # Red in BGR
                alpha = 0.5
                blended = cv2.addWeighted(person_region, 1 - alpha, red_mask, alpha, 0)
                frame[y1:y2, x1:x2] = blended
            
            # Draw red border
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Draw label
            _draw_label_with_background(frame, x1, y1, "ARMED PERSON", (0, 0, 255))
    
    # Draw unmatched weapons
    for weapon_idx, (weapon_box, weapon_cls, weapon_score) in enumerate(weapons_found):
        if weapon_idx in matched_weapons:
            continue
        
        x1, y1, x2, y2 = map(int, weapon_box)
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        if x2 > x1 and y2 > y1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = f"WEAPON DETECTED ({weapon_cls})"
            if weapon_score:
                label += f" {weapon_score:.2f}"
            _draw_label_with_background(frame, x1, y1, label, (0, 0, 255))
    
    return frame

