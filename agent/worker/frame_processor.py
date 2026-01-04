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

# Import zone utilities from zone_processor
from agent.worker.zone_processor import (
    draw_zone_polygon,
    filter_detections_by_zone
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

# Import zone utilities from zone_processor
from agent.worker.zone_processor import (
    draw_zone_polygon,
    filter_detections_by_zone
)


# ========================================================================
# Check if two bounding boxes overlap
# ========================================================================

def _check_boxes_overlap(box1, box2):
    """
    Simple function to check if two bounding boxes overlap significantly.
    
    This uses IoU (Intersection over Union) - a common way to measure box overlap.
    IoU = (Area of overlap) / (Area of union)
    
    Example: If IoU = 0.05, it means boxes overlap 5% of their combined area.
    
    Args:
        box1: First box as [x1, y1, x2, y2] (top-left and bottom-right corners)
        box2: Second box as [x1, y1, x2, y2]
    
    Returns:
        True if boxes overlap enough (IoU > 0.01), False otherwise
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
    return iou_ratio > 0.05


# ========================================================================
# Unified Drawing Pipeline
# handles ALL types of detections and visualizations
# This replaces the old rule-specific methods:
# - draw_weapon_detections
# - draw_pose_keypoints
# - draw_bounding_boxes
# ========================================================================

def draw_detections_unified(
    frame: np.ndarray,
    detections: Dict[str, Any],
    rules: List[Dict[str, Any]],
    task: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Unified drawing function that handles ALL types of detections.
    
    This replaces the old rule-specific methods:
    - draw_weapon_detections
    - draw_pose_keypoints
    - draw_bounding_boxes
    - draw_person_masks
    
    How it works:
    1. Draw restricted zone polygon (if requires_zone is true)
    2. Analyze rules to determine what to visualize
    3. Draw boxes for relevant classes
    4. Draw keypoints if needed
    5. Apply special overlays (weapon masks, etc.)
    6. Draw person masks (if needed)

    Args:
        frame: Input frame (numpy array in BGR format)
        detections: Detection dictionary with boxes, classes, scores, keypoints, masks
        rules: List of rule dictionaries
        task: Task dictionary (optional, needed for zone drawing)
    
    Returns:
        Frame with all visualizations drawn
    """
    if cv2 is None:
        return frame
    
    if not rules:
        return frame
    
    # Step 1: Make a copy of the frame (don't modify original)
    processed_frame = frame.copy()
    height, width = processed_frame.shape[:2]
    
    # Step 2: Draw restricted zone polygon if required
    # Check if task has requires_zone flag and zone coordinates
    if task:
        requires_zone = task.get("requires_zone", False)
        zone = task.get("zone", {})
        
        if requires_zone and zone:
            zone_type = zone.get("type", "").lower()
            zone_coordinates = zone.get("coordinates", [])
            
            if zone_type == "polygon" and zone_coordinates:
                # Draw the restricted zone polygon (yellow color)
                processed_frame = draw_zone_polygon(processed_frame, zone_coordinates, zone_color=(0, 255, 255))
    
    # Step 3: Analyze rules to get visualization configuration
    viz_config = analyze_rules_for_visualization(rules)
    
    # Step 4: Identify armed people (if weapon detection is enabled)
    # This needs to happen before drawing masks so we know which persons are armed
    armed_person_indices = set()
    if viz_config.get("draw_weapon_overlays", False):
        armed_person_indices = _identify_armed_people(detections, viz_config)
        # Store armed person info in detections for mask drawing
        detections["armed_person_indices"] = armed_person_indices
    
    # Step 5: Draw bounding boxes for relevant classes
    if viz_config.get("draw_boxes", False):
        processed_frame = _draw_boxes_unified(
            processed_frame,
            detections,
            viz_config
        )
    
    # Step 6: Draw keypoints (if needed)
    # Keypoints go last so they're visible on top
    if viz_config.get("draw_keypoints", False):
        processed_frame = _draw_keypoints_unified(
            processed_frame,
            detections,
            viz_config
        )

    # Step 7: Draw person masks (if needed) - green for normal, red for armed
    if viz_config.get("draw_person_masks", False):
        processed_frame = _draw_person_masks(
            processed_frame,
            detections
        )
    
    # Step 8: Draw weapon overlays (weapon masks in red)
    if viz_config.get("draw_weapon_overlays", False):
        processed_frame = _draw_weapon_overlays(
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
    Skips drawing boxes for detections that have masks (when segmentation is active).
    
    Args:
        frame: Frame to draw on
        detections: Detection data
        viz_config: Visualization configuration
    
    Returns:
        Frame with boxes drawn (only for detections without masks)
    """
    boxes = detections.get("boxes", [])
    classes = detections.get("classes", [])
    scores = detections.get("scores", [])
    masks = detections.get("masks", [])
    has_masks = detections.get("has_masks", False)
    
    if not boxes or not classes:
        return frame
    
    height, width = frame.shape[:2]
    
    # Check if masks are being drawn (if person masks are enabled, skip boxes for persons)
    draw_person_masks = viz_config.get("draw_person_masks", False)
    
    # Draw each box if it matches the config
    for idx, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        if not should_draw_box(cls, viz_config):
            continue
        
        # Skip drawing box if mask is available for this detection
        if has_masks and idx < len(masks):
            mask = masks[idx]
            if mask is not None:
                # Check if mask has any non-zero pixels (valid mask)
                if isinstance(mask, np.ndarray) and mask.size > 0:
                    if mask.max() > 0:  # Mask has content
                        # Skip box drawing for this detection (mask will be drawn instead)
                        continue
        
        # Also skip boxes for persons if person masks are being drawn AND masks are available
        # Only skip if we actually have masks to draw, otherwise draw the box
        if draw_person_masks and str(cls).lower() == "person":
            # Check if masks are actually available for this detection
            if has_masks and idx < len(masks) and masks[idx] is not None:
                # Skip box (mask will be drawn instead)
                continue
            # If no masks available, draw the box normally
        
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
# Draw person masks
################################################################################

def _draw_person_masks(
    frame: np.ndarray,
    detections: Dict[str, Any],
) -> np.ndarray:
    """
    Draw person masks as colored overlays on the frame.
    - Green masks for normal persons
    - Red masks for armed persons (with weapons)
    Only draws masks for detections where class is "person".
    Includes labels since boxes are skipped when masks are available.
    
    Args:
        frame: Frame to draw on
        detections: Detection data (may contain "armed_person_indices")
    
    Returns:
        Frame with person masks drawn as colored overlays
    """
    masks = detections.get("masks", [])
    classes = detections.get("classes", [])
    scores = detections.get("scores", [])
    armed_person_indices = detections.get("armed_person_indices", set())
    
    if not masks:
        return frame
    
    height, width = frame.shape[:2]
    
    # Default color for person masks (BGR format - green)
    default_mask_color = (0, 255, 0)  # Green in BGR
    armed_mask_color = (0, 0, 255)  # Red in BGR
    alpha = 0.4  # Transparency (0.0 = fully transparent, 1.0 = fully opaque)
    
    # Draw masks only for person detections
    # Masks array should align with classes array (one mask per detection)
    for idx, mask in enumerate(masks):
        if mask is None:
            continue
        
        # Only draw mask if corresponding class is "person"
        if idx < len(classes) and str(classes[idx]).lower() == "person":
            # Determine color: red if armed, green otherwise
            is_armed = idx in armed_person_indices
            mask_color = armed_mask_color if is_armed else default_mask_color
            label_text = "ARMED PERSON" if is_armed else "person"
            
            mask = mask.astype(np.uint8)
            
            # Resize mask to frame dimensions if needed
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Ensure mask is binary (0 or 255)
            if mask.max() <= 1:
                mask = (mask * 255).astype(np.uint8)
            
            # Create colored overlay: create a colored image with the mask shape
            colored_overlay = np.zeros((height, width, 3), dtype=np.uint8)
            colored_overlay[:] = mask_color
            
            # Apply mask to the colored overlay (only show color where mask is non-zero)
            mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(mask.shape) == 2 else mask
            mask_binary = (mask_3channel > 0).astype(np.uint8)
            colored_overlay = colored_overlay * mask_binary
            
            # Blend the colored overlay with the original frame
            frame = cv2.addWeighted(frame, 1.0, colored_overlay, alpha, 0)
            
            # Draw label on the mask (since box is not drawn)
            # Find the top-left point of the mask for label placement
            mask_2d = mask if len(mask.shape) == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            coords = np.column_stack(np.where(mask_2d > 0))
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                label_x = max(0, min(int(x_min), width - 1))
                label_y = max(0, min(int(y_min), height - 1))
                
                # Get score if available
                score = scores[idx] if idx < len(scores) else None
                if score and not is_armed:
                    label_text += f" {score:.2f}"
                
                # Draw label with mask color
                _draw_label_with_background(frame, label_x, label_y, label_text, mask_color, font_scale=0.5)
    
    return frame

################################################################################
# Check if weapon mask overlaps with person mask
################################################################################

def _check_weapon_mask_overlap(weapon_box: List[float], person_mask: np.ndarray) -> bool:
    """
    Check if a weapon bounding box overlaps with a person mask.
    
    Args:
        weapon_box: Weapon bounding box [x1, y1, x2, y2]
        person_mask: Binary person mask (numpy array)
    
    Returns:
        True if weapon overlaps with person mask, False otherwise
    """
    if person_mask is None or person_mask.size == 0:
        return False
    
    x1, y1, x2, y2 = map(int, weapon_box)
    height, width = person_mask.shape[:2]
    
    # Clamp coordinates to mask boundaries
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    # Extract the region of the mask where the weapon box is
    weapon_region_mask = person_mask[y1:y2, x1:x2]
    
    # Check if any pixels in the weapon box region are part of the person mask
    # (i.e., mask value > 0)
    if weapon_region_mask.size > 0:
        overlap_pixels = np.sum(weapon_region_mask > 0)
        total_pixels = weapon_region_mask.size
        overlap_ratio = overlap_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # Consider it an overlap if at least 5% of the weapon box overlaps with the mask
        return overlap_ratio > 0.05
    
    return False

################################################################################
# Identify armed people (have weapons overlapping with their masks)
################################################################################
def _identify_armed_people(
    detections: Dict[str, Any],
    viz_config: Dict[str, Any]
) -> set:
    """
    Identify which persons are armed (have weapons overlapping with their masks).
    Returns a set of detection indices for armed persons.
    
    Args:
        detections: Detection data (with boxes, classes, scores, masks)
        viz_config: Visualization configuration
    
    Returns:
        Set of detection indices for armed persons
    """
    boxes = detections.get("boxes", [])
    classes = detections.get("classes", [])
    masks = detections.get("masks", [])
    
    if not boxes or not classes:
        return set()
    
    weapon_overlay_classes = viz_config.get("weapon_overlay_classes", set())
    if not weapon_overlay_classes:
        return set()
    
    # Separate weapons and people (with their masks)
    weapons_found = []
    people_found = []  # (box, cls, score, mask, index)
    
    for idx, (box, cls) in enumerate(zip(boxes, classes)):
        cls_lower = str(cls).lower()
        
        # Check if weapon
        is_weapon = any(weapon in cls_lower for weapon in weapon_overlay_classes)
        if is_weapon:
            weapons_found.append((box, cls_lower, idx))
        elif cls_lower == "person":
            # Get corresponding mask if available
            person_mask = masks[idx] if idx < len(masks) else None
            people_found.append((box, cls_lower, person_mask, idx))
    
    # Match weapons to people using mask overlap
    armed_people_indices = set()
    
    for weapon_box, weapon_cls, weapon_det_idx in weapons_found:
        for person_box, person_cls, person_mask, person_det_idx in people_found:
            # Check if weapon overlaps with person mask
            if person_mask is not None:
                if _check_weapon_mask_overlap(weapon_box, person_mask):
                    armed_people_indices.add(person_det_idx)
                    break
            else:
                # Fallback to box overlap if no mask available
                if _check_boxes_overlap(weapon_box, person_box):
                    armed_people_indices.add(person_det_idx)
                    break
    
    return armed_people_indices

################################################################################
# Draw weapon overlays (weapon masks in red) - fallback for weapons without masks
################################################################################
def _draw_weapon_overlays(
    frame: np.ndarray,
    detections: Dict[str, Any],
    viz_config: Dict[str, Any]
) -> np.ndarray:
    """
    Draw weapon masks in red color.
    Weapons that overlap with person masks are drawn with red masks.
    Unmatched weapons are drawn with red boxes.
    
    Args:
        frame: Frame to draw on
        detections: Detection data (with boxes, classes, scores, masks)
        viz_config: Visualization configuration
    
    Returns:
        Frame with weapon overlays drawn
    """
    boxes = detections.get("boxes", [])
    classes = detections.get("classes", [])
    scores = detections.get("scores", [])
    masks = detections.get("masks", [])
    armed_person_indices = detections.get("armed_person_indices", set())
    
    if not boxes or not classes:
        return frame
    
    weapon_overlay_classes = viz_config.get("weapon_overlay_classes", set())
    if not weapon_overlay_classes:
        return frame
    
    height, width = frame.shape[:2]
    
    # Find weapons
    weapons_found = []
    for idx, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        cls_lower = str(cls).lower()
        is_weapon = any(weapon in cls_lower for weapon in weapon_overlay_classes)
        if is_weapon:
            weapon_mask = masks[idx] if idx < len(masks) else None
            weapons_found.append((box, cls_lower, score, weapon_mask, idx))
    
    # Draw weapon masks in red
    red_color = (0, 0, 255)  # Red in BGR
    alpha = 0.5  # Transparency
    
    for weapon_box, weapon_cls, weapon_score, weapon_mask, weapon_det_idx in weapons_found:
        if weapon_mask is not None:
            # Draw weapon mask in red
            mask = weapon_mask.astype(np.uint8)
            
            # Resize mask to frame dimensions if needed
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Ensure mask is binary (0 or 255)
            if mask.max() <= 1:
                mask = (mask * 255).astype(np.uint8)
            
            # Create red overlay using the weapon mask
            colored_overlay = np.zeros((height, width, 3), dtype=np.uint8)
            colored_overlay[:] = red_color
            
            # Apply mask to the colored overlay
            mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(mask.shape) == 2 else mask
            mask_binary = (mask_3channel > 0).astype(np.uint8)
            colored_overlay = colored_overlay * mask_binary
            
            # Blend the red overlay with the original frame
            frame = cv2.addWeighted(frame, 1.0, colored_overlay, alpha, 0)
            
            # Draw label on weapon mask
            mask_2d = mask if len(mask.shape) == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            coords = np.column_stack(np.where(mask_2d > 0))
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                label_x = max(0, min(int(x_min), width - 1))
                label_y = max(0, min(int(y_min), height - 1))
                
                label = f"WEAPON ({weapon_cls})"
                if weapon_score:
                    label += f" {weapon_score:.2f}"
                _draw_label_with_background(frame, label_x, label_y, label, red_color, font_scale=0.5)
        else:
            # Fallback: draw box for weapons without masks
            x1, y1, x2, y2 = map(int, weapon_box)
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), red_color, 3)
                label = f"WEAPON DETECTED ({weapon_cls})"
                if weapon_score:
                    label += f" {weapon_score:.2f}"
                _draw_label_with_background(frame, x1, y1, label, red_color)
    
    return frame

