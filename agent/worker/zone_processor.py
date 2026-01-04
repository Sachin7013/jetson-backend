"""
Zone Processor
==============

Handles all zone-related processing including:
- Zone geometry checks (point, box, mask in polygon)
- Filtering detections by zone
- Tracking objects entering/exiting zones
- Generating enter/exit events
"""
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import numpy as np

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None


################################################################################
# Zone Geometry Utilities
################################################################################

def is_point_in_polygon(point_x: float, point_y: float, polygon: List[List[float]]) -> bool:
    """
    Check if a point (x, y) is inside a polygon using ray casting algorithm.
    
    This is a simple algorithm that works by casting a ray from the point
    to infinity and counting how many polygon edges it crosses.
    If the count is odd, the point is inside; if even, it's outside.
    
    Args:
        point_x: X coordinate of the point
        point_y: Y coordinate of the point
        polygon: List of [x, y] coordinates defining the polygon
    
    Returns:
        True if point is inside polygon, False otherwise
    """
    if not polygon or len(polygon) < 3:
        return False
    
    # Count how many polygon edges the ray crosses
    crossings = 0
    num_points = len(polygon)
    
    for i in range(num_points):
        # Get current point and next point (wrapping around)
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % num_points]
        
        # Check if ray crosses this edge
        # Ray goes from (point_x, point_y) to (infinity, point_y)
        if ((y1 > point_y) != (y2 > point_y)):  # Edge crosses the ray's y-level
            # Calculate x-coordinate where edge crosses the ray
            if point_x < (x2 - x1) * (point_y - y1) / (y2 - y1) + x1:
                crossings += 1
    
    # Odd number of crossings means point is inside
    return crossings % 2 == 1


def is_box_in_zone(box: List[float], zone_polygon: List[List[float]]) -> bool:
    """
    Check if a bounding box is inside the restricted zone.
    Uses Option 2: Check if ANY corner of the box is inside the polygon.
    
    Args:
        box: Bounding box as [x1, y1, x2, y2] (top-left and bottom-right)
        zone_polygon: List of [x, y] coordinates defining the zone polygon
    
    Returns:
        True if any corner of the box is inside the zone, False otherwise
    """
    if not box or len(box) < 4:
        return False
    
    if not zone_polygon or len(zone_polygon) < 3:
        return False
    
    x1, y1, x2, y2 = box
    
    # Get all four corners of the bounding box
    corners = [
        (x1, y1),  # Top-left
        (x2, y1),  # Top-right
        (x2, y2),  # Bottom-right
        (x1, y2)   # Bottom-left
    ]
    
    # Check if ANY corner is inside the polygon
    for corner_x, corner_y in corners:
        if is_point_in_polygon(corner_x, corner_y, zone_polygon):
            return True
    
    return False


def is_mask_in_zone(mask: np.ndarray, zone_polygon: List[List[float]], frame_height: int, frame_width: int) -> bool:
    """
    Check if a mask overlaps with the restricted zone.
    
    This checks if any part of the mask is inside the polygon by:
    1. Finding all non-zero pixels in the mask
    2. Checking if any of those pixels are inside the polygon
    
    Args:
        mask: Binary mask (numpy array)
        zone_polygon: List of [x, y] coordinates defining the zone polygon
        frame_height: Height of the frame
        frame_width: Width of the frame
    
    Returns:
        True if mask overlaps with zone, False otherwise
    """
    if mask is None or mask.size == 0:
        return False
    
    if not zone_polygon or len(zone_polygon) < 3:
        return False
    
    # Resize mask to frame dimensions if needed
    if mask.shape[:2] != (frame_height, frame_width):
        mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
    
    # Ensure mask is binary (0 or 255)
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    
    # Convert to 2D if needed
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Find all non-zero pixels (where the object is)
    coords = np.column_stack(np.where(mask > 0))
    
    if len(coords) == 0:
        return False
    
    # Check if ANY pixel of the mask is inside the polygon
    # We sample some pixels to avoid checking every single one (for performance)
    # Check center point and a few sample points
    sample_indices = [0, len(coords) // 4, len(coords) // 2, 3 * len(coords) // 4, len(coords) - 1]
    
    for idx in sample_indices:
        if idx < len(coords):
            y, x = coords[idx]
            if is_point_in_polygon(float(x), float(y), zone_polygon):
                return True
    
    return False


################################################################################
# Extract target classes from rules for zone filtering
################################################################################

def extract_target_classes_from_rules(rules: List[Dict[str, Any]]) -> List[str]:
    """
    Extract target class names from rules that should be filtered by zone.
    
    This function analyzes rules to determine which classes need zone filtering.
    Works with class_presence, weapon_detection, class_count, and count_at_least rules.
    
    Args:
        rules: List of rule dictionaries from the task
    
    Returns:
        List of target class names (normalized to lowercase)
    """
    target_classes = set()
    
    if not rules:
        return []
    
    for rule in rules:
        rule_type = str(rule.get("type", "")).lower()
        
        # Extract classes from different rule types
        if rule_type in ["class_presence", "class_count", "count_at_least"]:
            # Support both "class" (singular) and "classes" (plural) formats
            rule_classes = rule.get("classes") or []
            rule_class = rule.get("class") or rule.get("target_class")
            
            if rule_class and not rule_classes:
                rule_classes = [rule_class]
            
            # Add all classes to target set
            for cls in rule_classes:
                if isinstance(cls, str):
                    target_classes.add(cls.lower().strip())
        
        elif rule_type == "weapon_detection":
            # For weapon detection, get the weapon class
            weapon_class = str(rule.get("class", "")).lower().strip()
            if weapon_class:
                target_classes.add(weapon_class)
    
    return list(target_classes)


################################################################################
# Filter detections by restricted zone
################################################################################

def filter_detections_by_zone(
    detections: Dict[str, Any],
    zone_polygon: List[List[float]],
    frame_height: int,
    frame_width: int,
    target_classes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Filter detections to only include objects inside the restricted zone.
    Works with both bounding boxes and masks.
    
    If target_classes is provided, only detections matching those classes are checked.
    If target_classes is None, all detections are checked (backward compatibility).
    
    Args:
        detections: Detection dictionary with boxes, classes, scores, masks, etc.
        zone_polygon: List of [x, y] coordinates defining the zone polygon
        frame_height: Height of the frame
        frame_width: Width of the frame
        target_classes: Optional list of class names to filter. If None, filters all classes.
                       If provided, only detections matching these classes are checked.
    
    Returns:
        Filtered detection dictionary (only objects inside zone that match target_classes)
    """
    boxes = detections.get("boxes", [])
    classes = detections.get("classes", [])
    scores = detections.get("scores", [])
    masks = detections.get("masks", [])
    keypoints = detections.get("keypoints", [])
    
    if not boxes or not classes:
        return detections
    
    # Normalize target classes to lowercase for comparison
    target_classes_lower = None
    if target_classes:
        target_classes_lower = [str(cls).lower().strip() for cls in target_classes if cls]
    
    # Keep track of which detections are inside the zone
    valid_indices = []
    
    for idx in range(len(boxes)):
        box = boxes[idx]
        detected_class = str(classes[idx]).lower() if idx < len(classes) else ""
        mask = masks[idx] if idx < len(masks) else None
        
        # If target_classes is specified, only check detections that match
        if target_classes_lower:
            # Check if this detection's class matches any of the target classes
            class_matches = False
            for target_class in target_classes_lower:
                # Support partial matching (e.g., "gun" matches "guns")
                if target_class in detected_class or detected_class in target_class:
                    class_matches = True
                    break
            
            if not class_matches:
                # Skip this detection - it's not one of the target classes
                continue
        
        # Check if this detection is in the zone
        is_in_zone = False
        
        if mask is not None:
            # Check mask overlap with zone
            is_in_zone = is_mask_in_zone(mask, zone_polygon, frame_height, frame_width)
        else:
            # Check bounding box corners
            is_in_zone = is_box_in_zone(box, zone_polygon)
        
        if is_in_zone:
            valid_indices.append(idx)
    
    # Filter all detection arrays to only keep valid indices
    filtered_boxes = [boxes[i] for i in valid_indices]
    filtered_classes = [classes[i] for i in valid_indices]
    filtered_scores = [scores[i] for i in valid_indices]
    filtered_masks = [masks[i] for i in valid_indices if i < len(masks)]
    filtered_keypoints = [keypoints[i] for i in valid_indices if i < len(keypoints)]
    
    # Create filtered detections dictionary
    filtered_detections = {
        "classes": filtered_classes,
        "scores": filtered_scores,
        "boxes": filtered_boxes,
        "masks": filtered_masks,
        "keypoints": filtered_keypoints,
        "ts": detections.get("ts"),
    }
    
    return filtered_detections


################################################################################
# Zone Enter/Exit Tracking
################################################################################

def _generate_object_id(box: List[float], detected_class: str) -> str:
    """
    Generate a unique identifier for an object based on its position and class.
    
    Uses the center of the bounding box rounded to nearest 50 pixels to handle
    small movements while still tracking the same object.
    
    Args:
        box: Bounding box as [x1, y1, x2, y2]
        detected_class: Class name of the object
    
    Returns:
        Unique string identifier for the object
    """
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Round to nearest 50 pixels to handle small movements
    rounded_x = int(round(center_x / 50) * 50)
    rounded_y = int(round(center_y / 50) * 50)
    
    return f"{detected_class.lower()}_{rounded_x}_{rounded_y}"


def get_objects_in_zone(
    detections: Dict[str, Any],
    zone_polygon: List[List[float]],
    frame_height: int,
    frame_width: int,
    target_classes: Optional[List[str]] = None
) -> Set[str]:
    """
    Get set of object IDs currently in the zone.
    
    Args:
        detections: Detection dictionary with boxes, classes, scores, masks, etc.
        zone_polygon: List of [x, y] coordinates defining the zone polygon
        frame_height: Height of the frame
        frame_width: Width of the frame
        target_classes: Optional list of class names to filter
    
    Returns:
        Set of object IDs (strings) currently in the zone
    """
    boxes = detections.get("boxes", [])
    classes = detections.get("classes", [])
    masks = detections.get("masks", [])
    
    if not boxes or not classes:
        return set()
    
    # Normalize target classes to lowercase for comparison
    target_classes_lower = None
    if target_classes:
        target_classes_lower = [str(cls).lower().strip() for cls in target_classes if cls]
    
    objects_in_zone = set()
    
    for idx in range(len(boxes)):
        box = boxes[idx]
        detected_class = str(classes[idx]).lower() if idx < len(classes) else ""
        mask = masks[idx] if idx < len(masks) else None
        
        # If target_classes is specified, only check detections that match
        if target_classes_lower:
            class_matches = False
            for target_class in target_classes_lower:
                if target_class in detected_class or detected_class in target_class:
                    class_matches = True
                    break
            
            if not class_matches:
                continue
        
        # Check if this detection is in the zone
        is_in_zone = False
        
        if mask is not None:
            is_in_zone = is_mask_in_zone(mask, zone_polygon, frame_height, frame_width)
        else:
            is_in_zone = is_box_in_zone(box, zone_polygon)
        
        if is_in_zone:
            # Generate unique ID for this object
            object_id = _generate_object_id(box, detected_class)
            objects_in_zone.add(object_id)
    
    return objects_in_zone


def detect_zone_enter_exit(
    current_objects: Set[str],
    previous_objects: Set[str],
    current_time: datetime
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Detect objects entering and exiting the zone by comparing current and previous states.
    
    Args:
        current_objects: Set of object IDs currently in zone
        previous_objects: Set of object IDs that were in zone in previous frame
        current_time: Current timestamp
    
    Returns:
        Dictionary with "entered" and "exited" lists, each containing event dictionaries:
        {
            "entered": [
                {
                    "object_id": str,
                    "class": str,
                    "enter_time": datetime
                },
                ...
            ],
            "exited": [
                {
                    "object_id": str,
                    "class": str,
                    "exit_time": datetime
                },
                ...
            ]
        }
    """
    entered = []
    exited = []
    
    # Objects that are in current but not in previous = entered
    for object_id in current_objects - previous_objects:
        # Extract class from object_id (format: "class_x_y")
        parts = object_id.split("_")
        if len(parts) >= 3:
            obj_class = "_".join(parts[:-2])  # Handle class names with underscores
        else:
            obj_class = parts[0] if parts else "unknown"
        
        entered.append({
            "object_id": object_id,
            "class": obj_class,
            "enter_time": current_time
        })
    
    # Objects that were in previous but not in current = exited
    for object_id in previous_objects - current_objects:
        # Extract class from object_id
        parts = object_id.split("_")
        if len(parts) >= 3:
            obj_class = "_".join(parts[:-2])
        else:
            obj_class = parts[0] if parts else "unknown"
        
        exited.append({
            "object_id": object_id,
            "class": obj_class,
            "exit_time": current_time
        })
    
    return {
        "entered": entered,
        "exited": exited
    }


################################################################################
# Draw restricted zone polygon
################################################################################

def draw_zone_polygon(frame: np.ndarray, zone_polygon: List[List[float]], zone_color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """
    Draw the restricted zone polygon on the frame.
    Draws both the outline and a semi-transparent fill.
    
    Args:
        frame: Frame to draw on
        zone_polygon: List of [x, y] coordinates defining the zone polygon
        zone_color: Color for the zone (BGR format), default is yellow
    
    Returns:
        Frame with zone polygon drawn
    """
    if cv2 is None:
        return frame
    
    if not zone_polygon or len(zone_polygon) < 3:
        return frame
    
    # Convert polygon to numpy array format for OpenCV
    polygon_points = np.array(zone_polygon, dtype=np.int32)
    
    # Create a copy of the frame
    result_frame = frame.copy()
    
    # Draw filled polygon with transparency
    overlay = result_frame.copy()
    cv2.fillPoly(overlay, [polygon_points], zone_color)
    alpha = 0.3  # Transparency (0.0 = fully transparent, 1.0 = fully opaque)
    result_frame = cv2.addWeighted(result_frame, 1 - alpha, overlay, alpha, 0)
    
    # Draw polygon outline (thicker, more visible)
    cv2.polylines(result_frame, [polygon_points], isClosed=True, color=zone_color, thickness=3)
    
    # Draw label at the first point
    if len(zone_polygon) > 0:
        label_x, label_y = int(zone_polygon[0][0]), int(zone_polygon[0][1])
        
        # Helper function to draw label (imported from frame_processor if needed)
        # For now, we'll use a simple text drawing
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize("RESTRICTED ZONE", font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(
            result_frame,
            (label_x, label_y - text_height - 10),
            (label_x + text_width + 5, label_y + 5),
            zone_color,
            -1
        )
        
        # Draw text
        cv2.putText(
            result_frame,
            "RESTRICTED ZONE",
            (label_x, label_y - 5),
            font,
            font_scale,
            (255, 255, 255),  # White text
            thickness
        )
    
    return result_frame

