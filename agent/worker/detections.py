"""
Detections utilities
--------------------

Unified helpers to extract all types of detections from YOLO results.
This module provides a single, generic method that works for all model types.
"""
from typing import List, Tuple, Dict, Any


def extract_all_detections(result, model_id: str = "", normalize_classes: bool = False) -> Dict[str, Any]:
    """
    Unified method to extract ALL types of detections from any YOLO result.
    
    This single method replaces the old rule-specific methods:
    - extract_detections_from_result
    - extract_keypoints_from_result  
    - extract_detections_from_weapon_result
    
    What it extracts:
    - Bounding boxes (x1, y1, x2, y2) for all detected objects
    - Class names for each detection
    - Confidence scores for each detection
    - Keypoints (if the model supports pose detection)
    
    Args:
        result: YOLO result object from model inference
        model_id: Optional model identifier (for debugging)
        normalize_classes: If True, convert class names to lowercase
    
    Returns:
        Dictionary with:
        {
            "boxes": [[x1, y1, x2, y2], ...],
            "classes": ["person", "car", ...],
            "scores": [0.95, 0.87, ...],
            "keypoints": [[[x, y, conf], ...], ...],  # One list per person
            "has_keypoints": bool,
            "model_id": str
        }
    """
    # Initialize empty results
    detections = {
        "boxes": [],
        "classes": [],
        "scores": [],
        "keypoints": [],
        "has_keypoints": False,
        "model_id": model_id
    }
    
    # If no result, return empty detections
    if result is None:
        return detections
    
    # Step 1: Extract bounding boxes, classes, and scores
    # All YOLO models (object detection, pose, weapon, etc.) have boxes
    if hasattr(result, "boxes"):
        try:
            # Get class name mapping (e.g., {0: "person", 1: "car"})
            class_names_by_id = result.names if hasattr(result, "names") else {}
            
            # Extract box coordinates, class IDs, and confidence scores
            xyxy = result.boxes.xyxy.tolist() if hasattr(result.boxes, "xyxy") else []
            cls_list = result.boxes.cls.tolist() if hasattr(result.boxes, "cls") else []
            conf_list = result.boxes.conf.tolist() if hasattr(result.boxes, "conf") else []
            
            # Convert each detection to our format
            for detection_index, box in enumerate(xyxy):
                # Box coordinates: [x1, y1, x2, y2]
                detections["boxes"].append([
                    float(box[0]), 
                    float(box[1]), 
                    float(box[2]), 
                    float(box[3])
                ])
                
                # Class name
                class_id = int(cls_list[detection_index]) if detection_index < len(cls_list) else -1
                class_name = class_names_by_id.get(class_id, str(class_id))
                
                # Normalize class name if requested (useful for weapon detection)
                if normalize_classes:
                    class_name = str(class_name).lower()
                else:
                    class_name = str(class_name)
                
                detections["classes"].append(class_name)
                
                # Confidence score
                score = float(conf_list[detection_index]) if detection_index < len(conf_list) else 0.0
                detections["scores"].append(score)
                
        except Exception:  # noqa: BLE001
            # If extraction fails, continue with empty boxes (keypoints might still work)
            pass
    
    # Step 2: Extract keypoints (only if model supports pose detection)
    # Pose models have a "keypoints" attribute
    kp = getattr(result, "keypoints", None)
    if kp is not None:
        try:
            # Try to get keypoints data
            if hasattr(kp, "data") and kp.data is not None:
                # Format: data contains [x, y, confidence] for each keypoint
                data = kp.data.tolist()
                for person in data:
                    person_keypoints: List[List[float]] = []
                    for pt in person:
                        if pt is None or len(pt) < 2:
                            continue
                        x = float(pt[0])
                        y = float(pt[1])
                        # Include confidence if available
                        if len(pt) >= 3:
                            confidence = float(pt[2])
                            person_keypoints.append([x, y, confidence])
                        else:
                            person_keypoints.append([x, y])
                    
                    if person_keypoints:
                        detections["keypoints"].append(person_keypoints)
                        detections["has_keypoints"] = True
                        
            elif hasattr(kp, "xy") and kp.xy is not None:
                # Alternative format: just [x, y] coordinates
                xy = kp.xy.tolist()
                for person in xy:
                    person_keypoints = [
                        [float(p[0]), float(p[1])] 
                        for p in person 
                        if p is not None and len(p) >= 2
                    ]
                    if person_keypoints:
                        detections["keypoints"].append(person_keypoints)
                        detections["has_keypoints"] = True
        except Exception:  # noqa: BLE001
            # If keypoint extraction fails, continue without keypoints
            pass
    
    return detections





#================================================================================================
# Keep old methods for backward compatibility (deprecated)
def extract_detections_from_result(result) -> Tuple[List[List[float]], List[str], List[float]]:
    """
    DEPRECATED: Use extract_all_detections() instead.
    Kept for backward compatibility.
    """
    detections = extract_all_detections(result)
    return detections["boxes"], detections["classes"], detections["scores"]


def extract_keypoints_from_result(result) -> List[List[List[float]]]:
    """
    DEPRECATED: Use extract_all_detections() instead.
    Kept for backward compatibility.
    """
    detections = extract_all_detections(result)
    return detections["keypoints"]


def extract_detections_from_weapon_result(result) -> Tuple[List[List[float]], List[str], List[float]]:
    """
    DEPRECATED: Use extract_all_detections(result, normalize_classes=True) instead.
    Kept for backward compatibility.
    """
    detections = extract_all_detections(result, normalize_classes=True)
    return detections["boxes"], detections["classes"], detections["scores"]