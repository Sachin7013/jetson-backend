"""
Detections utilities
--------------------

Unified helpers to extract all types of detections from YOLO results.
This module provides a single, generic method that works for all model types.

Confidence Thresholds:
- General objects: confidence >= 0.7 (reduces false positives)
- Weapons: confidence >= 0.5 (weapons are harder to detect, need lower threshold)
- Lower confidence detections are filtered out
"""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# General confidence threshold: Only keep detections with confidence >= 0.7
# This helps reduce false positives (incorrect detections)
# Value range: 0.0 (accept all) to 1.0 (very strict)
CONFIDENCE_THRESHOLD = 0.7

# Weapon detection confidence threshold: Only keep weapon detections with confidence >= 0.5
# Weapons are harder to detect, so we use a lower threshold to catch more weapon detections
# This helps identify weapons that might be missed with the higher general threshold
WEAPON_DETECTION_CONFIDENCE_THRESHOLD = 0.5

# List of weapon class names (all variations)
# These classes will use the lower weapon threshold (0.5) instead of general threshold (0.7)
WEAPON_CLASSES = [
    "gun", "guns", "pistol", "rifle", "1",  # Gun variations
    "knife", "knives", "knif", "blade",     # Knife variations
    "weapon", "weapons"                      # General weapon terms
]


def is_weapon_class(class_name: str) -> bool:
    """
    Check if a class name is a weapon class.
    
    This function checks if the given class name matches any weapon class
    (gun, knife, pistol, rifle, blade, etc.) so we can use a lower confidence
    threshold for weapons.
    
    Args:
        class_name: The class name to check (case-insensitive)
    
    Returns:
        True if the class is a weapon, False otherwise
    """
    if not class_name:
        return False
    
    class_name_lower = str(class_name).lower().strip()
    
    # Check if class name matches any weapon class
    for weapon_class in WEAPON_CLASSES:
        weapon_class_lower = weapon_class.lower()
        # Exact match or contains check
        if weapon_class_lower == class_name_lower or weapon_class_lower in class_name_lower or class_name_lower in weapon_class_lower:
            return True
    
    return False


def extract_all_detections(result, model_id: str = "") -> Dict[str, Any]:
    """
    Unified method to extract ALL types of detections from any YOLO result.
    
    This is the unified method for extracting all types of detections from YOLO results.
    
    What it extracts:
    - Bounding boxes (x1, y1, x2, y2) for all detected objects
    - Class names for each detection (original case, not normalized)
    - Confidence scores for each detection
    - Keypoints (if the model supports pose detection)
    - Masks (if the model supports segmentation)
    
    Note: Class names are kept in original case. Rules normalize classes internally if needed.
    
    Args:
        result: YOLO result object from model inference
        model_id: Optional model identifier (for debugging)
    
    Returns:
        Dictionary with:
        {
            "boxes": [[x1, y1, x2, y2], ...],
            "classes": ["person", "car", ...],
            "scores": [0.95, 0.87, ...],
            "keypoints": [[[x, y, conf], ...], ...],  # One list per person
            "has_keypoints": bool,
            "masks": [np.ndarray, ...],  # Binary masks for segmentation (one per detection)
            "has_masks": bool,
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
        "masks": [],  # Segmentation masks
        "has_masks": False,
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
            # We'll filter by confidence threshold based on class type (weapon vs general)
            for detection_index, box in enumerate(xyxy):
                # Get class name first (we need this to determine which threshold to use)
                class_id = int(cls_list[detection_index]) if detection_index < len(cls_list) else -1
                class_name = str(class_names_by_id.get(class_id, str(class_id)))
                
                # Get confidence score
                score = float(conf_list[detection_index]) if detection_index < len(conf_list) else 0.0
                
                # IMPORTANT: Use different confidence thresholds based on class type
                # - Weapons: Use lower threshold (0.5) because they're harder to detect
                # - Other objects: Use higher threshold (0.7) to reduce false positives
                is_weapon = is_weapon_class(class_name)
                required_threshold = WEAPON_DETECTION_CONFIDENCE_THRESHOLD if is_weapon else CONFIDENCE_THRESHOLD
                
                # Only add detection if confidence score meets the required threshold
                if score >= required_threshold:
                    # Box coordinates: [x1, y1, x2, y2]
                    detections["boxes"].append([
                        float(box[0]), 
                        float(box[1]), 
                        float(box[2]), 
                        float(box[3])
                    ])
                    
                    # Class name (keep original case - rules normalize internally if needed)
                    detections["classes"].append(class_name)
                    
                    # Confidence score (already checked above, but store it)
                    detections["scores"].append(score)
                # If confidence < threshold, we skip this detection (don't add it to any list)
                
        except Exception:  # noqa: BLE001
            # If extraction fails, continue with empty boxes (keypoints might still work)
            pass
    
    # Step 2: Extract keypoints (only if model supports pose detection)
    # Pose models have a "keypoints" attribute
    # NOTE: Keypoints should align with boxes - we only keep keypoints for boxes that passed confidence threshold
    kp = getattr(result, "keypoints", None)
    if kp is not None:
        try:
            # Get the original confidence scores and class names to match keypoints with boxes
            # We need to know which boxes passed the confidence threshold (weapon vs general)
            original_conf_list = result.boxes.conf.tolist() if hasattr(result, "boxes") and hasattr(result.boxes, "conf") else []
            original_cls_list = result.boxes.cls.tolist() if hasattr(result, "boxes") and hasattr(result.boxes, "cls") else []
            class_names_by_id = result.names if hasattr(result, "names") else {}
            
            # Try to get keypoints data
            if hasattr(kp, "data") and kp.data is not None:
                # Format: data contains [x, y, confidence] for each keypoint
                data = kp.data.tolist()
                for person_index, person in enumerate(data):
                    # Check if this person's box passed the confidence threshold
                    # Keypoints align with boxes by index
                    if person_index < len(original_conf_list):
                        person_confidence = float(original_conf_list[person_index])
                        
                        # Get class name to determine which threshold to use
                        class_id = int(original_cls_list[person_index]) if person_index < len(original_cls_list) else -1
                        class_name = str(class_names_by_id.get(class_id, str(class_id)))
                        is_weapon = is_weapon_class(class_name)
                        required_threshold = WEAPON_DETECTION_CONFIDENCE_THRESHOLD if is_weapon else CONFIDENCE_THRESHOLD
                        
                        # Only extract keypoints if the corresponding box passed confidence threshold
                        if person_confidence < required_threshold:
                            continue  # Skip this person's keypoints
                    
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
                original_conf_list = result.boxes.conf.tolist() if hasattr(result, "boxes") and hasattr(result.boxes, "conf") else []
                original_cls_list = result.boxes.cls.tolist() if hasattr(result, "boxes") and hasattr(result.boxes, "cls") else []
                class_names_by_id = result.names if hasattr(result, "names") else {}
                
                for person_index, person in enumerate(xy):
                    # Check if this person's box passed the confidence threshold
                    if person_index < len(original_conf_list):
                        person_confidence = float(original_conf_list[person_index])
                        
                        # Get class name to determine which threshold to use
                        class_id = int(original_cls_list[person_index]) if person_index < len(original_cls_list) else -1
                        class_name = str(class_names_by_id.get(class_id, str(class_id)))
                        is_weapon = is_weapon_class(class_name)
                        required_threshold = WEAPON_DETECTION_CONFIDENCE_THRESHOLD if is_weapon else CONFIDENCE_THRESHOLD
                        
                        # Only extract keypoints if the corresponding box passed confidence threshold
                        if person_confidence < required_threshold:
                            continue  # Skip this person's keypoints
                    
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
    
    # Step 3: Extract masks (only if model supports segmentation)
    # Segmentation models have a "masks" attribute
    # NOTE: Masks should align with boxes - we only keep masks for boxes that passed confidence threshold
    masks_attr = getattr(result, "masks", None)
    if masks_attr is not None:
        try:
            # Get original confidence scores and class names to match masks with boxes
            # We need to know which boxes passed the confidence threshold (weapon vs general)
            original_conf_list = result.boxes.conf.tolist() if hasattr(result, "boxes") and hasattr(result.boxes, "conf") else []
            original_cls_list = result.boxes.cls.tolist() if hasattr(result, "boxes") and hasattr(result.boxes, "cls") else []
            class_names_by_id = result.names if hasattr(result, "names") else {}
            
            # Get original image dimensions
            if hasattr(result, "orig_shape"):
                orig_height, orig_width = result.orig_shape[:2]
            else:
                # Fallback: try to get from boxes
                if detections["boxes"]:
                    max_y = max([box[3] for box in detections["boxes"]])
                    max_x = max([box[2] for box in detections["boxes"]])
                    orig_height, orig_width = int(max_y), int(max_x)
                else:
                    orig_height, orig_width = 640, 640  # Default fallback
            
            # YOLO v8 segmentation: Extract masks from result.masks
            # Try multiple methods to get masks
            import cv2
            
            masks_extracted = False
            
            # Method 1: Try using result.masks.data (tensor format)
            if hasattr(masks_attr, "data") and masks_attr.data is not None:
                try:
                    # Get mask data - convert to numpy if it's a tensor
                    if hasattr(masks_attr.data, "cpu"):
                        mask_tensors = masks_attr.data.cpu().numpy()
                    elif hasattr(masks_attr.data, "numpy"):
                        mask_tensors = masks_attr.data.numpy()
                    else:
                        mask_tensors = np.array(masks_attr.data)
                    
                    # Check if we got valid mask data (not all zeros)
                    if mask_tensors.size > 0 and mask_tensors.max() > 0:
                        # Process each mask tensor
                        # Shape is typically (num_masks, mask_height, mask_width)
                        if len(mask_tensors.shape) == 3:
                            num_masks = mask_tensors.shape[0]
                            for mask_idx in range(num_masks):
                                # Check if this mask's box passed the confidence threshold
                                # Masks align with boxes by index
                                if mask_idx < len(original_conf_list):
                                    mask_confidence = float(original_conf_list[mask_idx])
                                    
                                    # Get class name to determine which threshold to use
                                    class_id = int(original_cls_list[mask_idx]) if mask_idx < len(original_cls_list) else -1
                                    class_name = str(class_names_by_id.get(class_id, str(class_id)))
                                    is_weapon = is_weapon_class(class_name)
                                    required_threshold = WEAPON_DETECTION_CONFIDENCE_THRESHOLD if is_weapon else CONFIDENCE_THRESHOLD
                                    
                                    # Only extract mask if the corresponding box passed confidence threshold
                                    if mask_confidence < required_threshold:
                                        continue  # Skip this mask
                                
                                mask_tensor = mask_tensors[mask_idx]
                                
                                # Skip if mask is all zeros
                                if mask_tensor.max() == 0:
                                    continue
                                
                                # Convert to binary mask (0 or 255)
                                if mask_tensor.dtype != np.uint8:
                                    # Normalize and threshold
                                    if mask_tensor.max() <= 1.0:
                                        mask_binary = (mask_tensor > 0.5).astype(np.uint8) * 255
                                    else:
                                        # Scale to 0-255 range
                                        mask_normalized = (mask_tensor / mask_tensor.max() * 255).astype(np.uint8)
                                        mask_binary = (mask_normalized > 127).astype(np.uint8) * 255
                                else:
                                    mask_binary = mask_tensor
                                
                                # Resize mask to original image dimensions
                                if mask_binary.shape != (orig_height, orig_width):
                                    mask_binary = cv2.resize(
                                        mask_binary, 
                                        (orig_width, orig_height), 
                                        interpolation=cv2.INTER_NEAREST
                                    )
                                
                                detections["masks"].append(mask_binary)
                                detections["has_masks"] = True
                                masks_extracted = True
                        
                        elif len(mask_tensors.shape) == 2:
                            # Single mask - check confidence if we have boxes
                            if len(original_conf_list) > 0:
                                mask_confidence = float(original_conf_list[0])
                                
                                # Get class name to determine which threshold to use
                                class_id = int(original_cls_list[0]) if len(original_cls_list) > 0 else -1
                                class_name = str(class_names_by_id.get(class_id, str(class_id)))
                                is_weapon = is_weapon_class(class_name)
                                required_threshold = WEAPON_DETECTION_CONFIDENCE_THRESHOLD if is_weapon else CONFIDENCE_THRESHOLD
                                
                                # Only extract mask if the corresponding box passed confidence threshold
                                if mask_confidence < required_threshold:
                                    pass  # Skip this mask
                                else:
                                    mask_tensor = mask_tensors
                                    
                                    # Skip if mask is all zeros
                                    if mask_tensor.max() > 0:
                                        # Convert to binary mask
                                        if mask_tensor.dtype != np.uint8:
                                            if mask_tensor.max() <= 1.0:
                                                mask_binary = (mask_tensor > 0.5).astype(np.uint8) * 255
                                            else:
                                                mask_normalized = (mask_tensor / mask_tensor.max() * 255).astype(np.uint8)
                                                mask_binary = (mask_normalized > 127).astype(np.uint8) * 255
                                        else:
                                            mask_binary = mask_tensor
                                        
                                        # Resize to original image dimensions
                                        if mask_binary.shape != (orig_height, orig_width):
                                            mask_binary = cv2.resize(
                                                mask_binary, 
                                                (orig_width, orig_height), 
                                                interpolation=cv2.INTER_NEAREST
                                            )
                                        
                                        detections["masks"].append(mask_binary)
                                        detections["has_masks"] = True
                                        masks_extracted = True
                            else:
                                # No confidence info, extract mask anyway (fallback)
                                mask_tensor = mask_tensors
                                
                                # Skip if mask is all zeros
                                if mask_tensor.max() > 0:
                                    # Convert to binary mask
                                    if mask_tensor.dtype != np.uint8:
                                        if mask_tensor.max() <= 1.0:
                                            mask_binary = (mask_tensor > 0.5).astype(np.uint8) * 255
                                        else:
                                            mask_normalized = (mask_tensor / mask_tensor.max() * 255).astype(np.uint8)
                                            mask_binary = (mask_normalized > 127).astype(np.uint8) * 255
                                    else:
                                        mask_binary = mask_tensor
                                    
                                    # Resize to original image dimensions
                                    if mask_binary.shape != (orig_height, orig_width):
                                        mask_binary = cv2.resize(
                                            mask_binary, 
                                            (orig_width, orig_height), 
                                            interpolation=cv2.INTER_NEAREST
                                        )
                                    
                                    detections["masks"].append(mask_binary)
                                    detections["has_masks"] = True
                                    masks_extracted = True
                
                except Exception as mask_exc:  # noqa: BLE001
                    # Try alternative method
                    pass
            
            # Method 2: Try using polygon coordinates (result.masks.xy) if data method failed
            if not masks_extracted and hasattr(masks_attr, "xy") and masks_attr.xy is not None:
                try:
                    # Get polygon coordinates
                    polygons = masks_attr.xy
                    if hasattr(polygons, "cpu"):
                        polygons = polygons.cpu().numpy()
                    elif hasattr(polygons, "numpy"):
                        polygons = polygons.numpy()
                    else:
                        polygons = np.array(polygons)
                    
                    # Create binary masks from polygons
                    # Only keep masks for boxes that passed confidence threshold
                    for polygon_index, polygon in enumerate(polygons):
                        # Check if this mask's box passed the confidence threshold
                        if polygon_index < len(original_conf_list):
                            polygon_confidence = float(original_conf_list[polygon_index])
                            
                            # Get class name to determine which threshold to use
                            class_id = int(original_cls_list[polygon_index]) if polygon_index < len(original_cls_list) else -1
                            class_name = str(class_names_by_id.get(class_id, str(class_id)))
                            is_weapon = is_weapon_class(class_name)
                            required_threshold = WEAPON_DETECTION_CONFIDENCE_THRESHOLD if is_weapon else CONFIDENCE_THRESHOLD
                            
                            # Only extract mask if the corresponding box passed confidence threshold
                            if polygon_confidence < required_threshold:
                                continue  # Skip this mask
                        
                        if polygon is None or len(polygon) == 0:
                            continue
                        
                        # Create empty mask
                        mask_binary = np.zeros((orig_height, orig_width), dtype=np.uint8)
                        
                        # Convert polygon to integer coordinates
                        pts = polygon.reshape(-1, 2).astype(np.int32)
                        
                        # Fill polygon in mask
                        cv2.fillPoly(mask_binary, [pts], 255)
                        
                        detections["masks"].append(mask_binary)
                        detections["has_masks"] = True
                        masks_extracted = True
                
                except Exception:  # noqa: BLE001
                    pass
                    
        except Exception as e:  # noqa: BLE001
            # If mask extraction fails, continue without masks
            # This is expected for non-segmentation models
            pass
    
    # Summary: All detections have been filtered by confidence threshold
    # - Weapons: Use lower threshold (>= 0.5) because they're harder to detect
    # - Other objects: Use higher threshold (>= 0.7) to reduce false positives
    # - Boxes, classes, and scores: Filtered during extraction (Step 1) using appropriate threshold
    # - Keypoints: Only extracted for boxes that passed confidence threshold (Step 2)
    # - Masks: Only extracted for boxes that passed confidence threshold (Step 3)
    # All arrays (boxes, classes, scores, keypoints, masks) are aligned and filtered consistently
    
    return detections




