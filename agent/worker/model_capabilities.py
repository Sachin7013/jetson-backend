"""
Model Capabilities Detection
-----------------------------

This module helps identify what each YOLO model can do:
- Detect objects with bounding boxes?
- Detect pose keypoints?
- What classes can it detect?

This information helps us:
1. Only process models that are needed by the rules
2. Know what data to extract from each model
3. Optimize performance by skipping unnecessary models
"""
from typing import Dict, Any, List, Optional, Set


def detect_model_capabilities(model, model_id: str) -> Dict[str, Any]:
    """
    Detect what a model can do by examining its structure and classes.
    
    This is a generic check that inspects the model itself, not hard-coded filenames.
    We look at:
    - Model structure (e.g., has keypoints/masks support)
    - Model class names (e.g., detects weapons, persons, etc.)
    
    Args:
        model: The YOLO model object
        model_id: The model identifier (e.g., "yolov8n-pose.pt")
    
    Returns:
        Dictionary with capabilities:
        {
            "model_id": str,
            "has_boxes": bool,        # Can detect objects with boxes
            "has_keypoints": bool,     # Can detect pose keypoints
            "has_masks": bool,        # Can detect objects with masks
            "detection_type": str,     # "object", "pose", "weapon", etc.
            "likely_classes": List[str] # Common classes this model detects
        }
    """
    # Initialize capabilities
    capabilities = {
        "model_id": model_id,
        "has_boxes": True,  # All YOLO models have boxes
        "has_keypoints": False,
        "has_weapons": False,
        "detection_type": "object", 
        "has_masks": False,
        "likely_classes": []
    }
    
    # Inspect model structure to detect capabilities
    try:
        # Get model class names (what it can detect)
        class_names = {}
        if hasattr(model, "names") and model.names:
            class_names = model.names
        elif hasattr(model, "model") and hasattr(model.model, "names") and model.model.names:
            class_names = model.model.names
        
        # Convert to list of class name strings (keep original, don't normalize here)
        class_list = []
        if isinstance(class_names, dict):
            class_list = [str(name) for name in class_names.values()]
        elif isinstance(class_names, list):
            class_list = [str(name) for name in class_names]
        
        # Store original class names (normalization happens later in extract_all_detections if needed)
        capabilities["likely_classes"] = class_list
        
        # Check for segmentation/mask support FIRST (before other checks)
        # This is important because segmentation models should be detected as segmentation
        # even if they have "pose" or other keywords in their structure
        has_mask_support = False
        try:
            # First, check if model has a task attribute (YOLO models often have this)
            if hasattr(model, "task"):
                task = str(model.task).lower()
                if task == "segment" or "seg" in task:
                    has_mask_support = True
            # Also check model structure as fallback
            if not has_mask_support and hasattr(model, "model"):
                model_str = str(model.model).lower()
                # Check for segmentation-related layers
                if "seg" in model_str or "segment" in model_str or "mask" in model_str:
                    has_mask_support = True
        except Exception:  # noqa: BLE001
            pass
        
        # Check for pose/keypoint support by inspecting model structure
        # Only check if not already detected as segmentation
        has_keypoint_support = False
        if not has_mask_support:
            try:
                # First, check if model has a task attribute
                if hasattr(model, "task"):
                    task = str(model.task).lower()
                    if task == "pose" or "pose" in task:
                        has_keypoint_support = True
                # Also check model structure as fallback
                if not has_keypoint_support and hasattr(model, "model"):
                    model_str = str(model.model).lower()
                    # Check for keypoint-related layers/attributes
                    if "keypoint" in model_str or "pose" in model_str:
                        has_keypoint_support = True
            except Exception:  # noqa: BLE001
                pass
        
        # Detect weapon model by checking if it detects weapon-related classes
        # Only check if not already detected as segmentation or pose
        # Use strict matching (exact match or word boundaries) to avoid false positives
        has_weapon_classes = False
        if not has_mask_support and not has_keypoint_support:
            weapon_keywords = ["gun", "knife", "weapon", "pistol", "rifle", "blade"]
            # Normalize class names only for comparison (temporary, not stored)
            class_list_lower = [cls.lower().strip() for cls in class_list]
            # Check for exact matches or word boundaries (not just substring)
            for cls_lower in class_list_lower:
                for keyword in weapon_keywords:
                    keyword_lower = keyword.lower()
                    # Exact match or word boundary match
                    if cls_lower == keyword_lower or \
                       cls_lower.startswith(keyword_lower + " ") or \
                       cls_lower.endswith(" " + keyword_lower) or \
                       (" " + keyword_lower + " ") in (" " + cls_lower + " "):
                        has_weapon_classes = True
                        break
                if has_weapon_classes:
                    break
        
        # Determine detection type based on capabilities
        # Priority: segmentation > weapon > pose > object
        if has_mask_support:
            capabilities["detection_type"] = "object"
            capabilities["has_masks"] = True
        elif has_weapon_classes:
            capabilities["detection_type"] = "weapon"
            capabilities["has_weapons"] = True
        elif has_keypoint_support:
            capabilities["detection_type"] = "pose"
            capabilities["has_keypoints"] = True
        else:
            capabilities["detection_type"] = "object"
            
    except Exception:  # noqa: BLE001
        # If inspection fails, default to generic object detection
        # This is a fallback - model will still work but may not be optimized
        pass
    
    return capabilities


def get_models_needed_by_rules(
    rules: List[Dict[str, Any]], 
    available_models: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    OPTIMIZATION: Only return models that are actually needed by the rules.
    
    Purpose: Instead of running ALL models on every frame, we only run the models
    that are required by the active rules. This saves computation time.
    
    Example:
    - If rules need weapon detection → only use weapon models
    - If rules need pose detection → only use pose models  
    - If rules need masks → only use segmentation models
    - If no specific needs → use all models
    
    Args:
        rules: List of rule dictionaries from the task (e.g., [{"type": "weapon_detection"}, ...])
        available_models: List of all loaded model capabilities
    
    Returns:
        List of model capabilities that match the rules (or all models if no match)
    """
    # If no rules, use all models
    if not rules:
        return available_models
    
    # Step 1: Figure out what capabilities the rules need
    required_capabilities = {
        "weapon": False,  # Need weapon detection model?
        "pose": False,    # Need pose/keypoint model?
        "masks": False,   # Need segmentation model?
        "object": False  # Need regular object detection?
    }
    
    for rule in rules:
        rule_type = str(rule.get("type", "")).lower()
        
        # Map rule types to required capabilities
        if rule_type == "weapon_detection":
            required_capabilities["weapon"] = True
            required_capabilities["object"] = True
        
        elif rule_type in ["accident_presence", "fall_detection"]:
            required_capabilities["pose"] = True
            required_capabilities["object"] = True
        
        elif rule_type in ["class_presence", "class_count", "count_at_least"]:
            required_capabilities["masks"] = True
            required_capabilities["object"] = True
    
    # Step 2: If no specific requirements, use all models
    has_specific_requirements = any([
        required_capabilities["weapon"],
        required_capabilities["pose"],
        required_capabilities["masks"]
    ])
    
    if not has_specific_requirements:
        return available_models
    
    # Step 3: Filter models to only those that match requirements
    matching_models = []
    
    for model in available_models:
        model_type = model.get("detection_type", "object")
        model_has_masks = model.get("has_masks", False)
        
        # Check if this model matches any requirement
        is_needed = False
        
        if required_capabilities["weapon"] and model_type == "weapon":
            is_needed = True
        elif required_capabilities["pose"] and model_type == "pose":
            is_needed = True
        elif required_capabilities["masks"] and model_has_masks:
            is_needed = True
        elif required_capabilities["object"] and model_type == "object":
            is_needed = True
        
        if is_needed:
            matching_models.append(model)
    
    # Step 4: Safety fallback - if no matches, use all models
    if not matching_models:
        return available_models
    
    return matching_models