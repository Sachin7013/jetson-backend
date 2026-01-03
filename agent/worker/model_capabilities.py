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
    Detect what a model can do by examining its structure and name.
    
    This is a simple, fast check that doesn't require running inference.
    We look at:
    - Model name/ID (e.g., "pose" in name = pose detection)
    - Model attributes (e.g., has keypoints support)
    
    Args:
        model: The YOLO model object
        model_id: The model identifier (e.g., "yolov8n-pose.pt")
    
    Returns:
        Dictionary with capabilities:
        {
            "model_id": str,
            "has_boxes": bool,        # Can detect objects with boxes
            "has_keypoints": bool,     # Can detect pose keypoints
            "detection_type": str,     # "object", "pose", "weapon", etc.
            "likely_classes": List[str] # Common classes this model detects
        }
    """
    model_id_lower = str(model_id).lower()
    
    # Initialize capabilities
    capabilities = {
        "model_id": model_id,
        "has_boxes": True,  # All YOLO models have boxes
        "has_keypoints": False,
        "detection_type": "object",  # Default to object detection
        "likely_classes": []
    }
    
    # Check if it's a pose model (by name)
    if "pose" in model_id_lower:
        capabilities["has_keypoints"] = True
        capabilities["detection_type"] = "pose"
        capabilities["likely_classes"] = ["person"]
    
    # Check if it's a weapon detection model
    elif "weapon" in model_id_lower or "gun" in model_id_lower or "knife" in model_id_lower:
        capabilities["detection_type"] = "weapon"
        capabilities["likely_classes"] = ["gun", "knife", "weapon"]
    
    # Try to inspect model structure (if possible)
    try:
        # Check if model has keypoint-related attributes
        if hasattr(model, "model"):
            model_structure = str(model.model)
            if "keypoint" in model_structure.lower() or "pose" in model_structure.lower():
                capabilities["has_keypoints"] = True
                capabilities["detection_type"] = "pose"
    except Exception:  # noqa: BLE001
        # If inspection fails, rely on name-based detection
        pass
    
    return capabilities


def get_models_needed_by_rules(
    rules: List[Dict[str, Any]], 
    available_models: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Determine which models are needed based on the active rules.
    
    This helps optimize performance by only processing models that are actually needed.
    
    Args:
        rules: List of rule dictionaries from the task
        available_models: List of model capability dictionaries
    
    Returns:
        List of model capabilities that should be used
    """
    if not rules:
        # If no rules, use all models (backward compatibility)
        return available_models
    
    # Analyze rules to determine what we need
    needs_pose = False
    needs_weapons = False
    needs_objects = False
    
    for rule in rules:
        rule_type = str(rule.get("type", "")).lower()
        
        # Check rule types that need specific model capabilities
        if rule_type in ["accident_presence", "fall_detection"]:
            needs_pose = True
            needs_objects = True
        
        if rule_type == "weapon_detection":
            needs_weapons = True
            needs_objects = True
        
        # Most rules need object detection
        if rule_type in ["class_presence", "class_count", "count_at_least"]:
            needs_objects = True
    
    # Filter models based on needs
    selected_models = []
    
    for model_cap in available_models:
        model_type = model_cap.get("detection_type", "object")
        
        # Select models that match our needs
        if needs_weapons and model_type == "weapon":
            selected_models.append(model_cap)
        elif needs_pose and model_type == "pose":
            selected_models.append(model_cap)
        elif needs_objects and model_type == "object":
            selected_models.append(model_cap)
        elif not (needs_weapons or needs_pose):
            # If no specific needs, use all models
            selected_models.append(model_cap)
    
    # If no models selected, use all (safety fallback)
    if not selected_models:
        return available_models
    
    return selected_models

