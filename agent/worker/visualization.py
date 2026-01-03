"""
Visualization Configuration
---------------------------

This module analyzes the rules to determine what should be drawn on frames.
Instead of hard-coded priorities, we build a configuration that tells the
drawing system exactly what to visualize based on the active rules.
"""
from typing import Dict, Any, List, Set, Tuple


def analyze_rules_for_visualization(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze rules to determine what should be visualized on frames.
    
    This replaces the old hard-coded priority system (weapon > pose > boxes).
    Now, we analyze the rules and create a configuration that tells the
    drawing system what to draw.
    
    Args:
        rules: List of rule dictionaries from the task
    
    Returns:
        Visualization configuration dictionary:
        {
            "draw_boxes": bool,
            "box_classes": Set[str],           # Which classes to draw boxes for
            "draw_keypoints": bool,
            "draw_weapon_overlays": bool,      # Special weapon detection overlays
            "weapon_overlay_classes": Set[str], # Which classes get weapon overlays
            "colors": Dict[str, Tuple[int, int, int]],  # Color scheme
            "label_format": str
        }
    """
    # Start with default configuration
    config = {
        "draw_boxes": False,
        "box_classes": set(),
        "draw_keypoints": False,
        "draw_weapon_overlays": False,
        "weapon_overlay_classes": set(),
        "colors": {},
        "label_format": "class_score"  # Options: "class", "class_score", "custom"
    }
    
    if not rules:
        return config
    
    # Analyze each rule to determine visualization needs
    for rule in rules:
        rule_type = str(rule.get("type", "")).lower()
        
        # Weapon detection rules need special visualization
        if rule_type == "weapon_detection":
            config["draw_weapon_overlays"] = True
            config["draw_boxes"] = True
            
            # Get weapon class from rule
            weapon_class = str(rule.get("class", "")).lower()
            if weapon_class:
                # Add weapon variations
                if weapon_class == "gun":
                    config["weapon_overlay_classes"].update(["gun", "guns", "pistol", "rifle", "weapon"])
                elif weapon_class == "knife":
                    config["weapon_overlay_classes"].update(["knife", "knives", "knif", "blade"])
                else:
                    config["weapon_overlay_classes"].add(weapon_class)
                
                # Also draw boxes for weapons
                config["box_classes"].update(config["weapon_overlay_classes"])
            
            # Always draw person boxes for weapon detection (to show armed persons)
            config["box_classes"].add("person")
        
        # Pose/accident detection rules need keypoints
        elif rule_type in ["accident_presence", "fall_detection"]:
            config["draw_keypoints"] = True
            config["draw_boxes"] = True
            config["box_classes"].add("person")
        
        # Class presence/count rules need boxes for those classes
        elif rule_type in ["class_presence", "class_count", "count_at_least"]:
            config["draw_boxes"] = True
            
            # Get classes from rule (support both "class" and "classes" formats)
            rule_classes = rule.get("classes") or []
            rule_class = rule.get("class") or rule.get("target_class")
            
            if rule_class and not rule_classes:
                rule_classes = [rule_class]
            
            # Add all classes to box_classes
            for cls in rule_classes:
                if isinstance(cls, str):
                    config["box_classes"].add(cls.lower())
    
    # Set up color scheme
    # Default colors for different detection types
    config["colors"] = {
        "default": (0, 255, 0),      # Green for regular detections
        "weapon": (0, 0, 255),       # Red for weapons
        "person": (0, 255, 255),     # Yellow for persons
        "keypoint": (0, 255, 255),   # Yellow for keypoints
        "keypoint_line": (0, 255, 0), # Green for keypoint skeleton lines
    }
    
    return config


def get_class_color(class_name: str, config: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Get the color to use for drawing a specific class.
    
    Args:
        class_name: The class name (e.g., "person", "gun")
        config: Visualization configuration
    
    Returns:
        BGR color tuple (B, G, R) for OpenCV
    """
    class_name_lower = str(class_name).lower()
    colors = config.get("colors", {})
    
    # Check for weapon classes (red)
    weapon_classes = config.get("weapon_overlay_classes", set())
    if any(weapon in class_name_lower for weapon in weapon_classes):
        return colors.get("weapon", (0, 0, 255))
    
    # Check for person (yellow)
    if "person" in class_name_lower:
        return colors.get("person", (0, 255, 255))
    
    # Default color (green)
    return colors.get("default", (0, 255, 0))


def should_draw_box(class_name: str, config: Dict[str, Any]) -> bool:
    """
    Check if a box should be drawn for a given class based on config.
    
    Args:
        class_name: The class name
        config: Visualization configuration
    
    Returns:
        True if box should be drawn, False otherwise
    """
    if not config.get("draw_boxes", False):
        return False
    
    box_classes = config.get("box_classes", set())
    if not box_classes:
        # If no specific classes specified, draw all
        return True
    
    class_name_lower = str(class_name).lower()
    
    # Check if class matches any in box_classes
    for target_class in box_classes:
        if target_class in class_name_lower or class_name_lower in target_class:
            return True
    
    return False

