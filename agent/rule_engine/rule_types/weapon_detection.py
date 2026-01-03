"""
Rule: weapon_detection
----------------------

Schema:
{
  "type": "weapon_detection",
  "class": "gun" | "knife",
  "label": <optional custom label>,
  "min_count": <optional minimum count>
}
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent.rule_engine.registry import register_rule


@register_rule("weapon_detection")
def evaluate_weapon_detection(rule: Dict[str, Any], detections: Dict[str, Any], task: Dict[str, Any], rule_state: Dict[str, Any], now: datetime) -> Optional[Dict[str, Any]]:
    """
    Evaluate weapon detection rule against current detections.
    
    Checks if the specified weapon class (gun or knife) is present in detections.
    Supports optional min_count to require multiple detections.
    """
    detected_classes = [str(c).lower() for c in (detections.get("classes") or [])]
    if not detected_classes:
        rule_state["last_matched_since"] = None
        return None
    
    # Get weapon class from rule
    rule_class = str(rule.get("class") or "").strip().lower()
    if not rule_class:
        rule_state["last_matched_since"] = None
        return None
    
    # Normalize weapon class names (handle variations)
    weapon_classes = {
        "gun": ["gun", "guns", "pistol", "rifle", "weapon"],
        "knife": ["knife", "knives", "knif", "blade"]
    }
    
    # Find matching weapon class
    target_weapon_classes = []
    for weapon_type, variations in weapon_classes.items():
        if rule_class == weapon_type or rule_class in variations:
            target_weapon_classes = variations
            break
    
    if not target_weapon_classes:
        # If no match in predefined list, use rule_class directly
        target_weapon_classes = [rule_class]
    
    # Count occurrences of weapon in detections
    weapon_count = sum(1 for cls in detected_classes if any(weapon_cls in cls for weapon_cls in target_weapon_classes))
    
    if weapon_count == 0:
        rule_state["last_matched_since"] = None
        return None
    
    # Check min_count if specified
    min_count = rule.get("min_count")
    if min_count is not None and min_count != "null":
        try:
            min_count_int = int(min_count)
            if weapon_count < min_count_int:
                rule_state["last_matched_since"] = None
                return None
        except (ValueError, TypeError):
            pass  # Invalid min_count, ignore it
    
    # Generate label
    label = rule.get("label")
    if not label:
        weapon_display = rule_class.capitalize()
        if weapon_count > 1:
            label = f"{weapon_display}s detected ({weapon_count})"
        else:
            label = f"{weapon_display} detected"
    
    # Update rule state
    rule_state["last_matched_since"] = now
    
    return {
        "label": label,
        "matched_classes": [rule_class],
        "weapon_count": weapon_count
    }

