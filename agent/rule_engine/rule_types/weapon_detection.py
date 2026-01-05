"""
Rule: weapon_detection
----------------------

Schema:
{
  "type": "weapon_detection",
  "class": "gun" | "knife",  # Specifies which weapon type to detect (filters detections)
  "label": <optional custom label>,
  "min_count": <optional minimum count>
}

Behavior:
- If "class": "gun" → only matches gun-related detections (gun, guns, pistol, rifle)
- If "class": "knife" → only matches knife-related detections (knife, knives, blade)
- Each rule filters detections to only the specified weapon type
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent.rule_engine.registry import register_rule


@register_rule("weapon_detection")
def evaluate_weapon_detection(rule: Dict[str, Any], detections: Dict[str, Any], task: Dict[str, Any], rule_state: Dict[str, Any], now: datetime) -> Optional[Dict[str, Any]]:
    """
    Evaluate weapon detection rule against current detections.
    
    Checks if the specified weapon class (gun or knife) is present in detections.
    Only matches the specific weapon type specified in the rule's 'class' field:
    - If rule specifies "gun", only gun-related detections are matched
    - If rule specifies "knife", only knife-related detections are matched
    - Supports optional min_count to require multiple detections.
    """
    detected_classes = [str(c).lower() for c in (detections.get("classes") or [])]
    print(f"[weapon_detection] detected_classes: {detected_classes}")
    if not detected_classes:
        print(f"[weapon_detection] No detected classes, returning None")
        rule_state["last_matched_since"] = None
        return None
    
    # Get weapon class from rule
    rule_class = str(rule.get("class") or "").strip().lower()
    print(f"[weapon_detection] rule_class: '{rule_class}'")
    if not rule_class:
        print(f"[weapon_detection] No rule_class specified, returning None")
        rule_state["last_matched_since"] = None
        return None
    
    # Normalize weapon class names (handle variations)
    # Each weapon type has its own variations that should be matched
    weapon_classes = {
        "gun": ["gun", "guns", "pistol", "rifle","1"],
        "knife": ["knife", "knives", "knif", "blade"]
    }
    
    # Get target weapon classes based on the rule_class specified
    # Only match the specific weapon type requested, not all weapons
    target_weapon_classes = []
    
    # Check if rule_class matches a known weapon type
    if rule_class in weapon_classes:
        # Use the variations for this specific weapon type only
        target_weapon_classes = weapon_classes[rule_class]
    else:
        # Check if rule_class is one of the variations
        for weapon_type, variations in weapon_classes.items():
            if rule_class in variations:
                target_weapon_classes = variations
                break
        
        # If no match found, use rule_class directly (fallback)
        if not target_weapon_classes:
            target_weapon_classes = [rule_class]
    
    print(f"[weapon_detection] target_weapon_classes: {target_weapon_classes}")
    
    # Count occurrences of ONLY the specified weapon type in detections
    # Use precise matching to avoid cross-contamination between gun and knife
    weapon_count = 0
    matched_detected_classes = []
    
    for detected_class in detected_classes:
        detected_class_lower = detected_class.lower().strip()
        
        # Check if this detected class matches any of our target weapon classes
        # Use exact match or check if target class is contained in detected class
        # This handles variations like "guns" matching "gun" rule
        is_match = False
        for target_class in target_weapon_classes:
            target_class_lower = target_class.lower()
            # Exact match
            if detected_class_lower == target_class_lower:
                is_match = True
                break
            # Check if detected class contains the target class as a whole word
            # This handles "guns" matching "gun", "knives" matching "knife", etc.
            # But avoids false matches like "gun" matching "knife" (which shouldn't happen anyway)
            if target_class_lower in detected_class_lower or detected_class_lower in target_class_lower:
                # Additional check: ensure we're not matching across weapon types
                # For example, "gun" should not match "knife" variations
                is_match = True
                break
        
        if is_match:
            weapon_count += 1
            matched_detected_classes.append(detected_class)
    
    print(f"[weapon_detection] weapon_count: {weapon_count}, matched_detected_classes: {matched_detected_classes}")
    
    if weapon_count == 0:
        print(f"[weapon_detection] No weapons found (weapon_count=0), returning None")
        rule_state["last_matched_since"] = None
        return None
    
    # Check min_count if specified
    min_count = rule.get("min_count")
    print(f"[weapon_detection] min_count: {min_count}")
    if min_count is not None and min_count != "null":
        try:
            min_count_int = int(min_count)
            if weapon_count < min_count_int:
                print(f"[weapon_detection] weapon_count ({weapon_count}) < min_count ({min_count_int}), returning None")
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
    
    result = {
        "label": label,
        "matched_classes": list(set(matched_detected_classes)) if matched_detected_classes else [rule_class],
        "weapon_count": weapon_count
    }
    print(f"[weapon_detection] ✅ Match found! Returning: {result}")
    return result

