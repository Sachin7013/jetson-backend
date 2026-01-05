"""
Unified rule engine
-------------------

Engine is rule-agnostic. It delegates rule evaluation to handlers registered
in rules_registry via the register_rule decorator.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
import os
import sys

# Allow running this file directly by ensuring project root is on sys.path
if __package__ is None or __package__ == "":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

from agent.rule_engine.registry import rules_registry

# Import rule types so that their @register_rule decorators run and
# populate the global rules_registry. These imports are used only for
# their side effects.
from agent.rule_engine import rule_types  # noqa: F401


def evaluate_rules(rules: List[Dict[str, Any]], detections: Dict[str, Any], task: Dict[str, Any], state: Dict[int, Dict[str, Any]], now: datetime) -> Optional[List[Dict[str, Any]]]:
    """
    Evaluate a list of rules and return all matching results.

    - rules: list of rule dicts (loaded once at worker start)
    - detections: {'classes': [...], 'scores': [...], 'boxes': [...], 'ts': datetime}
    - task: full task dict (available to handlers)
    - state: dict indexed by rule index, each value is a per-rule state dict
    - now: current timestamp

    Returns:
      List of dicts [{'label': str, 'rule_index': int, ...}, ...] or None if no matches
    """
    # Print all registered rule types before evaluating
    print(f"[evaluate_rules] Registered rules: {list(rules_registry.keys())}")
    print(f"[evaluate_rules] rules: {rules}")
    
    all_results = []
    for rule_index, rule in enumerate(rules or []):
        print(f"[evaluate_rules] rule: {rule}")
        rule_type = (rule.get("type") or "").strip().lower()
        print(f"[evaluate_rules] rule_type: {rule_type}")
        handler = rules_registry.get(rule_type)
        if handler is None:
            continue
        rule_state = state.setdefault(rule_index, {"last_matched_since": None})
        evaluation_result = handler(rule, detections, task, rule_state, now)
        if evaluation_result and isinstance(evaluation_result, dict) and evaluation_result.get("label"):
            # Ensure rule_index is set
            evaluation_result.setdefault("rule_index", rule_index)
            print(f"[evaluate_rules] evaluation_result: {evaluation_result}")
            all_results.append(evaluation_result)

    print(f"[evaluate_rules] all_results: {all_results}")
    print(f"[evaluate_rules] returning all_results: len(all_results): {len(all_results)}")
    return all_results if all_results else None


