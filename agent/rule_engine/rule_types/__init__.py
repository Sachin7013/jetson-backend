"""
Rule type implementations live here.
Each module registers its rule via the @register_rule decorator.

Import the concrete rule modules so that they are registered on import.
This keeps things simple for callers: importing this package is enough
to populate the global rules registry.
"""

# NOTE: These imports are intentionally unused; importing the modules
# has the side‑effect of registering their handlers in rules_registry.
from agent.rule_engine.rule_types import accident_presence  # noqa: F401
from agent.rule_engine.rule_types import class_presence  # noqa: F401
from agent.rule_engine.rule_types import count_at_least  # noqa: F401
from agent.rule_engine.rule_types import class_count  # noqa: F401
from agent.rule_engine.rule_types import weapon_detection  # noqa: F401

