"""
Simple model registry: register a model name and a small loader function.
When a model_id is requested, we load it if registered; otherwise we return None.
"""
from typing import Optional
from .models import _MODEL_REGISTRY, Model


def load_model(model_id: Optional[str]) -> Optional[Model]:
    """Load a model by id/name from the registry only.
    - If empty/None, print a clear message and return None.
    - If the id is not registered, return None and print a clear message.
    """
    key = (model_id or "").strip().lower()
    if not key:
        print("❌ Model id not provided")
        return None

    loader = _MODEL_REGISTRY.get(key)
    if not loader:
        print(f"❌ Model not found in registry: '{key}'")
        return None

    try:
        return loader(key)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Failed to load model '{key}': {exc}")
        return None

