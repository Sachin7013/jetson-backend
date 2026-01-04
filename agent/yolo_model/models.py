
from ultralytics import YOLO
from typing import Optional
from typing import Any, Callable, Dict



# Type aliases for clarity
Model = Any
ModelLoader = Callable[[str], Optional[Model]]  # receives the requested model id/name

# Internal registry mapping a simple name to a loader function
_MODEL_REGISTRY: Dict[str, ModelLoader] = {}


def register_model(name: str) -> Callable[[ModelLoader], ModelLoader]:
    """
    Register a model loader under a simple name.

    Example:
        @register_model("yolov8n")
        def load_v8n(_name: str):
            return YOLO("yolov8n.pt")
    """
    def decorator(func: ModelLoader) -> ModelLoader:
        _MODEL_REGISTRY[name.lower()] = func
        return func

    return decorator



@register_model("yolov8m.pt")
def _yolov8m_loader(_name: str) -> Optional[Model]:
    if YOLO is None:
        print("⚠️ YOLO not available (ultralytics not installed). Skipping detection.")
        return None
    return YOLO("yolov8m.pt")

@register_model("yolov8m-seg.pt")
def _yolov8m_seg_loader(_name: str) -> Optional[Model]:
    if YOLO is None:
        print("==================================================⚠️ YOLO not available (ultralytics not installed). Skipping detection.==================================================")
        return None
    print("==================================================Loading yolov8m-seg.pt==================================================")    
    return YOLO("yolov8m-seg.pt")    


@register_model("yolov8n.pt")
def _yolov8n_loader(_name: str) -> Optional[Model]:
    if YOLO is None:
        print("⚠️ YOLO not available (ultralytics not installed). Skipping detection.")
        return None
    return YOLO("yolov8n.pt")


@register_model("yolov8s.pt")
def _yolov8s_loader(_name: str) -> Optional[Model]:
    if YOLO is None:
        print("⚠️ YOLO not available (ultralytics not installed). Skipping detection.")
        return None
    return YOLO("yolov8s.pt")

@register_model("yolov8n-pose.pt")
def _yolov8n_pose_loader(_name: str) -> Optional[Model]:
    if YOLO is None:
        print("⚠️ YOLO not available (ultralytics not installed). Skipping detection.")
        return None
    return YOLO("yolov8n-pose.pt")  


@register_model("yolov8s-pose.pt")
def _yolov8s_pose_loader(_name: str) -> Optional[Model]:
    if YOLO is None:
        print("⚠️ YOLO not available (ultralytics not installed). Skipping detection.")
        return None
    return YOLO("yolov8s-pose.pt")     

@register_model("epoch150.pt")
def _yolov8_weapon_detection_1(_name: str) -> Optional[Model]:
    if YOLO is None:
        print("⚠️ YOLO not available (ultralytics not installed). Skipping detection.")
        return None
    # Load the trained weapon detection model (guns and knives)
    trained_model_path = r"C:\Users\Manoj\Downloads\epoch150.pt"
    print(f"==================================================Loading weapon detection model from {trained_model_path}==================================================")
    return YOLO(trained_model_path)
