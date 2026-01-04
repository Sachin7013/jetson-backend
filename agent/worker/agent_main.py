"""
Agent main loop
---------------

Implements the per-agent processing with hybrid behavior:
- Load models
- Open source
- Modes:
  - continuous: read and process frames at agent FPS indefinitely
  - patrol: sleep for interval, then process frames for a short window at agent FPS, repeat
- Merge detections and apply rule engine
- Print alerts, send heartbeat
- Stop on end_at/stop_requested/file end
"""
import time
import os
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from bson import ObjectId
try:
    import cv2  # type: ignore
except Exception:  # noqa: BLE001
    cv2 = None  # type: ignore[assignment]

# Allow running when imported without installed package by ensuring project root is on sys.path
if __package__ is None or __package__ == "":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

from agent.utils.db import get_collection
from agent.utils.utils import iso_now, parse_iso
from agent.yolo_model.yolo_utils import init_yolo_model, check_event_match
from agent.rule_engine.engine import evaluate_rules
from agent.worker.video_io import open_video_capture
from agent.worker.detections import extract_all_detections
from agent.worker.frame_hub import reconstruct_frame
from agent.worker.frame_processor import draw_detections_unified
from agent.worker.model_capabilities import detect_model_capabilities, get_models_needed_by_rules
import numpy as np


def sleep_with_heartbeat(tasks_collection, task_id: str, seconds: int) -> bool:
    """
    Sleep for 'seconds' but wake up every 1 second to:
    - send heartbeat
    - check stop/end
    Returns True if a stop/end condition is met (caller should exit), False otherwise.
    """
    end_time = time.time() + max(0, seconds)
    while time.time() < end_time:
        # Heartbeat
        tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"updated_at": iso_now()}})
        # Stop / end checks
        task_document = tasks_collection.find_one({"_id": ObjectId(task_id)}, projection={"stop_requested": 1, "end_at": 1})
        # If task document was deleted, stop
        if not task_document:
            print(f"[worker {task_id}] ⏹️ Stopping (task deleted)")
            return True
        stop_requested = bool(task_document.get("stop_requested")) if task_document else False
        end_at_dt = parse_iso(task_document.get("end_at")) if task_document else None
        now = datetime.now(timezone.utc)
        if stop_requested or (end_at_dt and now >= end_at_dt):
            status = "completed" if end_at_dt and now >= end_at_dt else "cancelled"
            tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"status": status, "stopped_at": iso_now(), "updated_at": iso_now()}})
            print(f"[worker {task_id}] ⏹️ Stopping (status={status})")
            return True
        time.sleep(1)
    return False


def format_video_time_ms(milliseconds: float) -> str:
    """
    Convert milliseconds to H:MM:SS.mmm string.
    """
    if milliseconds < 0:
        milliseconds = 0
    total_seconds, milliseconds_remainder = divmod(int(milliseconds), 1000)
    hours, remaining_seconds = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remaining_seconds, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}.{milliseconds_remainder:03d}"


def get_video_time_ms(video_capture, frame_index: int, fps: int) -> float:
    """
    Read video time from OpenCV if available, else estimate using frame_index/fps.
    """
    position_ms = -1.0
    if cv2 is not None and video_capture is not None:
        try:
            position_ms = float(video_capture.get(cv2.CAP_PROP_POS_MSEC))
        except Exception:
            position_ms = -1.0
    if position_ms is None or position_ms <= 0:
        if fps > 0:
            return (frame_index / float(fps)) * 1000.0
        return 0.0
    return position_ms


def run_task_worker(task_id: str, shared_store: Optional["Dict[str, Any]"] = None) -> None:
    """
    Standalone agent main loop.
    """
    tasks_collection = get_collection()
    task = tasks_collection.find_one({"_id": ObjectId(task_id)})
    if not task:
        print(f"[worker {task_id}] ❌ Task not found. Exiting.")
        return

    agent_name = task.get("task_name") or f"agent-{task_id}"
    fps = int(task.get("fps", 5))
    model_ids: List[str] = task.get("model_ids", []) or []
    run_mode = (task.get("run_mode") or "continuous").strip().lower()  # "continuous" | "patrol"
    interval_minutes = int(task.get("interval_minutes", 5))  # patrol sleep interval
    check_duration_seconds = int(task.get("check_duration_seconds", 10))  # patrol detection window

    print(f"[worker {task_id}] ▶️ Starting '{agent_name}' | mode={run_mode} fps={fps} models={model_ids}")

    # Load models and detect their capabilities
    models = []
    model_capabilities = []
    for model_id in model_ids:
        model_instance = init_yolo_model({"yolo_model_path": model_id})
        if model_instance is not None:
            models.append(model_instance)
            # Detect what this model can do
            capabilities = detect_model_capabilities(model_instance, model_id)
            model_capabilities.append(capabilities)
            print(f"[worker {task_id}] ✅ Model loaded: {model_id} | type={capabilities['detection_type']} | keypoints={capabilities['has_keypoints']}")
        else:
            print(f"[worker {task_id}] ⚠️ Failed to load model: {model_id}")
    if not models:
        print(f"[worker {task_id}] ❌ No models loaded. Exiting.")
        return

    # Decide if using centralized hub (stream) or direct file capture
    source_uri = (task.get("source_uri") or "").strip()
    camera_id = (task.get("camera_id") or "").strip()
    use_hub = bool(source_uri and camera_id and shared_store is not None)

    # Open source (file fallback only)
    video_capture = None
    if not use_hub:
        video_capture, _capture_source = open_video_capture(task)
        if video_capture is None:
            # For file tasks with no capture, mark completed
            if (task.get("source_type") == "file") or (not task.get("source_uri") and bool(task.get("file_path"))):
                tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"status": "completed", "stopped_at": iso_now(), "updated_at": iso_now()}})
                print(f"[worker {task_id}] ⚠️ Could not open file. Marked completed.")
            return

    # Decide if this source should be treated as a file (finite) or stream (infinite)
    source_is_file = (task.get("source_type") == "file") or (not task.get("source_uri") and bool(task.get("file_path")))

    # Load rules ONCE at startup (do not refetch per frame)
    loaded_rules: List[Dict[str, any]] = task.get("rules") or []

    # Optimize: Determine which models are actually needed based on rules
    # This saves computation by only processing relevant models
    needed_model_capabilities = get_models_needed_by_rules(loaded_rules, model_capabilities)
    
    # Create a set of model IDs that are needed (for fast lookup)
    needed_model_ids = {cap.get("model_id", "") for cap in needed_model_capabilities}
    
    # Create a list of model indices to process
    # We'll only process models whose capabilities are in the needed list
    model_indices_to_process = []
    for idx, cap in enumerate(model_capabilities):
        model_id = cap.get("model_id", "")
        print("model_id: ", model_id)
        # Check if this model's capability is in the needed list by comparing model_id
        if model_id in needed_model_ids:
            model_indices_to_process.append(idx)
    
    # If no models selected, process all (safety fallback)
    if not model_indices_to_process:
        model_indices_to_process = list(range(len(models)))
        print(f"[worker {task_id}] ℹ️ All models will be processed (no filtering)")
    else:
        needed_names = [model_ids[i] for i in model_indices_to_process if i < len(model_ids)]
        print(f"[worker {task_id}] ⚡ Optimized: Processing {len(model_indices_to_process)}/{len(models)} models based on rules: {needed_names}")

    try:
        # per-agent in-memory rule state (indexed by rule index)
        rule_state: Dict[int, Dict[str, any]] = {}

        if run_mode == "continuous":
            # Process frames at FPS continuously
            min_interval = 1.0 / max(1, fps)
            next_tick = time.time()
            frame_index = 0
            # sampling diagnostics
            last_status = time.time()
            processed_in_window = 0
            skipped_in_window = 0
            last_seen_hub_index: Optional[int] = None
            while True:
                # stop/end checks
                task_document = tasks_collection.find_one({"_id": ObjectId(task_id)}, projection={"stop_requested": 1, "end_at": 1})
                # If task document deleted, stop immediately
                if not task_document:
                    print(f"[worker {task_id}] ⏹️ Stopping (task deleted)")
                    return
                stop_requested = bool(task_document.get("stop_requested")) if task_document else False
                end_at_dt = parse_iso(task_document.get("end_at")) if task_document else None
                now = datetime.now(timezone.utc)
                if stop_requested or (end_at_dt and now >= end_at_dt):
                    status = "completed" if end_at_dt and now >= end_at_dt else "cancelled"
                    tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"status": status, "stopped_at": iso_now(), "updated_at": iso_now()}})
                    print(f"[worker {task_id}] ⏹️ Stopping (status={status})")
                    return

                # FPS pacing
                # FPS pacing: control how often we process frames so we match the target FPS.
                # 'next_tick' is the theoretical time (in seconds since epoch) when we should process the next frame.
                now_timestamp = time.time()
                if now_timestamp < next_tick:
                    # If we're ahead of schedule, sleep until the next scheduled frame time.
                    time.sleep(max(0.0, next_tick - now_timestamp))
                # After processing, compute the next frame's scheduled time.
                # This ensures steady pacing and prevents drift: 
                # - next_tick is increased by min_interval (frame period)
                # - But if we're running late (e.g., heavy computation), catch up to the current time
                next_tick = max(next_tick + min_interval, time.time())

                # Acquire next frame
                if use_hub:
                    entry = shared_store.get(camera_id, {}) if shared_store is not None else {}
                    frame = reconstruct_frame(entry)
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    # sampling metrics
                    hub_index = int(entry.get("frame_index", 0)) if isinstance(entry, dict) else 0
                    if last_seen_hub_index is not None and hub_index > last_seen_hub_index + 1:
                        skipped_in_window += (hub_index - last_seen_hub_index - 1)
                    last_seen_hub_index = hub_index
                else:
                    is_frame_read, frame = video_capture.read()  # type: ignore[union-attr]
                    if not is_frame_read:
                        # File: end reached → complete. Stream: brief retry.
                        if source_is_file:
                            tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"status": "completed", "stopped_at": iso_now(), "updated_at": iso_now()}})
                            print(f"[worker {task_id}] ✅ File ended. Marked completed.")
                            return
                        else:
                            time.sleep(0.3)
                            continue

                frame_index += 1
                processed_in_window += 1
                # status once per second
                if (time.time() - last_status) >= 1.0:
                    base_fps = 0.0
                    hub_index_status = last_seen_hub_index or 0
                    if use_hub and isinstance(entry, dict):
                        try:
                            base_fps = float(entry.get("base_fps", 0.0))
                        except Exception:
                            base_fps = 0.0
                    print(f"[worker {task_id}] ⏱️ sampling agent_fps={fps} base_fps={base_fps} processed={processed_in_window}/s skipped={skipped_in_window} hub_frame_index={hub_index_status} camera_id={camera_id}")
                    processed_in_window = 0
                    skipped_in_window = 0
                    last_status = time.time()

                # Run YOLO on this frame for each model, merge detections using unified extraction
                merged_boxes: List[List[float]] = []
                merged_classes: List[str] = []
                merged_scores: List[float] = []
                merged_keypoints: List[List[List[float]]] = []
                merged_masks: List[Optional[np.ndarray]] = []

                # Process only models that are needed by the rules (optimization)
                for model_idx in model_indices_to_process:
                    if model_idx >= len(models):
                        continue
                    
                    model = models[model_idx]
                    try:
                        results = model(frame, verbose=False)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[worker {task_id}] ⚠️ YOLO error: {exc}")
                        continue
                    
                    if results:
                        first_result = results[0]
                        # Get model ID for this model
                        model_id = model_ids[model_idx] if model_idx < len(model_ids) else ""
                        
                        # Use unified extraction (handles all model types)
                        # For weapon models, normalize class names to lowercase
                        normalize = "weapon" in model_id.lower()
                        model_detections = extract_all_detections(first_result, model_id=model_id, normalize_classes=normalize)
                        print("model_detections: ", model_detections)
                        # Merge into combined detections
                        merged_boxes.extend(model_detections.get("boxes", []))
                        merged_classes.extend(model_detections.get("classes", []))
                        merged_scores.extend(model_detections.get("scores", []))
                        merged_masks.extend(model_detections.get("masks", []))
                        
                        # Merge keypoints if available
                        if model_detections.get("has_keypoints", False):
                            merged_keypoints.extend(model_detections.get("keypoints", []))

                # Build detections payload for rule engine
                detections = {
                    "classes": merged_classes,
                    "scores": merged_scores,
                    "boxes": merged_boxes,
                    "keypoints": merged_keypoints,
                    "masks": merged_masks,
                    "ts": datetime.now(timezone.utc),
                }
                
                # Debug: Log keypoint extraction (first frame and every 30 frames)
                if frame_index == 1 or frame_index % 30 == 0:
                    if merged_keypoints:
                        print(f"[worker {task_id}] ✅ Keypoints extracted: {len(merged_keypoints)} person(s) with pose data")
                    else:
                        print(f"[worker {task_id}] ⚠️ No keypoints extracted! Check if model is a pose model (e.g., yolov8n-pose.pt)")
                
                # Draw detections using unified pipeline (replaces old rule-specific methods)
                if shared_store is not None and loaded_rules:
                    try:
                        # Use unified drawing function (handles all rule types automatically)
                        print("detecttions: ", detections)
                        processed_frame = draw_detections_unified(frame.copy(), detections, loaded_rules)
                        # Get agent_id from task (use agent_id if available, else use task_id)
                        agent_id = task.get("agent_id") or task_id
                        
                        # Convert processed frame to bytes
                        frame_bytes = processed_frame.tobytes()
                        height, width = processed_frame.shape[0], processed_frame.shape[1]
                        
                        # Get frame metadata from hub entry if available
                        hub_frame_index = entry.get("frame_index", frame_index) if use_hub and isinstance(entry, dict) else frame_index
                        camera_fps = entry.get("camera_fps") if use_hub and isinstance(entry, dict) else None
                        actual_fps = entry.get("actual_fps") if use_hub and isinstance(entry, dict) else fps
                        
                        # Publish processed frame to agent-specific shared_store key
                        shared_store[agent_id] = {
                            "shape": (height, width, 3),
                            "dtype": "uint8",
                            "frame_index": hub_frame_index,
                            "ts_monotonic": time.time(),
                            "camera_fps": camera_fps,
                            "actual_fps": actual_fps,
                            "bytes": frame_bytes,
                            "agent_id": agent_id,
                            "task_name": agent_name,
                        }
                    except Exception as exc:  # noqa: BLE001
                        # Don't fail the worker if frame processing fails
                        print(f"[worker {task_id}] ⚠️  Error processing frame for stream: {exc}")

                # Prefer new rule engine if rules are provided; else fallback
                event = None
                if loaded_rules:
                    # Debug: log rules and detections (first frame only, then every 30 frames)
                    if frame_index == 1 or frame_index % 30 == 0:
                        print(f"[worker {task_id}] 🔍 Rules: {loaded_rules}")
                        print(f"[worker {task_id}] 🔍 Detected classes: {merged_classes[:5]}")  # First 5 only
                    event = evaluate_rules(loaded_rules, detections, task, rule_state, detections["ts"])
                else:
                    # compatibility path
                    label = check_event_match(task, merged_classes)
                    event = {"label": label} if label else None

                video_ms = get_video_time_ms(video_capture, frame_index, fps) if not use_hub else (frame_index / float(max(1, fps))) * 1000.0
                video_ts = format_video_time_ms(video_ms)

                if event and event.get("label"):
                    event_label = str(event["label"]).strip()
                    # Make weapon detection alerts more prominent
                    if any(keyword in event_label.lower() for keyword in ["gun detected", "knife detected", "weapon detected"]):
                        weapon_label = event_label.upper()
                        print(f"[worker {task_id}] 🚨🚨🚨 {'=' * 20} WEAPON DETECTED {'=' * 20}")
                        print(f"[worker {task_id}] 🚨🚨🚨 {weapon_label} 🚨🚨🚨 | agent='{agent_name}' | video_time={video_ts}")
                        print(f"[worker {task_id}] 🚨🚨🚨 {'=' * 60}")
                    # Make fall detection alerts more prominent
                    elif "fall" in event_label.lower() or "accident" in event_label.lower():
                        print(f"[worker {task_id}] 🚨🚨🚨 {'=' * 20} FALL DETECTED {'=' * 20}")
                        print(f"[worker {task_id}] 🚨🚨🚨 FALL DETECTED! {event_label} 🚨🚨🚨 | agent='{agent_name}' | video_time={video_ts}")
                        print(f"[worker {task_id}] 🚨🚨🚨 {'=' * 60}")
                        if event.get("fallen_count"):
                            print(f"[worker {task_id}]    └─ {event.get('fallen_count')} person(s) detected as fallen")
                    else:
                        print(f"[worker {task_id}] 🔔 {event_label} | agent='{agent_name}' | video_time={video_ts}")
                else:
                    print(f"[worker {task_id}] ℹ️ No rule match | agent='{agent_name}' | video_time={video_ts}")

                # Heartbeat
                tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"updated_at": iso_now()}})
        else:
            # Patrol mode: sleep interval, then process frames for a short window at FPS, repeat
            interval_seconds = max(0, int(interval_minutes) * 60)
            window_seconds = max(1, int(check_duration_seconds))
            print(f"[worker {task_id}] 💤 Patrol mode | sleep={interval_seconds}s window={window_seconds}s fps={fps}")
            while True:
                # Sleep with periodic heartbeat and stop checks
                if sleep_with_heartbeat(tasks_collection, task_id, interval_seconds):
                    return

                # Detection window
                window_end = time.time() + window_seconds
                min_interval = 1.0 / max(1, fps)
                next_tick = time.time()
                print(f"[worker {task_id}] 🔎 Patrol window started ({window_seconds}s)")
                # Reset per-window rule state for patrol windows
                rule_state = {}
                frame_index = 0
                # patrol diagnostics
                last_status = time.time()
                processed_in_window = 0
                skipped_in_window = 0
                last_seen_hub_index = None
                while time.time() < window_end:
                    # stop/end checks inside window
                    task_document = tasks_collection.find_one({"_id": ObjectId(task_id)}, projection={"stop_requested": 1, "end_at": 1})
                    # If task document deleted, stop immediately
                    if not task_document:
                        print(f"[worker {task_id}] ⏹️ Stopping (task deleted)")
                        return
                    stop_requested = bool(task_document.get("stop_requested")) if task_document else False
                    end_at_dt = parse_iso(task_document.get("end_at")) if task_document else None
                    now = datetime.now(timezone.utc)
                    if stop_requested or (end_at_dt and now >= end_at_dt):
                        status = "completed" if end_at_dt and now >= end_at_dt else "cancelled"
                        tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"status": status, "stopped_at": iso_now(), "updated_at": iso_now()}})
                        print(f"[worker {task_id}] ⏹️ Stopping (status={status})")
                        return

                    # FPS pacing
                    now_timestamp = time.time()
                    if now_timestamp < next_tick:
                        time.sleep(max(0.0, next_tick - now_timestamp))
                    next_tick = max(next_tick + min_interval, time.time())

                    # Acquire next frame
                    if use_hub:
                        entry = shared_store.get(camera_id, {}) if shared_store is not None else {}
                        frame = reconstruct_frame(entry)
                        if frame is None:
                            time.sleep(0.05)
                            continue
                        # sampling metrics
                        hub_index = int(entry.get("frame_index", 0)) if isinstance(entry, dict) else 0
                        if last_seen_hub_index is not None and hub_index > last_seen_hub_index + 1:
                            skipped_in_window += (hub_index - last_seen_hub_index - 1)
                        last_seen_hub_index = hub_index
                    else:
                        is_frame_read, frame = video_capture.read()  # type: ignore[union-attr]
                        if not is_frame_read:
                            if source_is_file:
                                tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"status": "completed", "stopped_at": iso_now(), "updated_at": iso_now()}})
                                print(f"[worker {task_id}] ✅ File ended. Marked completed.")
                                return
                            else:
                                time.sleep(0.3)
                                continue

                    frame_index += 1
                    processed_in_window += 1
                    if (time.time() - last_status) >= 1.0:
                        base_fps = 0.0
                        hub_index_status = last_seen_hub_index or 0
                        if use_hub and isinstance(entry, dict):
                            try:
                                base_fps = float(entry.get("base_fps", 0.0))
                            except Exception:
                                base_fps = 0.0
                        print(f"[worker {task_id}] ⏱️ sampling(agent patrol) agent_fps={fps} base_fps={base_fps} processed={processed_in_window}/s skipped={skipped_in_window} hub_frame_index={hub_index_status} camera_id={camera_id}")
                        processed_in_window = 0
                        skipped_in_window = 0
                        last_status = time.time()
                    merged_boxes: List[List[float]] = []
                    merged_classes: List[str] = []
                    merged_scores: List[float] = []
                    merged_keypoints: List[List[List[float]]] = []
                    merged_masks: List[Optional[np.ndarray]] = []
                    # Process only models that are needed by the rules (patrol mode - optimization)
                    for model_idx in model_indices_to_process:
                        if model_idx >= len(models):
                            continue
                        
                        model = models[model_idx]
                        try:
                            results = model(frame, verbose=False)
                        except Exception as exc:  # noqa: BLE001
                            print(f"[worker {task_id}] ⚠️ YOLO error: {exc}")
                            continue
                        
                        if results:
                            first_result = results[0]
                            # Get model ID for this model
                            model_id = model_ids[model_idx] if model_idx < len(model_ids) else ""
                            
                            # Use unified extraction (handles all model types)
                            normalize = "weapon" in model_id.lower()
                            model_detections = extract_all_detections(first_result, model_id=model_id, normalize_classes=normalize)
                            print("model_detections: ", model_detections)
                            # Merge into combined detections
                            merged_boxes.extend(model_detections.get("boxes", []))
                            merged_classes.extend(model_detections.get("classes", []))
                            merged_scores.extend(model_detections.get("scores", []))
                            merged_masks.extend(model_detections.get("masks", []))
                            
                            # Merge keypoints if available
                            if model_detections.get("has_keypoints", False):
                                merged_keypoints.extend(model_detections.get("keypoints", []))

                    detections = {
                        "classes": merged_classes,
                        "scores": merged_scores,
                        "boxes": merged_boxes,
                        "keypoints": merged_keypoints,
                        "masks": merged_masks,
                        "ts": datetime.now(timezone.utc),
                    }
                    
                    # Draw detections using unified pipeline (patrol mode)
                    if shared_store is not None and loaded_rules:
                        try:
                            # Use unified drawing function (handles all rule types automatically)
                            processed_frame = draw_detections_unified(frame.copy(), detections, loaded_rules)
                            agent_id = task.get("agent_id") or task_id
                            frame_bytes = processed_frame.tobytes()
                            height, width = processed_frame.shape[0], processed_frame.shape[1]
                            hub_frame_index = entry.get("frame_index", frame_index) if use_hub and isinstance(entry, dict) else frame_index
                            camera_fps = entry.get("camera_fps") if use_hub and isinstance(entry, dict) else None
                            actual_fps = entry.get("actual_fps") if use_hub and isinstance(entry, dict) else fps
                            
                            shared_store[agent_id] = {
                                "shape": (height, width, 3),
                                "dtype": "uint8",
                                "frame_index": hub_frame_index,
                                "ts_monotonic": time.time(),
                                "camera_fps": camera_fps,
                                "actual_fps": actual_fps,
                                "bytes": frame_bytes,
                                "agent_id": agent_id,
                                "task_name": agent_name,
                            }
                        except Exception as exc:  # noqa: BLE001
                            print(f"[worker {task_id}] ⚠️  Error processing frame for stream (patrol): {exc}")

                    event = None
                    if loaded_rules:
                        print(f"[worker {task_id}] ⏱️ Evaluating rules: {loaded_rules}")
                        for rule in loaded_rules:
                            print(f"[worker {task_id}] ⏱️ Rule: {rule}")
                        event = evaluate_rules(loaded_rules, detections, task, rule_state, detections["ts"])
                        print(f"[worker {task_id}] ⏱️ Event: {event}")
                    else:
                        label = check_event_match(task, merged_classes)
                        event = {"label": label} if label else None

                    video_ms = get_video_time_ms(video_capture, frame_index, fps) if not use_hub else (frame_index / float(max(1, fps))) * 1000.0
                    video_ts = format_video_time_ms(video_ms)

                    if event and event.get("label"):
                        event_label = str(event["label"]).strip()
                        # Make weapon detection alerts more prominent
                        if any(keyword in event_label.lower() for keyword in ["gun detected", "knife detected", "weapon detected"]):
                            weapon_label = event_label.upper()
                            print(f"[worker {task_id}] 🚨🚨🚨 {'=' * 20} WEAPON DETECTED {'=' * 20}")
                            print(f"[worker {task_id}] 🚨🚨🚨 {weapon_label} 🚨🚨🚨 | agent='{agent_name}' | video_time={video_ts}")
                            print(f"[worker {task_id}] 🚨🚨🚨 {'=' * 60}")
                        else:
                            print(f"[worker {task_id}] 🔔 {event_label} | agent='{agent_name}' | video_time={video_ts}")
                    else:
                        print(f"[worker {task_id}] ℹ️ No rule match | agent='{agent_name}' | video_time={video_ts}")

                    # Heartbeat during window
                    tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"updated_at": iso_now()}})
                print(f"[worker {task_id}] 💤 Patrol window ended; going back to sleep")

    except KeyboardInterrupt:
        pass
    finally:
        try:
            if not use_hub and video_capture is not None:
                video_capture.release()
        except Exception:
            pass
        print(f"[worker {task_id}] ⏹️ Exiting")


