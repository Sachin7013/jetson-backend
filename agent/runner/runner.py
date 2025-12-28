"""
Runner (Simple Orchestrator)
----------------------------

Role:
- Poll MongoDB for cameras and agent tasks
- Start CameraPublisher for each registered camera (runs at native FPS)
- Launch one worker process per agent task
- Clean up finished workers and stopped cameras

Data flow:
- Cameras Collection → CameraPublisher (one per camera)
- Tasks Collection → Worker (one per task)
- Both read from shared_store[camera_id] independently
"""
import os
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Process, Manager, Queue
from typing import Dict, Any, List, Tuple, Optional

# Allow running this file directly by ensuring project root is on sys.path
if __package__ is None or __package__ == "":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

from agent.utils.db import get_collection
from agent.utils.utils import iso_now, parse_iso
from agent.worker.agent_main import run_task_worker
from agent.worker.frame_hub import CameraPublisher, CameraCommand


def main(shared_store=None) -> None:
    """
    Monitor MongoDB for cameras and tasks, start publishers and workers.
    
    FLOW:
    -----
    1. Poll cameras collection - start CameraPublisher for each active camera
    2. Poll tasks collection - start Worker for each active task
    3. Workers read from shared_store (populated by CameraPublisher)
    4. Clean up stopped cameras and finished tasks
    
    Args:
        shared_store: Optional[shared store from multiprocessing.Manager().dict()]
                     If None, creates a new one.
    """
    print("[runner] 🚀 Starting Agent Runner")
    tasks_collection = get_collection()  # Default: 'Agents' collection
    cameras_collection = get_collection("cameras")  # 'cameras' collection
    poll_interval = int(os.getenv("AGENT_POLL_INTERVAL_SEC", "5"))

    # Use provided shared_store or create new one
    if shared_store is None:
        manager = Manager()
        shared_store = manager.dict()
        print("[runner] 📦 Created new shared_store")

    task_processes: Dict[str, Process] = {}
    # camera_id -> (process, command_queue, source_uri)

    #command queue = currently unused 
    camera_publishers: Dict[str, Tuple[Process, Queue, str]] = {}

    try:
        while True:
            now = datetime.now(timezone.utc)

            # ============================================================
            # STEP 1: Handle Cameras - Start CameraPublisher for each
            # ============================================================
            active_cameras = cameras_collection.find({"status": {"$ne": "inactive"}})
            active_camera_ids = set()
            
            for camera_doc in active_cameras:
                camera_id = camera_doc.get("camera_id")
                source_uri = camera_doc.get("rtsp_url") or camera_doc.get("source_uri")
                
                if not camera_id or not source_uri:
                    #skip if camera_id or source_uri is not present
                    continue
                    
                active_camera_ids.add(camera_id)
                
                # Check if CameraPublisher is already running
                if camera_id in camera_publishers:
                    publisher_process, command_queue, _source_uri = camera_publishers[camera_id]
                    # Check if process is still alive
                    if publisher_process.is_alive():
                        # Already running, skip
                        continue
                    else:
                        # Process died, remove from dict and restart
                        print(f"[runner] ⚠️  CameraPublisher for {camera_id} died, restarting...")
                        del camera_publishers[camera_id]
                
                # Start CameraPublisher (either new or restarted)
                command_queue: Queue = Queue()
                publisher_process = CameraPublisher(
                    camera_id=camera_id,
                    source_uri=source_uri,
                    shared_store=shared_store,
                    command_queue=command_queue,
                )
                publisher_process.start()
                camera_publishers[camera_id] = (publisher_process, command_queue, source_uri)
                print(f"[runner] 🎥 Started publisher for camera {camera_id} (native FPS)")

            # ============================================================
            # STEP 2: Stop publishers for cameras no longer active
            # ============================================================
            for camera_id in list(camera_publishers.keys()):
                if camera_id not in active_camera_ids:
                    publisher_process, command_queue, _source_uri = camera_publishers[camera_id]
                    try:
                        command_queue.put(CameraCommand(kind="stop"))
                    except Exception:
                        pass
                    if publisher_process.is_alive():
                        publisher_process.join(timeout=1.0)
                    del camera_publishers[camera_id]
                    print(f"[runner] 🎥 Stopped publisher for camera {camera_id}")

            # ============================================================
            # STEP 3: Handle Tasks - Start Worker for each active task
            # ============================================================
            active_task_cursor = tasks_collection.find({"status": {"$in": ["pending", "scheduled", "running", None]}}).sort("created_at", 1)
            tasks_list = list(active_task_cursor)
            
            # Terminate workers for tasks that are no longer active
            active_task_ids = {str(task["_id"]) for task in tasks_list}
            for task_id_for_cleanup, worker_process in list(task_processes.items()):
                if task_id_for_cleanup not in active_task_ids:
                    if worker_process.is_alive():
                        worker_process.terminate()
                        worker_process.join(timeout=0.5)
                    del task_processes[task_id_for_cleanup]
                    print(f"[runner] 🛑 Terminated worker for inactive/missing task {task_id_for_cleanup}")

            # ============================================================
            # STEP 4: Launch new workers for tasks
            # ============================================================
            for task in tasks_list:
                task_id = str(task["_id"])

                # Skip if already running
                if task_id in task_processes and task_processes[task_id].is_alive():
                    continue

                start_at = parse_iso(task.get("start_at"))
                end_at = parse_iso(task.get("end_at"))

                # Handle "scheduled" tasks (future start time)
                if start_at and now < start_at:
                    if task.get("status") != "scheduled":
                        tasks_collection.update_one(
                            {"_id": task["_id"]},
                            {"$set": {"status": "scheduled", "updated_at": iso_now()}},
                        )
                    continue

                # Handle "expired" tasks (past end time)
                if end_at and now >= end_at:
                    if task.get("status") not in {"completed", "cancelled"}:
                        tasks_collection.update_one(
                            {"_id": task["_id"]},
                            {"$set": {
                                "status": "completed",
                                "stopped_at": iso_now(),
                                "updated_at": iso_now(),
                            }},
                        )
                    continue

                # Launch new worker
                tasks_collection.update_one(
                    {"_id": task["_id"]},
                    {"$set": {
                        "status": "running",
                        "started_at": iso_now(),
                        "updated_at": iso_now(),
                    }},
                )

                # Pass shared_store proxy to worker so it can subscribe to frames
                worker_process = Process(target=run_task_worker, args=(task_id, shared_store))
                worker_process.daemon = True
                worker_process.start()
                task_processes[task_id] = worker_process

                print(f"[runner] 🏃 Launched worker for task {task_id} (pid={worker_process.pid})")

            # Clean up finished processes
            for finished_task_id, worker_process in list(task_processes.items()):
                if not worker_process.is_alive():
                    worker_process.join(timeout=0.1)
                    del task_processes[finished_task_id]
                    print(f"[runner] 🧹 Cleaned up worker for task {finished_task_id}")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n[runner] 🛑 Stopping Agent Runner (Ctrl+C)")
        # Stop publishers
        for camera_id, (publisher_process, command_queue, _source_uri) in list(camera_publishers.items()):
            try:
                command_queue.put(CameraCommand(kind="stop"))
            except Exception:
                pass
            if publisher_process.is_alive():
                publisher_process.terminate()
        camera_publishers.clear()
        for worker_process in task_processes.values():
            if worker_process.is_alive():
                worker_process.terminate()
        print("[runner] ✅ All workers terminated.")


if __name__ == "__main__":
    main()


