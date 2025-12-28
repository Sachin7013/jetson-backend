"""
Test seeding script
-------------------

Purpose:
- Insert a few sample Agent tasks into MongoDB to exercise the Redis-based pipeline.
- Hardcodes camera_id, file_path, model_ids, and fps values for quick testing.

What it creates:
- One agent on the same camera and model ("yolov8n.pt") with a rule:
-  - Alert when there are at least 2 persons detected (count_at_least).

How to use:
1) Ensure env vars: MONGO_URI (required), DB_NAME, COLLECTION_NAME (optional)
2) Place a local video file and set FILE_PATH below, or update SOURCE_URI to a stream URL.
3) Run:
   python -m app.agent.test.test_seed
4) Start the runner:
   python -m app.agent.runner.runner
5) Watch logs for alerts (e.g., \"Multiple persons detected\")
"""
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

from app.agent.utils.db import get_collection
from app.agent.utils.utils import iso_now


def _iso_in(seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    col = get_collection()

    # Hardcoded test config
    camera_id = "cam01"
    # Either set a local file path or an RTSP/HTTP stream URL.
    # For OpenCV, both file path and URL can be passed to VideoCapture.
    FILE_PATH = os.getenv("TEST_VIDEO_PATH", "")  # update this to an existing file
    SOURCE_URI = os.getenv("TEST_STREAM_URI", "rtsp://algoorange:algoorange2025@192.168.1.6:554/stream1")  # if you prefer a stream, set this and leave FILE_PATH empty

    # Choose a YOLO model; 'yolov8n.pt' is lightweight and suitable for testing
    model_id = os.getenv("TEST_MODEL_ID", "yolov8n.pt")

    # Task timing
    start_at = iso_now()  # start immediately
    end_at = _iso_in(600)  # run for 10 minutes (adjust as needed)

    # Single agent with a rule: alert when there are at least 2 persons
    agents: List[Dict[str, Any]] = [
        {
            "task_name": "laptop Count - Rules Test",
            "task_type": "object_detection",
            "camera_id": camera_id,
            # Prefer SOURCE_URI if set; otherwise use FILE_PATH
            "source_uri": SOURCE_URI if SOURCE_URI else None,
            "file_path": None if SOURCE_URI else FILE_PATH,
            "model_ids": [model_id],
            "fps": 50,
            "run_mode": "continuous",
            # "run_mode": "patrol",
            # "interval_minutes": 2,
            # "check_duration_seconds": 10,
           "rules": [
                {
                    "type": "class_count",
                    "class": "laptop",
                    "label": "laptop count",
                }
            ],
            # "rules": [
            #         {
            #             "type": "count_at_least",
            #             "class": "person",
            #             "min_count": 2,
            #             "duration_seconds": 0,
            #             "label": "Multiple persons detected",
            #         }
            #     ],
            "status": "pending",
            "start_at": start_at,
            "end_at": None,
            "created_at": iso_now(),
        }
    ]

    # Clean up None keys for Mongo
    to_insert = []
    for a in agents:
        a = {k: v for k, v in a.items() if v is not None}
        to_insert.append(a)

    result = col.insert_many(to_insert)
    print("✅ Inserted test Agents with IDs:")
    for _id in result.inserted_ids:
        print(f" - {str(_id)}")
    print("\nNext steps:")
    print("1) Run the runner: python -m app.agent.runner.runner")
    print("2) Watch logs for alerts like: Multiple persons detected")


if __name__ == "__main__":
    main()


