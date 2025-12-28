"""
Utility helpers
---------------

Role:
- Time helpers for consistent ISO formatting and robust ISO parsing.

Functions:
- iso_now(): 'YYYY-MM-DDTHH:MM:SSZ' (UTC, no microseconds).
- parse_iso(s): safely parse ISO-8601 string to datetime (returns None on error).

Used by:
- supervisor.py, detector.py, subscriber.py, runner.py for timestamps and heartbeats.
"""
from datetime import datetime, timezone
from typing import Optional


def iso_now() -> str:
    """Return the current UTC time in ISO 8601 format (without microseconds)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso(dt: Optional[str]) -> Optional[datetime]:
    """Convert an ISO 8601 string (e.g., 2025-11-10T08:00:00Z) into a datetime object."""
    if not dt:
        return None
    try:
        return datetime.fromisoformat(dt.replace("Z", "+00:00"))
    except Exception:
        return None


