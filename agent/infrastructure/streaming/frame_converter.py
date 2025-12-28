"""
Frame Converter
===============

Converts frames from shared_store (bytes) to VideoFrame for WebRTC.
"""
import numpy as np
from typing import Dict, Any, Optional
from av import VideoFrame
from fractions import Fraction


class FrameConverter:
    """
    Converts frames from shared_store format to VideoFrame.
    
    shared_store format:
    {
        "shape": (height, width, 3),
        "dtype": "uint8",
        "frame_index": int,
        "bytes": bytes,
        "ts_monotonic": float,
        ...
    }
    """
    
    @staticmethod
    def bytes_to_videoframe(entry: Dict[str, Any]) -> Optional[VideoFrame]:
        """
        Convert shared_store entry to VideoFrame.
        
        Args:
            entry: Frame data from shared_store
            
        Returns:
            VideoFrame if conversion successful, None otherwise
        """
        try:
            if not entry or "bytes" not in entry:
                return None
            
            # Reconstruct numpy array from bytes
            buffer_bytes = entry["bytes"]
            shape = tuple(entry["shape"])
            dtype = np.dtype(entry["dtype"])
            flat_array = np.frombuffer(buffer_bytes, dtype=dtype)
            frame_bgr = flat_array.reshape(shape)
            
            # Convert to VideoFrame
            video_frame = VideoFrame.from_ndarray(frame_bgr, format="bgr24")
            
            # Calculate proper PTS based on frame timing
            frame_index = entry.get("frame_index", 0)
            camera_fps = entry.get("camera_fps", 25.0)  # Default to 25fps if not available
            
            # PTS should be in time_base units
            # For 25fps: each frame = 1/25 seconds = 40ms = 40 time_base units (if time_base = 1/1000)
            # Use frame_index * (1000 / fps) for proper timing
            if camera_fps and camera_fps > 0:
                pts_value = int(frame_index * (1000.0 / camera_fps))
            else:
                # Fallback: assume 25fps
                pts_value = frame_index * 40
            
            video_frame.pts = pts_value
            video_frame.time_base = Fraction(1, 1000)  # milliseconds
            
            return video_frame
        except Exception as e:
            print(f"[converter] ⚠️  Error converting frame: {e}")
            return None

