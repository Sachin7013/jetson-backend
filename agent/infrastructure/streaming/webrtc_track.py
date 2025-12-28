"""
WebRTC Video Track
==================

VideoStreamTrack that reads from shared_store and streams via WebRTC.
"""
import asyncio
from typing import Optional, Dict, Any
from av import VideoFrame
from aiortc import VideoStreamTrack

from agent.infrastructure.streaming.frame_converter import FrameConverter


class SharedStoreVideoTrack(VideoStreamTrack):
    """
    WebRTC video track that reads frames from shared_store.
    
    This track continuously reads the latest frame from shared_store
    and streams it via WebRTC to connected clients.
    """
    
    def __init__(self, camera_id: str, shared_store: Dict[str, Any]):
        """
        Initialize video track.
        
        Args:
            camera_id: Camera identifier
            shared_store: Shared memory dict from multiprocessing.Manager
        """
        super().__init__()
        self.camera_id = camera_id
        self.shared_store = shared_store
        self._converter = FrameConverter()
        self._last_frame_index = -1
        self._no_frame_count = 0
        self._max_no_frame_wait = 100  # Max iterations to wait for frame
    
    @property
    def id(self) -> str:
        """Return unique track ID (camera_id)."""
        return self.camera_id
    
    @property
    def kind(self) -> str:
        """Return track kind (always 'video')."""
        return "video"
    
    async def recv(self) -> VideoFrame:
        """
        Get next frame from shared_store and return as VideoFrame.
        
        Returns:
            VideoFrame for WebRTC streaming
        """
        frames_sent = 0
        last_log_time = asyncio.get_event_loop().time()
        
        while True:
            entry = self.shared_store.get(self.camera_id)
            
            if entry is None:
                self._no_frame_count += 1
                if self._no_frame_count > self._max_no_frame_wait:
                    # Log if no frames for too long
                    if self._no_frame_count % 100 == 0:
                        print(f"[track {self.camera_id}] ⚠️  No frames in shared_store (waiting...)")
                    await asyncio.sleep(0.1)
                else:
                    await asyncio.sleep(0.05)
                continue
            
            # Check for errors
            if "error" in entry:
                print(f"[track {self.camera_id}] ❌ Error in shared_store: {entry.get('error')}")
                await asyncio.sleep(0.5)
                continue
            
            frame_index = entry.get("frame_index", -1)
            
            # Skip duplicate frames
            if frame_index == self._last_frame_index:
                await asyncio.sleep(0.01)
                continue
            
            self._last_frame_index = frame_index
            self._no_frame_count = 0
            
            # Convert to VideoFrame
            video_frame = self._converter.bytes_to_videoframe(entry)
            if video_frame is not None:
                frames_sent += 1
                # Log every 5 seconds
                current_time = asyncio.get_event_loop().time()
                if current_time - last_log_time >= 5.0:
                    print(f"[track {self.camera_id}] 📹 Sent {frames_sent} frames to WebRTC")
                    last_log_time = current_time
                return video_frame
            
            print(f"[track {self.camera_id}] ⚠️  Failed to convert frame {frame_index}")
            await asyncio.sleep(0.05)

