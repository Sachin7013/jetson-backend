"""
Agent WebRTC Video Track
=========================

VideoStreamTrack that reads processed frames from shared_store for agent streams.
"""
import asyncio
from typing import Optional, Dict, Any
from av import VideoFrame
from aiortc import VideoStreamTrack

from agent.infrastructure.streaming.frame_converter import FrameConverter


class AgentSharedStoreVideoTrack(VideoStreamTrack):
    """
    WebRTC video track that reads processed frames from shared_store for agents.
    
    This track reads frames that have been processed by the agent worker
    (with YOLO bounding boxes drawn) and streams them via WebRTC.
    """
    
    def __init__(self, agent_id: str, shared_store: Dict[str, Any]):
        """
        Initialize agent video track.
        
        Args:
            agent_id: Agent identifier
            shared_store: Shared memory dict from multiprocessing.Manager
        """
        super().__init__()
        self.agent_id = agent_id
        self.shared_store = shared_store
        self._converter = FrameConverter()
        self._last_frame_index = -1
        self._no_frame_count = 0
        self._max_no_frame_wait = 100
    
    @property
    def id(self) -> str:
        """Return unique track ID (agent_id)."""
        return self.agent_id
    
    @property
    def kind(self) -> str:
        """Return track kind (always 'video')."""
        return "video"
    
    async def recv(self) -> VideoFrame:
        """
        Get next processed frame from shared_store and return as VideoFrame.
        
        Returns:
            VideoFrame for WebRTC streaming
        """
        frames_sent = 0
        last_log_time = asyncio.get_event_loop().time()
        
        while True:
            entry = self.shared_store.get(self.agent_id)
            
            if entry is None:
                self._no_frame_count += 1
                if self._no_frame_count > self._max_no_frame_wait:
                    if self._no_frame_count % 100 == 0:
                        print(f"[agent-track {self.agent_id}] ⚠️  No frames in shared_store (waiting...)")
                    await asyncio.sleep(0.1)
                else:
                    await asyncio.sleep(0.05)
                continue
            
            if "error" in entry:
                print(f"[agent-track {self.agent_id}] ❌ Error in shared_store: {entry.get('error')}")
                await asyncio.sleep(0.5)
                continue
            
            frame_index = entry.get("frame_index", -1)
            
            if frame_index == self._last_frame_index:
                await asyncio.sleep(0.01)
                continue
            
            self._last_frame_index = frame_index
            self._no_frame_count = 0
            
            video_frame = self._converter.bytes_to_videoframe(entry)
            if video_frame is not None:
                frames_sent += 1
                current_time = asyncio.get_event_loop().time()
                if current_time - last_log_time >= 5.0:
                    print(f"[agent-track {self.agent_id}] 📹 Sent {frames_sent} processed frames to WebRTC")
                    last_log_time = current_time
                return video_frame
            
            print(f"[agent-track {self.agent_id}] ⚠️  Failed to convert frame {frame_index}")
            await asyncio.sleep(0.05)