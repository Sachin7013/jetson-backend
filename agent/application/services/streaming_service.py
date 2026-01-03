"""
Streaming Service
=================

Application service for WebRTC streaming coordination.
Manages both camera streams (raw video) and agent streams (processed video).
"""
import asyncio
import os
from typing import Dict, Any, Optional
from agent.infrastructure.streaming.aws_signaling_client import AWSSignalingClient
from agent.infrastructure.streaming.agent_aws_signaling_client import AgentAWSSignalingClient


class StreamingService:
    """
    Application service for WebRTC streaming.
    
    Manages connections to AWS signaling server for each active camera.
    """
    
    def __init__(self, shared_store: Dict[str, Any]):
        """
        Initialize streaming service.
        
        Args:
            shared_store: Shared memory dict from multiprocessing.Manager
        """
        self.shared_store = shared_store
        # Camera streams (raw video)
        self._clients: Dict[str, AWSSignalingClient] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        # Agent streams (processed video with bounding boxes)
        self._agent_clients: Dict[str, AgentAWSSignalingClient] = {}
        self._agent_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
    
    async def start_camera_stream(self, user_id: str, camera_id: str) -> None:
        """
        Start streaming for a specific camera.
        
        Args:
            user_id: User ID
            camera_id: Camera ID
        """
        key = f"{user_id}:{camera_id}"
        
        if key in self._clients:
            print(f"[streaming] ⚠️  Stream already running for {camera_id}")
            return
        
        client = AWSSignalingClient(self.shared_store, user_id, camera_id)
        self._clients[key] = client
        
        # Start streaming in background task
        task = asyncio.create_task(client.connect_and_stream())
        self._tasks[key] = task
        
        print(f"[streaming] 🎥 Started AWS stream for camera: {camera_id} (user: {user_id})")
    
    async def stop_camera_stream(self, user_id: str, camera_id: str) -> None:
        """Stop streaming for a specific camera."""
        key = f"{user_id}:{camera_id}"
        
        if key in self._clients:
            await self._clients[key].stop()
            if key in self._tasks:
                self._tasks[key].cancel()
                try:
                    await self._tasks[key]
                except asyncio.CancelledError:
                    pass
            del self._clients[key]
            del self._tasks[key]
            print(f"[streaming] 🛑 Stopped AWS stream for camera: {camera_id}")
    
    async def start(self) -> None:
        """
        Start the streaming service.
        
        This service monitors active cameras and connects to AWS for each.
        """
        if self._running:
            print("[streaming] ⚠️  Streaming service already running")
            return
        
        self._running = True
        print("[streaming] 🚀 Started Streaming Service (AWS mode)")
        
        # Monitor cameras and start/stop streams
        # This will be called periodically from runner or main
        while self._running:
            await asyncio.sleep(5)  # Check every 5 seconds
    
    async def stop(self) -> None:
        """Stop all camera and agent streams."""
        self._running = False
        
        # Stop all camera clients
        for key in list(self._clients.keys()):
            user_id, camera_id = key.split(":", 1)
            await self.stop_camera_stream(user_id, camera_id)
        
        # Stop all agent clients
        for key in list(self._agent_clients.keys()):
            user_id, camera_id, agent_id = key.split(":", 2)
            await self.stop_agent_stream(user_id, camera_id, agent_id)
        
        print("[streaming] 🛑 Streaming service stopped")
    
    def is_running(self) -> bool:
        """Check if streaming service is running."""
        return self._running
    
    def get_active_connections(self) -> int:
        """Get number of active camera streams."""
        return len(self._clients)
    
    async def start_agent_stream(self, user_id: str, camera_id: str, agent_id: str) -> None:
        """
        Start streaming for a specific agent.
        
        Args:
            user_id: User ID
            agent_id: Agent ID
        """
        key = f"{user_id}:{camera_id}:{agent_id}"
        
        if key in self._agent_clients:
            print(f"[streaming] ⚠️  Agent stream already running for {agent_id}")
            return
        
        client = AgentAWSSignalingClient(self.shared_store, user_id, agent_id, camera_id)
        self._agent_clients[key] = client
        
        # Start streaming in background task
        task = asyncio.create_task(client.connect_and_stream())
        self._agent_tasks[key] = task
        
        print(f"[streaming] 🎯 Started AWS stream for agent: {agent_id} (user: {user_id})")
    
    async def stop_agent_stream(self, user_id: str, camera_id: str, agent_id: str) -> None:
        """Stop streaming for a specific agent."""
        key = f"{user_id}:{camera_id}:{agent_id}"
        
        if key in self._agent_clients:
            await self._agent_clients[key].stop()
            if key in self._agent_tasks:
                self._agent_tasks[key].cancel()
                try:
                    await self._agent_tasks[key]
                except asyncio.CancelledError:
                    pass
            del self._agent_clients[key]
            del self._agent_tasks[key]
            print(f"[streaming] 🛑 Stopped AWS stream for agent: {agent_id}")
    
    def get_active_agent_connections(self) -> int:
        """Get number of active agent streams."""
        return len(self._agent_clients)