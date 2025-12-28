"""
Get Stream Config Use Case
===========================

Business use case for getting WebRTC configuration.
"""
import os
from typing import List, Dict, Any

from agent.domain.repositories.camera_repository import CameraRepository


class GetStreamConfigUseCase:
    """
    Use case for getting WebRTC stream configuration.
    
    This encapsulates the business logic for WebRTC configuration.
    """
    
    def __init__(self, camera_repository: CameraRepository):
        """
        Initialize use case with repository.
        
        Args:
            camera_repository: Repository for camera persistence
        """
        self._repository = camera_repository
    
    def execute(self, user_id: str) -> Dict[str, Any]:
        """
        Execute the get stream config use case.
        
        Args:
            user_id: User ID requesting the configuration
            
        Returns:
            Dictionary with signaling_url and ice_servers
            
        Raises:
            ValueError: If user has no active cameras
        """
        if not user_id or not user_id.strip():
            raise ValueError("User ID is required")
        
        # Validate user has at least one active camera
        user_cameras = self._repository.find_by_user_id(user_id.strip())
        active_cameras = [cam for cam in user_cameras if cam.is_active()]
        
        if not active_cameras:
            raise ValueError(f"No active cameras found for user '{user_id}'")
        
        # Get signaling URL from environment (AWS signaling server)
        # For AWS, use AWS_SIGNALING_URL, fallback to SIGNALING_WS for local
        aws_signaling_url = os.getenv("AWS_SIGNALING_URL")
        if aws_signaling_url:
            # AWS format: ws://aws-url:8000/ws/viewer:{user_id}
            signaling_url = f"{aws_signaling_url.rstrip('/')}/ws/viewer:{user_id.strip()}"
        else:
            # Local format: ws://localhost:8765/viewer:{user_id}/{camera_id}
            signaling_ws = os.getenv("SIGNALING_WS", "ws://localhost:8765")
            signaling_url = f"{signaling_ws.rstrip('/')}/viewer:{user_id.strip()}"
        
        # Build ICE servers
        # Order matters: STUN first (tried first), TURN second (fallback)
        ice_servers = [
            {"urls": "stun:stun.l.google.com:19302"}
        ]
        
        # Add TURN server if configured
        aws_turn_ip = os.getenv("AWS_TURN_IP")
        aws_turn_port = os.getenv("AWS_TURN_PORT")
        aws_turn_user = os.getenv("AWS_TURN_USER")
        aws_turn_pass = os.getenv("AWS_TURN_PASS")
        
        if aws_turn_ip and aws_turn_port and aws_turn_user and aws_turn_pass:
            ice_servers.append({
                "urls": [
                    f"turn:{aws_turn_ip}:{aws_turn_port}?transport=udp",
                    f"turn:{aws_turn_ip}:{aws_turn_port}?transport=tcp",
                ],
                "username": aws_turn_user,
                "credential": aws_turn_pass,
            })
        else:
            # Use hardcoded TURN as fallback if env vars not set
            ice_servers.append({
                "urls": [
                    "turn:13.49.159.215:3478?transport=udp",
                    "turn:13.49.159.215:3478?transport=tcp",
                ],
                "username": "Algo_webrtc",
                "credential": "AlgoOrange2025",
            })
        
        return {
            "signaling_url": signaling_url,
            "ice_servers": ice_servers,
        }

