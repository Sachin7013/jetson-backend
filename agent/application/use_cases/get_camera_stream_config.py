"""
Get Camera Stream Config Use Case
==================================

Use case for retrieving WebRTC configuration for a specific camera stream.
"""
import os
from typing import Dict, Any
from agent.domain.repositories.camera_repository import CameraRepository


class GetCameraStreamConfigUseCase:
    """Use case for getting camera stream configuration."""
    
    def __init__(self, camera_repository: CameraRepository):
        self._repository = camera_repository
    
    def execute(self, camera_id: str, user_id: str) -> Dict[str, Any]:
        """
        Get WebRTC configuration for camera stream.
        
        Args:
            camera_id: Camera identifier
            user_id: User ID (for validation)
        
        Returns:
            Dict with signaling_url and ice_servers
        
        Raises:
            ValueError: If camera not found or doesn't belong to user
        """
        camera = self._repository.find_by_id(camera_id)
        if not camera:
            raise ValueError(f"Camera '{camera_id}' not found")
        
        if camera.user_id != user_id:
            raise ValueError(f"Camera '{camera_id}' does not belong to user '{user_id}'")
        
        # Get signaling URL from environment
        aws_signaling_url = os.getenv("AWS_SIGNALING_URL")
        if aws_signaling_url:
            # Format: ws://aws-url:8000/ws/viewer:{user_id}
            # Note: Currently all cameras for a user share the same signaling URL
            # The signaling server routes based on user_id
            signaling_url = f"{aws_signaling_url.rstrip('/')}/ws/viewer:{user_id}"
        else:
            signaling_ws = os.getenv("SIGNALING_WS", "ws://localhost:8765")
            signaling_url = f"{signaling_ws.rstrip('/')}/viewer:{user_id}"
        
        # Build ICE servers (same as generic stream config)
        ice_servers = [
            {"urls": "stun:stun.l.google.com:19302"}
        ]
        
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
            # Fallback to hardcoded TURN
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
            "camera_id": camera_id,
            "camera_name": camera.camera_name,
        }

