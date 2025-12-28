"""
Get Agent Stream Config Use Case
=================================

Use case for retrieving WebRTC configuration for agent-specific streams.
"""
import os
from typing import Dict, Any
from agent.domain.repositories.agent_repository import AgentRepository
from agent.domain.repositories.camera_repository import CameraRepository


class GetAgentStreamConfigUseCase:
    """Use case for getting agent stream configuration."""
    
    def __init__(self, agent_repository: AgentRepository, camera_repository: CameraRepository):
        self._agent_repository = agent_repository
        self._camera_repository = camera_repository
    
    def execute(self, agent_id: str) -> Dict[str, Any]:
        """
        Get WebRTC configuration for agent stream.
        
        Args:
            agent_id: Agent identifier
        
        Returns:
            Dict with signaling_url and ice_servers
        """
        agent = self._agent_repository.find_by_id(agent_id)
        if not agent:
            raise ValueError(f"Agent '{agent_id}' not found")
        
        # Get camera to get user_id
        camera = self._camera_repository.find_by_id(agent.camera_id)
        if not camera:
            raise ValueError(f"Camera '{agent.camera_id}' not found for agent '{agent_id}'")
        
        user_id = camera.user_id
        
        # Get signaling URL from environment
        aws_signaling_url = os.getenv("AWS_SIGNALING_URL")
        if aws_signaling_url:
            # Format: ws://aws-url:8000/ws/viewer:{user_id}:{agent_id}
            # Viewer connects as viewer, agent connects as agent
            signaling_url = f"{aws_signaling_url.rstrip('/')}/ws/viewer:{user_id}:{agent_id}"
        else:
            signaling_ws = os.getenv("SIGNALING_WS", "ws://localhost:8765")
            signaling_url = f"{signaling_ws.rstrip('/')}/viewer:{user_id}:{agent_id}"
        
        # Build ICE servers (same as camera stream)
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
            "agent_id": agent_id,
            "agent_name": agent.task_name,
        }

