"""
WebRTC Signaling Handler
========================

Handles WebSocket signaling for WebRTC connections.
"""
import json
import asyncio
import os
from typing import Dict, Any, Optional
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.signaling import candidate_from_sdp
from websockets.server import WebSocketServerProtocol

from agent.infrastructure.streaming.webrtc_track import SharedStoreVideoTrack


class SignalingHandler:
    """
    Handles WebRTC signaling via WebSocket.
    
    Manages peer connections and frame streaming for each viewer.
    """
    
    def __init__(self, shared_store: Dict[str, Any]):
        """
        Initialize signaling handler.
        
        Args:
            shared_store: Shared memory dict from multiprocessing.Manager
        """
        self.shared_store = shared_store
        self._active_connections: Dict[str, RTCPeerConnection] = {}
        self._ice_servers = self._build_ice_servers()
    
    def _build_ice_servers(self) -> list:
        """Build ICE servers list from environment variables."""
        ice_servers = [
            RTCIceServer(urls="stun:stun.l.google.com:19302")
        ]
        
        # Add TURN server if configured
        aws_turn_ip = os.getenv("AWS_TURN_IP")
        aws_turn_port = os.getenv("AWS_TURN_PORT")
        aws_turn_user = os.getenv("AWS_TURN_USER")
        aws_turn_pass = os.getenv("AWS_TURN_PASS")
        
        if aws_turn_ip and aws_turn_port and aws_turn_user and aws_turn_pass:
            ice_servers.extend([
                RTCIceServer(
                    urls=f"turn:{aws_turn_ip}:{aws_turn_port}?transport=udp",
                    username=aws_turn_user,
                    credential=aws_turn_pass,
                ),
                RTCIceServer(
                    urls=f"turn:{aws_turn_ip}:{aws_turn_port}?transport=tcp",
                    username=aws_turn_user,
                    credential=aws_turn_pass,
                ),
            ])
            print(f"[streaming] ✅ TURN server configured: {aws_turn_ip}:{aws_turn_port}")
        else:
            # Fallback to hardcoded TURN server (matches frontend DB config)
            ice_servers.extend([
                RTCIceServer(
                    urls="turn:13.49.159.215:3478?transport=udp",
                    username="Algo_webrtc",
                    credential="AlgoOrange2025",
                ),
                RTCIceServer(
                    urls="turn:13.49.159.215:3478?transport=tcp",
                    username="Algo_webrtc",
                    credential="AlgoOrange2025",
                ),
            ])
            print(f"[streaming] ✅ TURN server configured (fallback): 13.49.159.215:3478")
        
        return ice_servers
    
    async def handle_viewer(
        self,
        websocket: WebSocketServerProtocol,
        path: str
    ) -> None:
        """
        Handle a viewer WebSocket connection.
        
        Path format: /viewer:{user_id}/{camera_id}
        Example: /viewer:6928422b8c9933d948cfdc21/CAM-6CBFAA3D8AB5
        """
        pc: Optional[RTCPeerConnection] = None
        camera_id: Optional[str] = None
        connection_id: Optional[str] = None
        
        try:
            # Parse path to extract camera_id
            # Path format: /viewer:{user_id}/{camera_id}
            parts = path.strip("/").split("/")
            if len(parts) < 2:
                print(f"[streaming] ⚠️  Invalid path format: {path}")
                await websocket.close()
                return
            
            # Extract camera_id (last part)
            camera_id = parts[-1]
            connection_id = f"{camera_id}_{id(websocket)}"
            
            # Create peer connection with ICE servers
            config = RTCConfiguration(iceServers=self._ice_servers)
            pc = RTCPeerConnection(configuration=config)
            self._active_connections[connection_id] = pc
            
            # Create video track from shared_store
            track = SharedStoreVideoTrack(camera_id, self.shared_store)
            pc.addTrack(track)
            
            print(f"[streaming] 🎥 WebRTC connection established for camera: {camera_id}")
            
            # Handle signaling messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    
                    if msg_type == "offer":
                        # Set remote description (offer from client)
                        await pc.setRemoteDescription(
                            RTCSessionDescription(
                                sdp=data["sdp"],
                                type=data["type"]
                            )
                        )
                        
                        # Create answer
                        answer = await pc.createAnswer()
                        await pc.setLocalDescription(answer)
                        
                        # Send answer to client
                        await websocket.send(json.dumps({
                            "type": answer.type,
                            "sdp": answer.sdp
                        }))
                        
                        print(f"[streaming] ✅ Sent answer for camera: {camera_id}")
                    
                    elif msg_type == "ice-candidate":
                        # Handle ICE candidate from client
                        candidate = candidate_from_sdp(data["candidate"])
                        await pc.addIceCandidate(candidate)
                        print(f"[streaming] 🧊 Added ICE candidate for camera: {camera_id}")
                
                except json.JSONDecodeError as e:
                    print(f"[streaming] ⚠️  Invalid JSON message: {e}")
                except Exception as e:
                    print(f"[streaming] ⚠️  Error handling message: {e}")
                    break
        
        except Exception as e:
            print(f"[streaming] ❌ Error in viewer connection: {e}")
        finally:
            # Cleanup
            if connection_id and connection_id in self._active_connections:
                del self._active_connections[connection_id]
            if pc:
                try:
                    await pc.close()
                except Exception:
                    pass
            if camera_id:
                print(f"[streaming] 🛑 Closed WebRTC connection for camera: {camera_id}")
    
    def get_active_connections_count(self) -> int:
        """Get number of active WebRTC connections."""
        return len(self._active_connections)

