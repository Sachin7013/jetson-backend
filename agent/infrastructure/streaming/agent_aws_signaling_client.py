"""
Agent AWS Signaling Client
===========================

Connects to AWS signaling server as an agent client for streaming processed frames.
"""
import asyncio
import json
import os
from typing import Dict, Any, Optional
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.signaling import candidate_from_sdp

from agent.infrastructure.streaming.agent_webrtc_track import AgentSharedStoreVideoTrack


class AgentAWSSignalingClient:
    """
    AWS signaling client for agent-specific streams.
    
    Connects to AWS signaling server as agent:{user_id}:{agent_id}
    and streams processed frames (with bounding boxes).
    """
    
    def __init__(self, shared_store: Dict[str, Any], user_id: str, agent_id: str, camera_id: str):
        """
        Initialize agent AWS signaling client.
        
        Args:
            shared_store: Shared memory dict from multiprocessing.Manager
            user_id: User ID
            agent_id: Agent ID
        """
        self.shared_store = shared_store
        self.user_id = user_id
        self.agent_id = agent_id
        self.camera_id = camera_id
        self.client_id = f"agent:{self.user_id}:{self.camera_id}:{self.agent_id}"
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.pc: Optional[RTCPeerConnection] = None
        self._running = False
        self._ice_servers = self._build_ice_servers()
        self._candidate_types = []  # Track candidate types for verification
    
    def _build_ice_servers(self) -> list:
        """Build ICE servers list (same as camera client)."""
        ice_servers = [RTCIceServer(urls="stun:stun.l.google.com:19302")]

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
            print(f"[agent-aws-client {self.agent_id}] 🎯 TURN server configured: {aws_turn_ip}:{aws_turn_port} (UDP + TCP)")
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
            print(f"[agent-aws-client {self.agent_id}] 🎯 TURN server configured (fallback): 13.49.159.215:3478 (UDP + TCP)")

        return ice_servers
    
    async def connect_and_stream(self) -> None:
        """Connect to AWS and stream processed frames."""
        if self._running:
            print(f"[agent-aws-client {self.agent_id}] ⚠️  Already running")
            return
        
        aws_signaling_url = os.getenv("AWS_SIGNALING_URL", "ws://localhost:8000")
        if not aws_signaling_url or aws_signaling_url == "ws://localhost:8000":
            print(f"[agent-aws-client {self.agent_id}] ⚠️  AWS_SIGNALING_URL not set! Using default localhost:8000")

        ws_url = f"{aws_signaling_url.rstrip('/')}/ws/{self.client_id}"

        print(f"[agent-aws-client {self.agent_id}] 🔌 Connecting to AWS: {ws_url}")
        
        reconnect_delay = 5
        
        while True:
            try:
                self._running = True
                
                if self.pc:
                    try:
                        await self.pc.close()
                    except Exception:
                        pass
                    self.pc = None
                
                async with websockets.connect(
                    ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5
                ) as websocket:
                    self.ws = websocket
                    print(f"[agent-aws-client {self.agent_id}] ✅ Connected to AWS signaling server")
                    
                    config = RTCConfiguration(iceServers=self._ice_servers)
                    self.pc = RTCPeerConnection(configuration=config)
                    
                    track = AgentSharedStoreVideoTrack(self.agent_id, self.shared_store)
                    self.pc.addTrack(track)
                    print(f"[agent-aws-client {self.agent_id}] 🎥 Added processed video track")
                    
                    offer = await self.pc.createOffer()
                    await self.pc.setLocalDescription(offer)
                    
                    # Target the specific viewer for this agent stream
                    viewer_id = f"viewer:{self.user_id}:{self.camera_id}:{self.agent_id}"
                    offer_msg = {
                        "type": "offer",
                        "from": self.client_id,
                        "to": viewer_id,
                        "sdp": offer.sdp
                    }
                    await websocket.send(json.dumps(offer_msg))
                    print(f"[agent-aws-client {self.agent_id}] 📤 Sent offer to AWS (target: {viewer_id})")
                    
                    connection_needs_reconnect = False
                    
                    @self.pc.on("connectionstatechange")
                    async def on_connection_state_change():
                        nonlocal connection_needs_reconnect
                        state = self.pc.connectionState
                        print(f"[agent-aws-client {self.agent_id}] 🔄 Connection state: {state}")
                        if state == "closed" or state == "failed":
                            connection_needs_reconnect = True
                    
                    @self.pc.on("iceconnectionstatechange")
                    async def on_ice_connection_state_change():
                        nonlocal connection_needs_reconnect
                        state = self.pc.iceConnectionState
                        print(f"[agent-aws-client {self.agent_id}] 🧊 ICE connection state: {state}")
                        if state == "closed" or state == "failed":
                            connection_needs_reconnect = True
                        elif state == "connected" or state == "completed":
                            # Show summary based on gathered candidates
                            turn_count = self._candidate_types.count("turn")
                            stun_count = self._candidate_types.count("stun")
                            host_count = self._candidate_types.count("host")
                            
                            if turn_count > 0:
                                print(f"[agent-aws-client {self.agent_id}] ✅ ICE connection established - TURN relay candidates available (cross-network capable)")
                            elif stun_count > 0:
                                print(f"[agent-aws-client {self.agent_id}] ✅ ICE connection established - STUN reflexive candidates available (same network/simple NAT)")
                            else:
                                print(f"[agent-aws-client {self.agent_id}] ✅ ICE connection established - Direct/host connection")
                    
                    @self.pc.on("icecandidate")
                    async def on_icecandidate(event):
                        candidate = event.candidate
                        if candidate and self.ws:
                            # Log candidate type to diagnose TURN usage
                            candidate_type = "unknown"
                            if candidate.candidate:
                                cand_str = candidate.candidate.lower()
                                if "typ host" in cand_str or "typ 0" in cand_str:
                                    candidate_type = "host (direct)"
                                    self._candidate_types.append("host")
                                elif "typ srflx" in cand_str or "typ 1" in cand_str:
                                    candidate_type = "srflx (STUN)"
                                    self._candidate_types.append("stun")
                                    print(f"[agent-aws-client {self.agent_id}] 📡 STUN candidate gathered (reflexive)")
                                elif "typ relay" in cand_str or "typ 2" in cand_str:
                                    candidate_type = "relay (TURN)"
                                    self._candidate_types.append("turn")
                                    print(f"[agent-aws-client {self.agent_id}] 🎯 TURN relay candidate gathered! (for cross-network NAT traversal)")
                                else:
                                    candidate_type = f"other ({candidate.candidate[:50]}...)"
                                    self._candidate_types.append("other")
                            
                            ice_msg = {
                                "type": "ice",
                                "from": self.client_id,
                                "to": viewer_id,  # Route to specific viewer
                                "candidate": {
                                    "candidate": candidate.candidate,
                                    "sdpMid": candidate.sdpMid,
                                    "sdpMLineIndex": candidate.sdpMLineIndex
                                }
                            }
                            try:
                                await self.ws.send(json.dumps(ice_msg))
                            except Exception as e:
                                print(f"[agent-aws-client {self.agent_id}] ⚠️  Error sending ICE: {e}")
                        elif not candidate and self.ws:
                            # End of candidates - show summary
                            turn_count = self._candidate_types.count("turn")
                            stun_count = self._candidate_types.count("stun")
                            host_count = self._candidate_types.count("host")
                            print(f"[agent-aws-client {self.agent_id}] ✅ ICE gathering complete - Candidates: {turn_count} TURN, {stun_count} STUN, {host_count} Host")
                            
                            ice_msg = {
                                "type": "ice",
                                "from": self.client_id,
                                "to": viewer_id,
                                "candidate": {}
                            }
                            try:
                                await self.ws.send(json.dumps(ice_msg))
                                print(f"[agent-aws-client {self.agent_id}] ✅ Sent end-of-candidates to {viewer_id}")
                            except Exception as e:
                                print(f"[agent-aws-client {self.agent_id}] ⚠️  Error sending end-of-candidates: {e}")
                    
                    async for message in websocket:
                        if not self._running:
                            break
                        
                        if connection_needs_reconnect:
                            break
                        
                        try:
                            data = json.loads(message)
                            msg_type = data.get("type")
                            
                            if msg_type == "answer":
                                current_state = self.pc.signalingState
                                if current_state in ["have-local-offer", "have-remote-offer"]:
                                    try:
                                        await self.pc.setRemoteDescription(
                                            RTCSessionDescription(
                                                sdp=data["sdp"],
                                                type="answer"
                                            )
                                        )
                                        print(f"[agent-aws-client {self.agent_id}] ✅ Received and processed answer")
                                    except Exception as e:
                                        print(f"[agent-aws-client {self.agent_id}] ⚠️  Error setting remote description: {e}")
                                elif current_state == "stable":
                                    print(f"[agent-aws-client {self.agent_id}] ℹ️  Ignoring answer - connection already established")
                            
                            elif msg_type == "ice":
                                candidate_data = data.get("candidate", {})
                                if candidate_data and candidate_data.get("candidate"):
                                    try:
                                        candidate_str = candidate_data.get("candidate")
                                        candidate = candidate_from_sdp(candidate_str)
                                        if candidate_data.get("sdpMid"):
                                            candidate.sdpMid = candidate_data.get("sdpMid")
                                        if candidate_data.get("sdpMLineIndex") is not None:
                                            candidate.sdpMLineIndex = candidate_data.get("sdpMLineIndex")
                                        await self.pc.addIceCandidate(candidate)
                                    except Exception as e:
                                        print(f"[agent-aws-client {self.agent_id}] ⚠️  Error adding ICE candidate: {e}")
                                elif candidate_data and not candidate_data.get("candidate"):
                                    try:
                                        await self.pc.addIceCandidate(None)
                                        print(f"[agent-aws-client {self.agent_id}] ✅ ICE candidates complete")
                                    except Exception as e:
                                        print(f"[agent-aws-client {self.agent_id}] ⚠️  Error adding end-of-candidates: {e}")
                        
                        except json.JSONDecodeError:
                            pass
                        except Exception as e:
                            print(f"[agent-aws-client {self.agent_id}] ⚠️  Error handling message: {e}")
            
            except Exception as e:
                if self._running:
                    print(f"[agent-aws-client {self.agent_id}] ❌ Error: {e}, reconnecting in {reconnect_delay}s...")
                    await asyncio.sleep(reconnect_delay)
                else:
                    break
            finally:
                if self.pc:
                    try:
                        await self.pc.close()
                    except Exception:
                        pass
                    self.pc = None
                
                if not self._running:
                    break
        
        print(f"[agent-aws-client {self.agent_id}] 🛑 Disconnected from AWS")
    
    async def stop(self) -> None:
        """Stop the agent stream."""
        self._running = False
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
        if self.pc:
            try:
                await self.pc.close()
            except Exception:
                pass

