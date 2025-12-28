"""
AWS Signaling Client
====================

Connects to AWS signaling server as camera client and manages WebRTC streams.
"""
import asyncio
import json
import os
from typing import Dict, Any, Optional
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.signaling import candidate_from_sdp

from agent.infrastructure.streaming.webrtc_track import SharedStoreVideoTrack


class AWSSignalingClient:
    """
    Client that connects to AWS signaling server as camera:{user_id}
    and manages WebRTC peer connections for streaming.
    """
    
    def __init__(self, shared_store: Dict[str, Any], user_id: str, camera_id: str):
        """
        Initialize AWS signaling client.
        
        Args:
            shared_store: Shared memory dict with frames
            user_id: User ID for this camera
            camera_id: Camera ID
        """
        self.shared_store = shared_store
        self.user_id = user_id
        self.camera_id = camera_id
        self.client_id = f"camera:{user_id}:{camera_id}"
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.pc: Optional[RTCPeerConnection] = None
        self._active_connections: Dict[str, RTCPeerConnection] = {}  # viewer_id -> peer_connection
        self._running = False
        self._ice_servers = self._build_ice_servers()
        self._candidate_types = []  # Track candidate types for verification
    
    def _build_ice_servers(self) -> list:
        """Build ICE servers list."""
        ice_servers = [RTCIceServer(urls="stun:stun.l.google.com:19302")]
        print(f"[aws-client {self.camera_id}] 📡 STUN server configured: stun:stun.l.google.com:19302")

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
            print(f"[aws-client {self.camera_id}] 🎯 TURN server configured: {aws_turn_ip}:{aws_turn_port} (UDP + TCP)")
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
            print(f"[aws-client {self.camera_id}] 🎯 TURN server configured (fallback): 13.49.159.215:3478 (UDP + TCP)")

        print(f"[aws-client {self.camera_id}] ✅ Total ICE servers: {len(ice_servers)}")
        return ice_servers
    
    async def connect_and_stream(self) -> None:
        """
        Connect to AWS signaling server and start streaming.
        """
        if self._running:
            print(f"[aws-client {self.camera_id}] ⚠️  Already running")
            return
        
        # Get AWS signaling server URL
        aws_signaling_url = os.getenv("AWS_SIGNALING_URL", "ws://localhost:8000")
        if not aws_signaling_url or aws_signaling_url == "ws://localhost:8000":
            print(f"[aws-client {self.camera_id}] ⚠️  AWS_SIGNALING_URL not set! Using default localhost:8000")
            print(f"[aws-client {self.camera_id}] ⚠️  Please set AWS_SIGNALING_URL in .env file (e.g., ws://13.61.105.168:8000)")
        
        ws_url = f"{aws_signaling_url.rstrip('/')}/ws/{self.client_id}"
        
        print(f"[aws-client {self.camera_id}] 🔌 Connecting to AWS: {ws_url}")
        
        reconnect_delay = 5
        
        while True:
            try:
                self._running = True
                
                # Connect to AWS signaling server
                async with websockets.connect(
                    ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5
                ) as websocket:
                    self.ws = websocket
                    print(f"[aws-client {self.camera_id}] ✅ Connected to AWS signaling server")
                    
                    # Create WebRTC peer connection
                    config = RTCConfiguration(iceServers=self._ice_servers)
                    self.pc = RTCPeerConnection(configuration=config)
                    
                    # Add video track from shared_store
                    track = SharedStoreVideoTrack(self.camera_id, self.shared_store)
                    self.pc.addTrack(track)
                    print(f"[aws-client {self.camera_id}] 🎥 Added video track")
                    
                    # Create and send offer
                    offer = await self.pc.createOffer()
                    await self.pc.setLocalDescription(offer)
                    
                    # Send offer to AWS (will be forwarded to viewers)
                    viewer_id = f"viewer:{self.user_id}:{self.camera_id}"
                    offer_msg = {
                        "type": "offer",
                        "from": self.client_id,
                        "to": viewer_id,
                        "sdp": offer.sdp
                    }
                    await websocket.send(json.dumps(offer_msg))
                    print(f"[aws-client {self.camera_id}] 📤 Sent offer to AWS (signaling state: {self.pc.signalingState})")
                    
                    # Monitor connection state - track if we need to reconnect
                    connection_needs_reconnect = False
                    
                    @self.pc.on("connectionstatechange")
                    async def on_connection_state_change():
                        nonlocal connection_needs_reconnect
                        state = self.pc.connectionState
                        print(f"[aws-client {self.camera_id}] 🔄 Connection state: {state}")
                        if state == "closed" or state == "failed":
                            print(f"[aws-client {self.camera_id}] ⚠️  Connection {state}, will reconnect...")
                            connection_needs_reconnect = True
                    
                    @self.pc.on("iceconnectionstatechange")
                    async def on_ice_connection_state_change():
                        nonlocal connection_needs_reconnect
                        state = self.pc.iceConnectionState
                        print(f"[aws-client {self.camera_id}] 🧊 ICE connection state: {state}")
                        
                        if state == "failed":
                            print(f"[aws-client {self.camera_id}] ❌ ICE connection failed - STUN and TURN both failed")
                            print(f"[aws-client {self.camera_id}] ⚠️  Check: 1) TURN server accessible, 2) TURN credentials correct, 3) Firewall allows TURN")
                            connection_needs_reconnect = True
                        elif state == "closed":
                            print(f"[aws-client {self.camera_id}] ⚠️  ICE connection closed, will reconnect...")
                            connection_needs_reconnect = True
                        elif state == "checking":
                            print(f"[aws-client {self.camera_id}] 🔍 ICE checking - trying to establish connection...")
                        elif state == "connected" or state == "completed":
                            # Show summary based on gathered candidates
                            turn_count = self._candidate_types.count("turn")
                            stun_count = self._candidate_types.count("stun")
                            host_count = self._candidate_types.count("host")
                            
                            if turn_count > 0:
                                print(f"[aws-client {self.camera_id}] ✅ ICE connection established - TURN relay candidates available (cross-network capable)")
                            elif stun_count > 0:
                                print(f"[aws-client {self.camera_id}] ✅ ICE connection established - STUN reflexive candidates available (same network/simple NAT)")
                            else:
                                print(f"[aws-client {self.camera_id}] ✅ ICE connection established - Direct/host connection")
                    
                    # Handle ICE candidates
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
                                    print(f"[aws-client {self.camera_id}] 📡 STUN candidate gathered (reflexive)")
                                elif "typ relay" in cand_str or "typ 2" in cand_str:
                                    candidate_type = "relay (TURN)"  # This is what we need for mobile!
                                    self._candidate_types.append("turn")
                                    print(f"[aws-client {self.camera_id}] 🎯 TURN relay candidate gathered! (for cross-network NAT traversal)")
                                else:
                                    candidate_type = f"other ({candidate.candidate[:50]}...)"
                                    self._candidate_types.append("other")
                            
                            if candidate_type not in ["relay (TURN)", "srflx (STUN)"]:
                                print(f"[aws-client {self.camera_id}] 📡 ICE candidate: {candidate_type}")
                            
                            ice_msg = {
                                "type": "ice",
                                "from": self.client_id,
                                "to": viewer_id,
                                "candidate": {
                                    "candidate": candidate.candidate,
                                    "sdpMid": candidate.sdpMid,
                                    "sdpMLineIndex": candidate.sdpMLineIndex
                                }
                            }
                            try:
                                await self.ws.send(json.dumps(ice_msg))
                            except Exception as e:
                                print(f"[aws-client {self.camera_id}] ⚠️  Error sending ICE: {e}")
                        elif not candidate:
                            # End of candidates - show summary
                            turn_count = self._candidate_types.count("turn")
                            stun_count = self._candidate_types.count("stun")
                            host_count = self._candidate_types.count("host")
                            print(f"[aws-client {self.camera_id}] ✅ ICE gathering complete - Candidates: {turn_count} TURN, {stun_count} STUN, {host_count} Host")
                    
                    # Handle messages from AWS
                    async for message in websocket:
                        if not self._running:
                            print(f"[aws-client {self.camera_id}] 🛑 Stopping (running=False)")
                            break
                        
                        # Check if connection closed and needs reconnection
                        if connection_needs_reconnect:
                            print(f"[aws-client {self.camera_id}] 🔄 Connection closed, breaking message loop to reconnect...")
                            break
                            
                        try:
                            data = json.loads(message)
                            msg_type = data.get("type")
                            
                            if msg_type == "answer":
                                # Received answer from viewer
                                current_state = self.pc.signalingState
                                viewer_id = data.get("from", "unknown")
                                
                                # Process answer if we're in the right signaling state
                                if current_state in ["have-local-offer", "have-remote-offer"]:
                                    try:
                                        await self.pc.setRemoteDescription(
                                            RTCSessionDescription(
                                                sdp=data["sdp"],
                                                type="answer"
                                            )
                                        )
                                        print(f"[aws-client {self.camera_id}] ✅ Received and processed answer from viewer: {viewer_id}")
                                    except Exception as e:
                                        print(f"[aws-client {self.camera_id}] ⚠️  Error setting remote description: {e}")
                                
                                # If state is stable, it means we already have an active connection
                                # In WebRTC, one peer connection = one viewer connection
                                # Multiple viewers need multiple peer connections, but we're using one connection
                                # So we ignore subsequent answers to avoid conflicts
                                elif current_state == "stable":
                                    print(f"[aws-client {self.camera_id}] ℹ️  Ignoring answer from {viewer_id} - connection already established (state: stable)")
                                    print(f"[aws-client {self.camera_id}] ℹ️  Note: Multiple viewers require separate peer connections per viewer")
                                
                                else:
                                    print(f"[aws-client {self.camera_id}] ⚠️  Ignoring answer from {viewer_id} - signaling state is '{current_state}'")
                            
                            elif msg_type == "ice":
                                # Received ICE candidate from viewer
                                candidate_data = data.get("candidate", {})
                                if candidate_data and candidate_data.get("candidate"):
                                    try:
                                        # Create candidate with proper fields
                                        candidate_str = candidate_data.get("candidate")
                                        sdp_mid = candidate_data.get("sdpMid")
                                        sdp_mline_index = candidate_data.get("sdpMLineIndex")
                                        
                                        if not sdp_mid and sdp_mline_index is None:
                                            print(f"[aws-client {self.camera_id}] ⚠️  ICE candidate missing sdpMid and sdpMLineIndex, skipping")
                                            continue
                                        
                                        candidate = candidate_from_sdp(candidate_str)
                                        if sdp_mid:
                                            candidate.sdpMid = sdp_mid
                                        if sdp_mline_index is not None:
                                            candidate.sdpMLineIndex = sdp_mline_index
                                        
                                        await self.pc.addIceCandidate(candidate)
                                        print(f"[aws-client {self.camera_id}] 🧊 Added ICE candidate from viewer")
                                    except Exception as e:
                                        print(f"[aws-client {self.camera_id}] ⚠️  Error adding ICE candidate: {e}")
                                elif candidate_data and not candidate_data.get("candidate"):
                                    # End of candidates
                                    try:
                                        await self.pc.addIceCandidate(None)
                                        print(f"[aws-client {self.camera_id}] ✅ ICE candidates complete")
                                    except Exception as e:
                                        print(f"[aws-client {self.camera_id}] ⚠️  Error adding end-of-candidates: {e}")
                            
                            elif msg_type == "pong":
                                # Keep-alive
                                pass
                        
                        except json.JSONDecodeError as e:
                            print(f"[aws-client {self.camera_id}] ⚠️  Invalid JSON: {e}")
                        except Exception as e:
                            print(f"[aws-client {self.camera_id}] ⚠️  Error handling message: {e}")
            
            except websockets.exceptions.InvalidStatusCode as e:
                if self._running:
                    print(f"[aws-client {self.camera_id}] ❌ HTTP {e.status_code} error connecting to AWS")
                    print(f"[aws-client {self.camera_id}] ⚠️  Check: 1) AWS_SIGNALING_URL is correct, 2) Server is accessible, 3) Firewall allows connection")
                    await asyncio.sleep(reconnect_delay)
                else:
                    break
            except websockets.exceptions.ConnectionClosed:
                if self._running:
                    print(f"[aws-client {self.camera_id}] 🔌 WebSocket connection closed, reconnecting in {reconnect_delay}s...")
                    # Clean up peer connection
                    if self.pc:
                        try:
                            await self.pc.close()
                        except Exception:
                            pass
                        self.pc = None
                    await asyncio.sleep(reconnect_delay)
                    continue  # Continue to reconnect
                else:
                    break
            except Exception as e:
                if self._running:
                    error_msg = str(e)
                    if "403" in error_msg or "Forbidden" in error_msg:
                        print(f"[aws-client {self.camera_id}] ❌ HTTP 403 Forbidden - Server rejected connection")
                        print(f"[aws-client {self.camera_id}] ⚠️  Possible causes: Authentication required, CORS issue, or server security settings")
                    else:
                        print(f"[aws-client {self.camera_id}] ❌ Error: {e}")
                    print(f"[aws-client {self.camera_id}] Reconnecting in {reconnect_delay}s...")
                    # Clean up peer connection
                    if self.pc:
                        try:
                            await self.pc.close()
                        except Exception:
                            pass
                        self.pc = None
                    await asyncio.sleep(reconnect_delay)
                    continue  # Continue to reconnect
                else:
                    break
            finally:
                # Only clean up if we're not reconnecting
                if not self._running:
                    if self.pc:
                        try:
                            await self.pc.close()
                        except Exception:
                            pass
                        self.pc = None
        
        print(f"[aws-client {self.camera_id}] 🛑 Disconnected from AWS")
    
    async def stop(self) -> None:
        """Stop the client."""
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

