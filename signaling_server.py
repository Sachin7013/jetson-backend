# signaling_server.py
import json
import logging
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [SIGNALING] %(levelname)s: %(message)s'
)
logger = logging.getLogger("signaling")
print("[DEBUG] Logging configured for signaling server", flush=True)

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Client State
# --------------------------------------------------
class Client:
    def __init__(self, client_id: str, ws: WebSocket):
        self.id = client_id
        self.ws = ws
        # camera OR agent are publishers
        self.is_publisher = client_id.startswith("camera") or client_id.startswith("agent")
        self.last_offer = None
        print(f"[DEBUG] Client created id={self.id} is_publisher={self.is_publisher}", flush=True)

clients: Dict[str, Client] = {}

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
# def get_publisher_id_for_viewer(viewer_id: str) -> str | None:
#     """
#     viewer:<user>:<camera>            -> camera:<user>
#     viewer:<user>:<agent>             -> agent:<user>:<agent>
#     """
#     parts = viewer_id.split(":")
#     if len(parts) == 2:
#         # viewer:user
#         return f"camera:{parts[1]}"
#     if len(parts) == 3:
#         # viewer:user:camera_id OR viewer:user:agent_id
#         return f"agent:{parts[1]}:{parts[2]}"
#     return None

def get_publisher_id_for_viewer(viewer_id: str) -> str | None:
    parts = viewer_id.split(":")
    
    # viewer:user:camera
    if len(parts) == 3:
        user_id, camera_id = parts[1], parts[2]
        return f"camera:{user_id}:{camera_id}"
    
    # viewer:user:camera:agent
    if len(parts) == 4:
        user_id, camera_id, agent_id = parts[1], parts[2], parts[3]
        return f"agent:{user_id}:{camera_id}:{agent_id}"
    
    return None


# --------------------------------------------------
# WebSocket Endpoint
# --------------------------------------------------
@app.websocket("/ws/{client_id}")
async def ws_endpoint(ws: WebSocket, client_id: str):
    print(f"[DEBUG] ws_endpoint start client_id={client_id}", flush=True)
    await ws.accept()
    print(f"[DEBUG] WebSocket accepted for client_id={client_id}", flush=True)
    client = Client(client_id, ws)
    clients[client_id] = client
    print(f"[DEBUG] Client stored. Total clients now: {len(clients)}", flush=True)

    role = "PUBLISHER" if client.is_publisher else "VIEWER"
    print(f"[DEBUG] Role determined for {client_id}: {role}", flush=True)
    logger.info(f"✅ {role} connected: {client_id}")

    # --------------------------------------------------
    # If viewer joins late → replay last offer
    # --------------------------------------------------
    if not client.is_publisher:
        print(f"[DEBUG] {client_id} is a viewer. Attempting to replay last offer.", flush=True)
        publisher_id = get_publisher_id_for_viewer(client_id)
        print(f"[DEBUG] Computed publisher_id for {client_id}: {publisher_id}", flush=True)
        if publisher_id and publisher_id in clients:
            print(f"[DEBUG] Publisher connected status for {publisher_id}: {publisher_id in clients}", flush=True)
            publisher = clients[publisher_id]
            print(f"[DEBUG] Publisher client object obtained for {publisher_id}", flush=True)
            if publisher.last_offer:
                print(f"[DEBUG] Replaying last offer from {publisher_id} to {client_id}", flush=True)
                await ws.send_text(json.dumps({
                    "type": "offer",
                    "from": publisher_id,
                    "to": client_id,
                    "sdp": publisher.last_offer
                }))
                logger.info(f"📤 Replayed offer {publisher_id} → {client_id}")
            else:
                print(f"[DEBUG] No last offer available for publisher {publisher_id}", flush=True)
        else:
            print(f"[DEBUG] Publisher not connected or unresolved for viewer {client_id}. publisher_id={publisher_id}", flush=True)

    try:
        while True:
            print(f"[DEBUG] Waiting to receive message from {client.id}", flush=True)
            data = await ws.receive_text()
            print(f"[DEBUG] Received raw data from {client.id}: {data}", flush=True)
            msg = json.loads(data)
            print(f"[DEBUG] Parsed message dict: {msg}", flush=True)

            msg_type = msg.get("type")
            target = msg.get("to")
            print(f"[DEBUG] Message type={msg_type} from={client.id} to={target}", flush=True)

            # --------------------------------------------------
            # OFFER (camera / agent → viewer)
            # --------------------------------------------------
            if msg_type == "offer" and client.is_publisher:
                client.last_offer = msg.get("sdp")
                logger.info(f"📨 OFFER from {client.id}")

                # 🔥 CASE 1: AGENT → SPECIFIC VIEWER
                if target and target in clients:
                    await clients[target].ws.send_text(data)
                    logger.info(f"📤 OFFER forwarded → {target}")

                # 🔥 CASE 2: CAMERA OFFER → IGNORE (camera should NOT talk to viewer)
                else:
                    logger.warning(
                        f"⚠️ Ignoring offer from {client.id} (no target). "
                        "Camera offers are not routed directly."
                    )

            # --------------------------------------------------
            # ANSWER (viewer → camera / agent)
            # --------------------------------------------------
            elif msg_type == "answer" and not client.is_publisher:
                if target and target in clients:
                    print(f"[DEBUG] Handling ANSWER from viewer {client.id}", flush=True)
                    print(f"[DEBUG] Forwarding ANSWER to {target}", flush=True)
                    await clients[target].ws.send_text(data)
                    logger.info(f"📤 ANSWER forwarded → {target}")
                else:
                    print(f"[DEBUG] ANSWER has invalid or disconnected target: {target}", flush=True)

            # --------------------------------------------------
            # ICE
            # --------------------------------------------------
            elif msg_type == "ice":
                print(f"[DEBUG] Handling ICE from {client.id} to {target}", flush=True)
                if target and target in clients:
                    print(f"[DEBUG] Forwarding ICE to {target}", flush=True)
                    await clients[target].ws.send_text(data)
                else:
                    print(f"[DEBUG] ICE has invalid or disconnected target: {target}", flush=True)

            # --------------------------------------------------
            # PING
            # --------------------------------------------------
            elif msg_type == "ping":
                print(f"[DEBUG] Received PING from {client.id}; sending PONG", flush=True)
                await ws.send_text(json.dumps({"type": "pong"}))

            else:
                print(f"[DEBUG] Unknown message type received: {msg_type} from {client.id} -> {target}. Full message: {msg}", flush=True)

    except WebSocketDisconnect:
        print(f"[DEBUG] WebSocketDisconnect caught for {client_id}", flush=True)
        logger.info(f"🔌 Disconnected: {client_id}")
    finally:
        print(f"[DEBUG] Removing client from registry: {client_id}", flush=True)
        clients.pop(client_id, None)
        logger.info(f"📊 Active clients: {len(clients)}")
        print(f"[DEBUG] Active clients count: {len(clients)}", flush=True)

# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    print("[DEBUG] Starting signaling server on port 8002", flush=True)
    uvicorn.run(app, port=8002)
