"""
Camera API Schemas
==================

Pydantic models for camera API requests and responses.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class CameraCreateRequest(BaseModel):
    """Request schema for creating/updating a camera."""
    camera_id: str = Field(..., description="Unique camera identifier")
    rtsp_url: str = Field(..., description="RTSP stream URL")
    camera_name: str = Field(..., description="Human-readable camera name")
    user_id: str = Field(..., description="User ID who owns this camera")
    device_id: Optional[str] = Field(None, description="Optional device identifier")
    
    class Config:
        schema_extra = {
            "example": {
                "camera_id": "CAM-6CBFAA3D8AB5",
                "rtsp_url": "rtsp://algoorange:algoorange2025@192.168.1.6:554/stream1",
                "camera_name": "samit_cam",
                "user_id": "6928422b8c9933d948cfdc21",
                "device_id": "69465dc123d2a21e3847c065"
            }
        }


class CameraResponse(BaseModel):
    """Response schema for camera data."""
    camera_id: str
    rtsp_url: str
    camera_name: str
    user_id: str
    device_id: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "camera_id": "CAM-6CBFAA3D8AB5",
                "rtsp_url": "rtsp://algoorange:algoorange2025@192.168.1.6:554/stream1",
                "camera_name": "samit_cam",
                "user_id": "6928422b8c9933d948cfdc21",
                "device_id": "69465dc123d2a21e3847c065",
                "status": "active",
                "created_at": "2025-12-20T09:11:50.840Z",
                "updated_at": "2025-12-20T09:11:50.840Z"
            }
        }


class CameraDeleteResponse(BaseModel):
    """Response schema for camera deletion."""
    status: str
    camera_id: str
    message: str


class WebRTCConfigResponse(BaseModel):
    """Response schema for WebRTC configuration."""
    signaling_url: str = Field(..., description="WebSocket URL for WebRTC signaling")
    ice_servers: List[Dict[str, Any]] = Field(..., description="ICE servers for NAT traversal")
    camera_id: Optional[str] = Field(None, description="Camera ID (only for camera-specific config)")
    camera_name: Optional[str] = Field(None, description="Camera name (only for camera-specific config)")
    
    class Config:
        schema_extra = {
            "example": {
                "signaling_url": "ws://localhost:8765/viewer:6928422b8c9933d948cfdc21",
                "ice_servers": [
                    {"urls": "stun:stun.l.google.com:19302"}
                ]
            }
        }

