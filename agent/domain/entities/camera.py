"""
Camera Entity
=============

Domain entity representing a camera in the system.
This is a pure domain object with no infrastructure dependencies.
"""
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class Camera:
    """
    Camera domain entity.
    
    Represents a camera in the system with all its business attributes.
    This entity is independent of any persistence mechanism.
    """
    camera_id: str
    rtsp_url: str
    camera_name: str
    user_id: str
    device_id: Optional[str] = None
    status: str = "active"  # "active" | "inactive"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def activate(self) -> None:
        """Activate the camera."""
        self.status = "active"
        self.updated_at = datetime.utcnow()
    
    def deactivate(self) -> None:
        """Deactivate the camera."""
        self.status = "inactive"
        self.updated_at = datetime.utcnow()
    
    def is_active(self) -> bool:
        """Check if camera is active."""
        return self.status == "active"
    
    def update_rtsp_url(self, new_url: str) -> None:
        """Update RTSP URL."""
        if not new_url or not new_url.strip():
            raise ValueError("RTSP URL cannot be empty")
        self.rtsp_url = new_url.strip()
        self.updated_at = datetime.utcnow()
    
    def update_name(self, new_name: str) -> None:
        """Update camera name."""
        if not new_name or not new_name.strip():
            raise ValueError("Camera name cannot be empty")
        self.camera_name = new_name.strip()
        self.updated_at = datetime.utcnow()

