"""
Register Camera Use Case
========================

Business use case for registering a new camera or updating an existing one.
"""
from typing import Optional
from datetime import datetime

from agent.domain.entities.camera import Camera
from agent.domain.repositories.camera_repository import CameraRepository


class RegisterCameraUseCase:
    """
    Use case for registering or updating a camera.
    
    This encapsulates the business logic for camera registration.
    """
    
    def __init__(self, camera_repository: CameraRepository):
        """
        Initialize use case with repository.
        
        Args:
            camera_repository: Repository for camera persistence
        """
        self._repository = camera_repository
    
    def execute(
        self,
        camera_id: str,
        rtsp_url: str,
        camera_name: str,
        user_id: str,
        device_id: Optional[str] = None,
    ) -> Camera:
        """
        Execute the register camera use case.
        
        Args:
            camera_id: Unique camera identifier
            rtsp_url: RTSP stream URL
            camera_name: Human-readable camera name
            user_id: User ID who owns the camera
            device_id: Optional device identifier
            
        Returns:
            Registered camera entity
            
        Raises:
            ValueError: If input validation fails
        """
        # Validate inputs
        if not camera_id or not camera_id.strip():
            raise ValueError("Camera ID is required")
        if not rtsp_url or not rtsp_url.strip():
            raise ValueError("RTSP URL is required")
        if not camera_name or not camera_name.strip():
            raise ValueError("Camera name is required")
        if not user_id or not user_id.strip():
            raise ValueError("User ID is required")
        
        # Check if camera already exists
        existing_camera = self._repository.find_by_id(camera_id)
        
        if existing_camera:
            # Update existing camera
            existing_camera.update_rtsp_url(rtsp_url)
            existing_camera.update_name(camera_name)
            existing_camera.device_id = device_id
            existing_camera.activate()  # Ensure it's active
            return self._repository.update(existing_camera)
        else:
            # Create new camera
            new_camera = Camera(
                camera_id=camera_id.strip(),
                rtsp_url=rtsp_url.strip(),
                camera_name=camera_name.strip(),
                user_id=user_id.strip(),
                device_id=device_id,
                status="active",
            )
            return self._repository.create(new_camera)

