"""
Camera Service
==============

Application service that coordinates camera-related operations.
This service orchestrates multiple use cases.
"""
from typing import List, Optional

from agent.domain.entities.camera import Camera
from agent.domain.repositories.camera_repository import CameraRepository
from agent.application.use_cases.register_camera import RegisterCameraUseCase
from agent.application.use_cases.remove_camera import RemoveCameraUseCase
from agent.application.use_cases.get_stream_config import GetStreamConfigUseCase
from agent.application.use_cases.get_camera_stream_config import GetCameraStreamConfigUseCase


class CameraService:
    """
    Application service for camera operations.
    
    This service coordinates multiple use cases and provides
    a high-level interface for camera management.
    """
    
    def __init__(self, camera_repository: CameraRepository):
        """
        Initialize service with repository.
        
        Args:
            camera_repository: Repository for camera persistence
        """
        self._repository = camera_repository
        self._register_use_case = RegisterCameraUseCase(camera_repository)
        self._remove_use_case = RemoveCameraUseCase(camera_repository)
        self._get_config_use_case = GetStreamConfigUseCase(camera_repository)
        self._get_camera_config_use_case = GetCameraStreamConfigUseCase(camera_repository)
    
    def register_camera(
        self,
        camera_id: str,
        rtsp_url: str,
        camera_name: str,
        user_id: str,
        device_id: Optional[str] = None,
    ) -> Camera:
        """
        Register or update a camera.
        
        Args:
            camera_id: Unique camera identifier
            rtsp_url: RTSP stream URL
            camera_name: Human-readable camera name
            user_id: User ID who owns the camera
            device_id: Optional device identifier
            
        Returns:
            Registered camera entity
        """
        return self._register_use_case.execute(
            camera_id=camera_id,
            rtsp_url=rtsp_url,
            camera_name=camera_name,
            user_id=user_id,
            device_id=device_id,
        )
    
    def remove_camera(self, camera_id: str) -> bool:
        """
        Remove (deactivate) a camera.
        
        Args:
            camera_id: Unique camera identifier
            
        Returns:
            True if camera was found and removed, False otherwise
        """
        return self._remove_use_case.execute(camera_id)
    
    def get_camera(self, camera_id: str) -> Optional[Camera]:
        """
        Get a camera by ID.
        
        Args:
            camera_id: Unique camera identifier
            
        Returns:
            Camera entity if found, None otherwise
        """
        return self._repository.find_by_id(camera_id)
    
    def list_cameras(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Camera]:
        """
        List cameras with optional filters.
        
        Args:
            user_id: Filter by user ID
            status: Filter by status ('active' or 'inactive')
            
        Returns:
            List of camera entities
        """
        if user_id:
            cameras = self._repository.find_by_user_id(user_id)
        else:
            cameras = self._repository.find_all_active()
        
        if status:
            cameras = [cam for cam in cameras if cam.status == status]
        
        return cameras
    
    def get_stream_config(self, user_id: str) -> dict:
        """
        Get WebRTC stream configuration for a user.
        
        Args:
            user_id: User ID requesting the configuration
            
        Returns:
            Dictionary with signaling_url and ice_servers
        """
        return self._get_config_use_case.execute(user_id)
    
    def get_camera_stream_config(self, camera_id: str, user_id: str) -> dict:
        """
        Get WebRTC stream configuration for a specific camera.
        
        Args:
            camera_id: Camera identifier
            user_id: User ID (for validation)
            
        Returns:
            Dictionary with signaling_url, ice_servers, camera_id, and camera_name
        """
        return self._get_camera_config_use_case.execute(camera_id, user_id)

