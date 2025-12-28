"""
Remove Camera Use Case
======================

Business use case for removing (deactivating) a camera.
"""
from agent.domain.repositories.camera_repository import CameraRepository


class RemoveCameraUseCase:
    """
    Use case for removing a camera.
    
    This encapsulates the business logic for camera removal.
    """
    
    def __init__(self, camera_repository: CameraRepository):
        """
        Initialize use case with repository.
        
        Args:
            camera_repository: Repository for camera persistence
        """
        self._repository = camera_repository
    
    def execute(self, camera_id: str) -> bool:
        """
        Execute the remove camera use case.
        
        Args:
            camera_id: Unique camera identifier
            
        Returns:
            True if camera was found and removed, False otherwise
            
        Raises:
            ValueError: If camera_id is invalid
        """
        if not camera_id or not camera_id.strip():
            raise ValueError("Camera ID is required")
        
        return self._repository.delete(camera_id.strip())

