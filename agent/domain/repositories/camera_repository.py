"""
Camera Repository Interface
===========================

Abstract interface for camera data access.
Implementations should be in the infrastructure layer.
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from agent.domain.entities.camera import Camera


class CameraRepository(ABC):
    """
    Abstract repository for camera persistence operations.
    
    This interface defines the contract for camera data access.
    Concrete implementations should be in the infrastructure layer.
    """
    
    @abstractmethod
    def create(self, camera: Camera) -> Camera:
        """
        Create a new camera.
        
        Args:
            camera: Camera entity to create
            
        Returns:
            Created camera entity
        """
        pass
    
    @abstractmethod
    def update(self, camera: Camera) -> Camera:
        """
        Update an existing camera.
        
        Args:
            camera: Camera entity with updated data
            
        Returns:
            Updated camera entity
        """
        pass
    
    @abstractmethod
    def find_by_id(self, camera_id: str) -> Optional[Camera]:
        """
        Find a camera by its ID.
        
        Args:
            camera_id: Unique camera identifier
            
        Returns:
            Camera entity if found, None otherwise
        """
        pass
    
    @abstractmethod
    def find_by_user_id(self, user_id: str) -> List[Camera]:
        """
        Find all cameras for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of camera entities
        """
        pass
    
    @abstractmethod
    def find_all_active(self) -> List[Camera]:
        """
        Find all active cameras.
        
        Returns:
            List of active camera entities
        """
        pass
    
    @abstractmethod
    def delete(self, camera_id: str) -> bool:
        """
        Delete (deactivate) a camera.
        
        Args:
            camera_id: Unique camera identifier
            
        Returns:
            True if camera was found and deleted, False otherwise
        """
        pass
    
    @abstractmethod
    def exists(self, camera_id: str) -> bool:
        """
        Check if a camera exists.
        
        Args:
            camera_id: Unique camera identifier
            
        Returns:
            True if camera exists, False otherwise
        """
        pass

