"""
Dependency Container
====================

Dependency injection container for FastAPI.
Provides singleton instances of repositories and services.
"""
from typing import Dict, Any, Optional

from agent.domain.repositories.camera_repository import CameraRepository
from agent.infrastructure.database.camera_repository_impl import MongoCameraRepository
from agent.application.services.camera_service import CameraService
from agent.application.services.streaming_service import StreamingService
from agent.domain.repositories.agent_repository import AgentRepository
from agent.infrastructure.database.agent_repository_impl import MongoAgentRepository
from agent.application.services.agent_service import AgentService
from agent.domain.repositories.camera_repository import CameraRepository


# Singleton instances
_camera_repository: Optional[CameraRepository] = None
_camera_service: Optional[CameraService] = None
_streaming_service: Optional[StreamingService] = None
_agent_repository: Optional[AgentRepository] = None
_agent_service: Optional[AgentService] = None


def get_camera_repository() -> CameraRepository:
    """
    Get camera repository instance (singleton).
    
    Returns:
        CameraRepository instance
    """
    global _camera_repository
    if _camera_repository is None:
        _camera_repository = MongoCameraRepository()
    return _camera_repository


def get_camera_service() -> CameraService:
    """
    Get camera service instance (singleton).
    
    Returns:
        CameraService instance
    """
    global _camera_service
    if _camera_service is None:
        repository = get_camera_repository()
        _camera_service = CameraService(repository)
    return _camera_service


def get_streaming_service(shared_store: Dict[str, Any]) -> StreamingService:
    """
    Get streaming service instance (singleton).
    
    Args:
        shared_store: Shared memory dict from multiprocessing.Manager
        
    Returns:
        StreamingService instance
    """
    global _streaming_service
    if _streaming_service is None:
        _streaming_service = StreamingService(shared_store)
    return _streaming_service


def get_agent_repository() -> AgentRepository:
    """
    Get agent repository instance (singleton).
    
    Returns:
        AgentRepository instance
    """
    global _agent_repository
    if _agent_repository is None:
        _agent_repository = MongoAgentRepository()
    return _agent_repository


def get_agent_service() -> AgentService:
    """
    Get agent service instance (singleton).
    
    Returns:
        AgentService instance
    """
    global _agent_service
    if _agent_service is None:
        repository = get_agent_repository()
        camera_repository = get_camera_repository()
        _agent_service = AgentService(repository, camera_repository)
    return _agent_service

