"""
Camera Routes
=============

FastAPI routes for camera management endpoints.
"""
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status, Depends

from agent.api.schemas.camera_schemas import (
    CameraCreateRequest,
    CameraResponse,
    CameraDeleteResponse,
    WebRTCConfigResponse,
)
from agent.api.dependencies.container import get_camera_service
from agent.application.services.camera_service import CameraService

router = APIRouter(prefix="/api", tags=["cameras"])


@router.post(
    "/cameras",
    response_model=CameraResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register or update a camera",
    description="""
    Register a new camera or update an existing one.
    
    When a camera is registered:
    1. Camera config is saved to MongoDB 'cameras' collection
    2. Runner will automatically start CameraPublisher on next poll (every 5 seconds)
    3. CameraPublisher will begin streaming frames to shared_store
    """
)
async def register_camera(
    request: CameraCreateRequest,
    service: CameraService = Depends(get_camera_service),
) -> CameraResponse:
    """Register or update a camera."""
    try:
        camera = service.register_camera(
            camera_id=request.camera_id,
            rtsp_url=request.rtsp_url,
            camera_name=request.camera_name,
            user_id=request.user_id,
            device_id=request.device_id,
        )
        
        return CameraResponse(
            camera_id=camera.camera_id,
            rtsp_url=camera.rtsp_url,
            camera_name=camera.camera_name,
            user_id=camera.user_id,
            device_id=camera.device_id,
            status=camera.status,
            created_at=camera.created_at,
            updated_at=camera.updated_at,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.delete(
    "/cameras/{camera_id}",
    response_model=CameraDeleteResponse,
    summary="Remove a camera",
    description="""
    Remove a camera by setting its status to 'inactive'.
    
    Runner will automatically stop CameraPublisher on next poll.
    """
)
async def remove_camera(
    camera_id: str,
    service: CameraService = Depends(get_camera_service),
) -> CameraDeleteResponse:
    """Remove a camera."""
    try:
        success = service.remove_camera(camera_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Camera '{camera_id}' not found"
            )
        
        return CameraDeleteResponse(
            status="removed",
            camera_id=camera_id,
            message=f"Camera '{camera_id}' has been removed. CameraPublisher will stop on next poll."
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get(
    "/cameras",
    response_model=List[CameraResponse],
    summary="List cameras",
    description="Get list of all cameras, optionally filtered by user_id or status."
)
async def list_cameras(
    user_id: Optional[str] = None,
    status_filter: Optional[str] = None,
    service: CameraService = Depends(get_camera_service),
) -> List[CameraResponse]:
    """List cameras with optional filters."""
    cameras = service.list_cameras(user_id=user_id, status=status_filter)
    
    return [
        CameraResponse(
            camera_id=cam.camera_id,
            rtsp_url=cam.rtsp_url,
            camera_name=cam.camera_name,
            user_id=cam.user_id,
            device_id=cam.device_id,
            status=cam.status,
            created_at=cam.created_at,
            updated_at=cam.updated_at,
        )
        for cam in cameras
    ]


@router.get(
    "/cameras/{camera_id}",
    response_model=CameraResponse,
    summary="Get camera by ID",
    description="Get details of a specific camera."
)
async def get_camera(
    camera_id: str,
    service: CameraService = Depends(get_camera_service),
) -> CameraResponse:
    """Get a specific camera by ID."""
    camera = service.get_camera(camera_id)
    
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{camera_id}' not found"
        )
    
    return CameraResponse(
        camera_id=camera.camera_id,
        rtsp_url=camera.rtsp_url,
        camera_name=camera.camera_name,
        user_id=camera.user_id,
        device_id=camera.device_id,
        status=camera.status,
        created_at=camera.created_at,
        updated_at=camera.updated_at,
    )


@router.get(
    "/stream-config",
    response_model=WebRTCConfigResponse,
    summary="Get WebRTC configuration",
    description="Get WebRTC configuration for frontend (signaling URL and ICE servers)."
)
async def get_stream_config(
    user_id: str,
    service: CameraService = Depends(get_camera_service),
) -> WebRTCConfigResponse:
    """Get WebRTC configuration for a user."""
    try:
        config = service.get_stream_config(user_id)
        return WebRTCConfigResponse(**config)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.get(
    "/cameras/{camera_id}/stream-config",
    response_model=WebRTCConfigResponse,
    summary="Get camera stream configuration",
    description="Get WebRTC configuration for viewing a specific camera's live stream."
)
async def get_camera_stream_config(
    camera_id: str,
    user_id: str,
    service: CameraService = Depends(get_camera_service),
) -> WebRTCConfigResponse:
    """Get WebRTC configuration for a specific camera stream."""
    try:
        config = service.get_camera_stream_config(camera_id, user_id)
        return WebRTCConfigResponse(**config)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get camera stream config: {e}"
        )

