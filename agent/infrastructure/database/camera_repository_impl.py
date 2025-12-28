"""
MongoDB Camera Repository Implementation
========================================

Concrete implementation of CameraRepository using MongoDB.
"""
from datetime import datetime
from typing import List, Optional
from pymongo import ReturnDocument

from agent.domain.entities.camera import Camera
from agent.domain.repositories.camera_repository import CameraRepository
from agent.infrastructure.database.mongo_client import get_mongo_client


class MongoCameraRepository(CameraRepository):
    """
    MongoDB implementation of CameraRepository.
    
    Handles all camera persistence operations using MongoDB.
    """
    
    COLLECTION_NAME = "cameras"
    
    def __init__(self):
        """Initialize repository with MongoDB client."""
        self._client = get_mongo_client()
        self._collection = self._client.get_collection(self.COLLECTION_NAME)
    
    def _to_entity(self, doc: dict) -> Camera:
        """Convert MongoDB document to Camera entity."""
        return Camera(
            camera_id=doc["camera_id"],
            rtsp_url=doc["rtsp_url"],
            camera_name=doc["camera_name"],
            user_id=doc["user_id"],
            device_id=doc.get("device_id"),
            status=doc.get("status", "active"),
            created_at=doc.get("created_at", datetime.utcnow()),
            updated_at=doc.get("updated_at", datetime.utcnow()),
        )
    
    def _to_document(self, camera: Camera) -> dict:
        """Convert Camera entity to MongoDB document."""
        return {
            "camera_id": camera.camera_id,
            "rtsp_url": camera.rtsp_url,
            "camera_name": camera.camera_name,
            "user_id": camera.user_id,
            "device_id": camera.device_id,
            "status": camera.status,
            "created_at": camera.created_at,
            "updated_at": camera.updated_at,
        }
    
    def create(self, camera: Camera) -> Camera:
        """Create a new camera."""
        camera.created_at = datetime.utcnow()
        camera.updated_at = datetime.utcnow()
        
        doc = self._to_document(camera)
        self._collection.insert_one(doc)
        
        return camera
    
    def update(self, camera: Camera) -> Camera:
        """Update an existing camera."""
        camera.updated_at = datetime.utcnow()
        
        doc = self._to_document(camera)
        result = self._collection.find_one_and_update(
            {"camera_id": camera.camera_id},
            {"$set": {k: v for k, v in doc.items() if k != "created_at"}},
            return_document=ReturnDocument.AFTER,
        )
        
        if not result:
            raise ValueError(f"Camera '{camera.camera_id}' not found")
        
        return self._to_entity(result)
    
    def find_by_id(self, camera_id: str) -> Optional[Camera]:
        """Find a camera by its ID."""
        doc = self._collection.find_one({"camera_id": camera_id})
        if not doc:
            return None
        return self._to_entity(doc)
    
    def find_by_user_id(self, user_id: str) -> List[Camera]:
        """Find all cameras for a user."""
        docs = self._collection.find({"user_id": user_id}).sort("created_at", -1)
        return [self._to_entity(doc) for doc in docs]
    
    def find_all_active(self) -> List[Camera]:
        """Find all active cameras."""
        docs = self._collection.find({"status": {"$ne": "inactive"}}).sort("created_at", -1)
        return [self._to_entity(doc) for doc in docs]
    
    def delete(self, camera_id: str) -> bool:
        """Delete (deactivate) a camera."""
        result = self._collection.update_one(
            {"camera_id": camera_id},
            {
                "$set": {
                    "status": "inactive",
                    "updated_at": datetime.utcnow(),
                }
            }
        )
        return result.matched_count > 0
    
    def exists(self, camera_id: str) -> bool:
        """Check if a camera exists."""
        count = self._collection.count_documents({"camera_id": camera_id}, limit=1)
        return count > 0

