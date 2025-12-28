"""
Agent API Schemas
=================

Pydantic models for agent API request/response validation.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AgentRuleRequest(BaseModel):
    """Request schema for a single agent rule (internal format)."""
    type: str = Field(..., description="Rule type (e.g., 'class_presence', 'count_at_least')")
    class_name: str = Field(..., alias="class", description="Target class to detect (e.g., 'person', 'car')")
    label: Optional[str] = Field(None, description="Human-readable label for this rule")
    min_count: Optional[int] = Field(None, description="Minimum count required (for count-based rules)")
    
    class Config:
        allow_population_by_field_name = True


class WebBackendAgentRule(BaseModel):
    """Rule format from web backend."""
    type: str = Field(..., description="Rule type (e.g., 'detection_rule')")
    class_name: Optional[str] = Field(None, description="Target class to detect (e.g., 'person', 'car')")
    confidence: Optional[float] = Field(None, description="Confidence threshold")
    label: Optional[str] = Field(None, description="Human-readable label")
    min_count: Optional[int] = Field(None, description="Minimum count required")
    # Allow additional fields
    class Config:
        extra = "allow"


class AgentCreateRequest(BaseModel):
    """Request schema for creating/updating an agent (internal format)."""
    agent_id: str = Field(..., description="Unique agent identifier")
    task_name: str = Field(..., description="Human-readable task name")
    task_type: str = Field(..., description="Type of task (e.g., 'object_detection')")
    camera_id: str = Field(..., description="Camera ID to monitor")
    source_uri: str = Field(..., description="RTSP URL or file path")
    model_ids: List[str] = Field(..., description="List of YOLO model IDs")
    fps: int = Field(default=5, description="Frames per second to process")
    run_mode: str = Field(default="continuous", description="Run mode: 'continuous' or 'patrol'")
    rules: List[AgentRuleRequest] = Field(default_factory=list, description="List of rules")
    start_at: datetime = Field(..., description="Start time (ISO format)")
    end_at: datetime = Field(..., description="End time (ISO format)")


class WebBackendAgentRequest(BaseModel):
    """Request schema from web backend (matches vision-backend format)."""
    id: str = Field(..., description="Agent ID")
    name: str = Field(..., description="Agent name")
    camera_id: str = Field(..., description="Camera ID to monitor")
    model: str = Field(..., description="Model name (single model)")
    fps: Optional[int] = Field(5, description="Frames per second to process")
    rules: List[WebBackendAgentRule] = Field(default_factory=list, description="List of rules")
    run_mode: Optional[str] = Field("continuous", description="Run mode: 'continuous' or 'patrol'")
    interval_minutes: Optional[int] = Field(None, description="Patrol interval (minutes)")
    check_duration_seconds: Optional[int] = Field(None, description="Patrol check duration (seconds)")
    start_time: Optional[str] = Field(None, description="Start time (ISO format string)")
    end_time: Optional[str] = Field(None, description="End time (ISO format string)")
    zone: Optional[Dict[str, Any]] = Field(None, description="Zone configuration (if requires_zone is true)")
    requires_zone: bool = Field(False, description="Whether zone is required")
    status: Optional[str] = Field("PENDING", description="Agent status")
    created_at: Optional[str] = Field(None, description="Creation time (ISO format string)")
    owner_user_id: Optional[str] = Field(None, description="Owner user ID")
    
    class Config:
        extra = "allow"  # Allow additional fields from web backend


class AgentResponse(BaseModel):
    """Response schema for agent data."""
    agent_id: str
    task_name: str
    task_type: str
    camera_id: str
    source_uri: str
    model_ids: List[str]
    fps: int
    run_mode: str
    rules: List[Dict[str, Any]]
    status: str
    start_at: datetime
    end_at: datetime
    created_at: datetime
    updated_at: datetime


class AgentDeleteResponse(BaseModel):
    """Response schema for agent deletion."""
    status: str = "success"
    message: str = "Agent marked as cancelled."


class AgentStreamConfigResponse(BaseModel):
    """Response schema for agent stream configuration."""
    signaling_url: str = Field(..., description="WebSocket URL for WebRTC signaling")
    ice_servers: List[Dict[str, Any]] = Field(..., description="ICE servers for NAT traversal")
    agent_id: str = Field(..., description="Agent ID")
    agent_name: str = Field(..., description="Agent name")

