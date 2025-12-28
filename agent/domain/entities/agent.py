"""
Agent Entity
============

Domain entity representing an automation agent/task.
"""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class AgentRule(BaseModel):
    """A single rule/condition for an agent."""
    type: str = Field(..., description="Rule type (e.g., 'class_presence', 'count_at_least')")
    target_class: str = Field(..., alias="class", description="Target class to detect (e.g., 'person', 'car')")
    label: str = Field(..., description="Human-readable label for this rule")
    min_count: Optional[int] = Field(None, description="Minimum count required (for count-based rules)")
    
    class Config:
        allow_population_by_field_name = True


class Agent(BaseModel):
    """Domain entity representing an automation agent/task."""
    agent_id: str = Field(..., description="Unique identifier for the agent")
    task_name: str = Field(..., description="Human-readable name of the task")
    task_type: str = Field(..., description="Type of task (e.g., 'object_detection')")
    camera_id: str = Field(..., description="Camera ID to monitor")
    source_uri: str = Field(..., description="RTSP URL or file path")
    model_ids: List[str] = Field(..., description="List of YOLO model IDs to use")
    fps: int = Field(default=5, description="Frames per second to process")
    run_mode: str = Field(default="continuous", description="Run mode: 'continuous' or 'patrol'")
    rules: List[AgentRule] = Field(default_factory=list, description="List of rules to evaluate")
    status: str = Field(default="pending", description="Current status: 'pending', 'scheduled', 'running', 'completed', 'cancelled'")
    start_at: datetime = Field(..., description="When to start the agent")
    end_at: datetime = Field(..., description="When to stop the agent")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the agent was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the agent was last updated")
    
    def is_active(self) -> bool:
        """Check if agent is in an active state."""
        return self.status in ["pending", "scheduled", "running", None]

