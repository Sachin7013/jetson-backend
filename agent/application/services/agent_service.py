"""
Agent Service
=============

Application service for agent-related operations.
Orchestrates use cases and coordinates business logic.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from agent.domain.entities.agent import Agent
from agent.domain.repositories.agent_repository import AgentRepository
from agent.domain.repositories.camera_repository import CameraRepository
from agent.application.use_cases.register_agent import RegisterAgentUseCase
from agent.application.use_cases.get_agent_stream_config import GetAgentStreamConfigUseCase


class AgentService:
    """Application service for agent operations."""
    
    def __init__(self, repository: AgentRepository, camera_repository: CameraRepository):
        self._repository = repository
        self._camera_repository = camera_repository
        self._register_use_case = RegisterAgentUseCase(repository)
        self._get_stream_config_use_case = GetAgentStreamConfigUseCase(repository, camera_repository)
    
    def register_agent(
        self,
        agent_id: str,
        task_name: str,
        task_type: str,
        camera_id: str,
        source_uri: str,
        model_ids: List[str],
        fps: int,
        run_mode: str,
        rules: List[dict],
        start_at: datetime,
        end_at: datetime,
    ) -> Agent:
        """Register or update an agent."""
        return self._register_use_case.execute(
            agent_id, task_name, task_type, camera_id, source_uri,
            model_ids, fps, run_mode, rules, start_at, end_at
        )
    
    def get_agent_by_id(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by its ID."""
        return self._repository.find_by_id(agent_id)
    
    def list_agents(self, camera_id: Optional[str] = None, status: Optional[str] = None) -> List[Agent]:
        """List agents, optionally filtered by camera_id or status."""
        if camera_id:
            return self._repository.find_by_camera_id(camera_id)
        elif status:
            # Filter active agents by status
            all_active = self._repository.find_all_active()
            return [a for a in all_active if a.status == status]
        else:
            return self._repository.find_all_active()
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove (cancel) an agent."""
        return self._repository.delete(agent_id)
    
    def get_agent_stream_config(self, agent_id: str) -> Dict[str, Any]:
        """Get WebRTC configuration for agent stream."""
        return self._get_stream_config_use_case.execute(agent_id)

