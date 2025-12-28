"""
Agent Repository Interface
===========================

Abstract interface for agent data access operations.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from agent.domain.entities.agent import Agent


class AgentRepository(ABC):
    """Abstract repository interface for agent operations."""
    
    @abstractmethod
    def create(self, agent: Agent) -> Agent:
        """Create a new agent."""
        pass
    
    @abstractmethod
    def update(self, agent: Agent) -> Agent:
        """Update an existing agent."""
        pass
    
    @abstractmethod
    def find_by_id(self, agent_id: str) -> Optional[Agent]:
        """Find an agent by its ID."""
        pass
    
    @abstractmethod
    def find_by_camera_id(self, camera_id: str) -> List[Agent]:
        """Find all agents for a specific camera."""
        pass
    
    @abstractmethod
    def find_all_active(self) -> List[Agent]:
        """Find all active agents."""
        pass
    
    @abstractmethod
    def delete(self, agent_id: str) -> bool:
        """Delete (cancel) an agent by setting status to 'cancelled'."""
        pass
    
    @abstractmethod
    def exists(self, agent_id: str) -> bool:
        """Check if an agent exists."""
        pass

