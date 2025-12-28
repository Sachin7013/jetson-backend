"""
MongoDB Agent Repository Implementation
========================================

Concrete implementation of AgentRepository using MongoDB.
"""
from typing import List, Optional
from datetime import datetime
import logging
from pymongo.collection import Collection
from pymongo import ReturnDocument
from agent.domain.entities.agent import Agent, AgentRule
from agent.domain.repositories.agent_repository import AgentRepository
from agent.infrastructure.database.mongo_client import get_mongo_client

logger = logging.getLogger(__name__)


class MongoAgentRepository(AgentRepository):
    """MongoDB implementation of AgentRepository."""
    
    def __init__(self):
        self._client = get_mongo_client()
        self._collection: Collection = self._client.get_collection("Agents")
    
    def _to_entity(self, doc: dict) -> Agent:
        """Convert MongoDB document to Agent entity."""
        if not doc:
            raise ValueError("Document cannot be empty")
        
        # Convert datetime fields from MongoDB format to ISO strings
        doc = dict(doc)  # Make a copy to avoid modifying original
        
        # Handle datetime fields - MongoDB stores as datetime, convert to ISO string for Pydantic
        if isinstance(doc.get("created_at"), datetime):
            doc["created_at"] = doc["created_at"].isoformat() + "Z"
        if isinstance(doc.get("updated_at"), datetime):
            doc["updated_at"] = doc["updated_at"].isoformat() + "Z"
        if isinstance(doc.get("start_at"), datetime):
            doc["start_at"] = doc["start_at"].isoformat() + "Z"
        if isinstance(doc.get("end_at"), datetime):
            doc["end_at"] = doc["end_at"].isoformat() + "Z"
        if isinstance(doc.get("started_at"), datetime):
            doc["started_at"] = doc["started_at"].isoformat() + "Z"
        if isinstance(doc.get("stopped_at"), datetime):
            doc["stopped_at"] = doc["stopped_at"].isoformat() + "Z"
        
        return Agent(**doc)
    
    def _to_document(self, agent: Agent) -> dict:
        """Convert Agent entity to MongoDB document."""
        # Use model_dump with by_alias=True to ensure rules use 'class' instead of 'target_class'
        agent_dict = agent.model_dump(by_alias=True, exclude_none=False)
        
        # Ensure rules are properly serialized with aliases
        if agent.rules:
            serialized_rules = []
            for rule in agent.rules:
                # Serialize each rule with alias to use 'class' instead of 'target_class'
                try:
                    rule_dict = rule.model_dump(by_alias=True, exclude_none=False)
                    serialized_rules.append(rule_dict)
                    logger.debug(f"Serialized rule: {rule_dict}")
                except Exception as e:
                    logger.error(f"Failed to serialize rule {rule}: {e}")
            agent_dict["rules"] = serialized_rules
            logger.info(f"Serialized {len(serialized_rules)} rules for agent {agent.agent_id}")
        else:
            logger.warning(f"Agent {agent.agent_id} has no rules!")
            agent_dict["rules"] = []
        
        # Convert ISO string dates back to datetime for MongoDB
        if isinstance(agent_dict.get("created_at"), str):
            agent_dict["created_at"] = datetime.fromisoformat(agent_dict["created_at"].replace("Z", "+00:00"))
        if isinstance(agent_dict.get("updated_at"), str):
            agent_dict["updated_at"] = datetime.fromisoformat(agent_dict["updated_at"].replace("Z", "+00:00"))
        if isinstance(agent_dict.get("start_at"), str):
            agent_dict["start_at"] = datetime.fromisoformat(agent_dict["start_at"].replace("Z", "+00:00"))
        if isinstance(agent_dict.get("end_at"), str):
            agent_dict["end_at"] = datetime.fromisoformat(agent_dict["end_at"].replace("Z", "+00:00"))
        
        return agent_dict
    
    def create(self, agent: Agent) -> Agent:
        """Create a new agent in MongoDB."""
        agent_dict = self._to_document(agent)
        self._collection.insert_one(agent_dict)
        return agent
    
    def update(self, agent: Agent) -> Agent:
        """Update an existing agent in MongoDB."""
        agent_dict = self._to_document(agent)
        result = self._collection.find_one_and_update(
            {"agent_id": agent.agent_id},
            {"$set": agent_dict},
            return_document=ReturnDocument.AFTER
        )
        if not result:
            raise ValueError(f"Agent with ID {agent.agent_id} not found for update.")
        return self._to_entity(result)
    
    def find_by_id(self, agent_id: str) -> Optional[Agent]:
        """Find an agent by agent_id."""
        doc = self._collection.find_one({"agent_id": agent_id})
        return self._to_entity(doc) if doc else None
    
    def find_by_camera_id(self, camera_id: str) -> List[Agent]:
        """Find all agents for a specific camera."""
        docs = list(self._collection.find({"camera_id": camera_id}).sort("created_at", -1))
        return [self._to_entity(doc) for doc in docs]
    
    def find_all_active(self) -> List[Agent]:
        """Find all active agents (pending, scheduled, running, or None status)."""
        docs = list(self._collection.find({"status": {"$in": ["pending", "scheduled", "running", None]}}).sort("created_at", 1))
        return [self._to_entity(doc) for doc in docs]
    
    def delete(self, agent_id: str) -> bool:
        """Delete (cancel) an agent by setting status to 'cancelled'."""
        result = self._collection.update_one(
            {"agent_id": agent_id},
            {"$set": {"status": "cancelled", "updated_at": datetime.utcnow()}}
        )
        return result.matched_count > 0
    
    def exists(self, agent_id: str) -> bool:
        """Check if an agent exists."""
        count = self._collection.count_documents({"agent_id": agent_id}, limit=1)
        return count > 0

