"""
Register Agent Use Case
========================

Use case for registering or updating an agent configuration.
"""
from datetime import datetime
from typing import List
from agent.domain.entities.agent import Agent, AgentRule
from agent.domain.repositories.agent_repository import AgentRepository


class RegisterAgentUseCase:
    """Use case for registering or updating an agent."""
    
    def __init__(self, repository: AgentRepository):
        self._repository = repository
    
    def execute(
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
        """
        Register or update an agent configuration.
        
        If agent_id exists, updates the existing agent.
        Otherwise, creates a new agent.
        
        Args:
            agent_id: Unique identifier for the agent
            task_name: Human-readable name of the task
            task_type: Type of task (e.g., 'object_detection')
            camera_id: Camera ID to monitor
            source_uri: RTSP URL or file path
            model_ids: List of YOLO model IDs to use
            fps: Frames per second to process
            run_mode: Run mode ('continuous' or 'patrol')
            rules: List of rule dictionaries
            start_at: When to start the agent
            end_at: When to stop the agent
        
        Returns:
            Agent entity
        """
        existing_agent = self._repository.find_by_id(agent_id)
        now = datetime.utcnow()
        
        # Convert rules dict to AgentRule objects
        # Rules come in with 'class' key, which maps to 'target_class' via alias
        import logging
        logger = logging.getLogger(__name__)
        agent_rules = []
        for rule in rules:
            try:
                # Ensure rule has required fields
                if not isinstance(rule, dict):
                    logger.warning(f"Skipping non-dict rule: {rule}")
                    continue
                
                # Normalize rule dict to ensure it has 'class' key (not 'target_class')
                normalized_rule = dict(rule)
                
                # Handle 'target_class' -> 'class' conversion (AgentRule uses alias 'class')
                if "target_class" in normalized_rule and "class" not in normalized_rule:
                    normalized_rule["class"] = normalized_rule.pop("target_class")
                elif "class_name" in normalized_rule and "class" not in normalized_rule:
                    normalized_rule["class"] = normalized_rule.pop("class_name")
                
                # Ensure 'class' key exists
                if "class" not in normalized_rule:
                    logger.warning(f"Skipping rule without 'class' field: {rule}")
                    continue
                
                # Normalize min_count: convert empty string to None
                if "min_count" in normalized_rule:
                    min_count_val = normalized_rule["min_count"]
                    if min_count_val == "" or min_count_val is None:
                        normalized_rule["min_count"] = None
                    else:
                        try:
                            normalized_rule["min_count"] = int(min_count_val)
                        except (ValueError, TypeError):
                            normalized_rule["min_count"] = None
                
                # Create AgentRule - 'class' will map to 'target_class' via alias
                agent_rule = AgentRule(**normalized_rule)
                agent_rules.append(agent_rule)
                logger.debug(f"Successfully created AgentRule: {agent_rule}")
            except Exception as e:
                logger.error(f"Failed to create AgentRule from {rule}: {e}", exc_info=True)
                # Skip invalid rules but continue processing
                continue
        
        if existing_agent:
            # Update existing agent
            existing_agent.task_name = task_name
            existing_agent.task_type = task_type
            existing_agent.camera_id = camera_id
            existing_agent.source_uri = source_uri
            existing_agent.model_ids = model_ids
            existing_agent.fps = fps
            existing_agent.run_mode = run_mode
            existing_agent.rules = agent_rules
            existing_agent.start_at = start_at
            existing_agent.end_at = end_at
            existing_agent.status = "pending"  # Reset to pending when updated
            existing_agent.updated_at = now
            return self._repository.update(existing_agent)
        else:
            # Create new agent
            new_agent = Agent(
                agent_id=agent_id,
                task_name=task_name,
                task_type=task_type,
                camera_id=camera_id,
                source_uri=source_uri,
                model_ids=model_ids,
                fps=fps,
                run_mode=run_mode,
                rules=agent_rules,
                status="pending",
                start_at=start_at,
                end_at=end_at,
                created_at=now,
                updated_at=now,
            )
            return self._repository.create(new_agent)

