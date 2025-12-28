"""
Agent API Routes
================

FastAPI routes for agent management.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
from fastapi import APIRouter, Depends, HTTPException, status, Body
from agent.api.schemas.agent_schemas import (
    AgentCreateRequest, AgentResponse, AgentDeleteResponse, AgentStreamConfigResponse,
    WebBackendAgentRequest
)
from agent.application.services.agent_service import AgentService
from agent.api.dependencies.container import get_agent_service, get_camera_repository
from agent.domain.repositories.camera_repository import CameraRepository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["agents"])


def _convert_web_backend_to_internal(
    web_request: WebBackendAgentRequest,
    camera_repository: CameraRepository
) -> dict:
    """
    Convert web backend request format to internal format.
    
    Args:
        web_request: Request from web backend
        camera_repository: Repository to get camera RTSP URL
        
    Returns:
        Dictionary with internal format fields
    """
    # Get camera to retrieve RTSP URL (source_uri)
    camera = camera_repository.find_by_id(web_request.camera_id)
    if not camera:
        raise ValueError(f"Camera '{web_request.camera_id}' not found. Please register the camera first.")
    
    # Convert rules from web backend format to internal format
    internal_rules = []
    for rule in web_request.rules:
        # Handle both Pydantic model and dict
        if isinstance(rule, dict):
            rule_type = rule.get("type", "detection_rule")
            class_name = rule.get("class_name") or rule.get("class") or rule.get("target_class", "")
            label = rule.get("label")
            min_count = rule.get("min_count")
        else:
            rule_type = rule.type
            class_name = rule.class_name or getattr(rule, 'class', '') or getattr(rule, 'target_class', '')
            label = rule.label
            min_count = rule.min_count

        # Skip rules without class_name
        if not class_name:
            continue
        
        # Web backend uses 'detection_rule' type, we use 'class_presence'
        if rule_type == "detection_rule":
            rule_type = "class_presence"
        elif rule_type not in ["class_presence", "count_at_least", "class_count"]:
            rule_type = "class_presence"  # Default to class_presence
        
        # Web backend uses 'class_name', internal uses 'class' (with alias)
        # AgentRule expects 'class' key (alias for 'target_class')
        internal_rule = {
            "type": rule_type,
            "class": class_name,  # Use 'class' key (alias) for AgentRule
            "label": label or f"{class_name} detection",  # Use provided label or generate
        }
        
        # Add min_count if present and valid (not empty string)
        if min_count is not None and min_count != "":
            try:
                internal_rule["min_count"] = int(min_count) if isinstance(min_count, str) else min_count
            except (ValueError, TypeError):
                # Skip invalid min_count
                pass
        
        internal_rules.append(internal_rule)
    
    # Parse datetime strings
    start_at = datetime.utcnow()
    end_at = datetime.utcnow()
    
    if web_request.start_time:
        try:
            start_at = datetime.fromisoformat(web_request.start_time.replace("Z", "+00:00"))
        except Exception:
            # Fallback: try parsing without timezone
            try:
                start_at = datetime.fromisoformat(web_request.start_time)
            except Exception:
                pass
    
    if web_request.end_time:
        try:
            end_at = datetime.fromisoformat(web_request.end_time.replace("Z", "+00:00"))
        except Exception:
            try:
                end_at = datetime.fromisoformat(web_request.end_time)
            except Exception:
                pass
    
    # Convert model (single string) to model_ids (list)
    model_ids = [web_request.model] if web_request.model else []
    
    return {
        "agent_id": web_request.id,
        "task_name": web_request.name,
        "task_type": "object_detection",  # Default task type
        "camera_id": web_request.camera_id,
        "source_uri": camera.rtsp_url,  # Get from camera
        "model_ids": model_ids,
        "fps": web_request.fps or 5,
        "run_mode": web_request.run_mode or "continuous",
        "rules": internal_rules,
        "start_at": start_at,
        "end_at": end_at,
    }


@router.post(
    "/agents",
    response_model=AgentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register or update an agent",
    description="Register a new agent or update an existing one. The runner will automatically start processing. Accepts both internal format and web backend format."
)
async def register_agent_route(
    request_data: Dict[str, Any] = Body(...),
    agent_service: AgentService = Depends(get_agent_service),
    camera_repository: CameraRepository = Depends(get_camera_repository)
) -> AgentResponse:
    """
    Register or update an agent.
    
    This endpoint accepts two formats:
    1. Web backend format (from vision-backend): Uses 'id', 'name', 'model', 'class_name', 'start_time', 'end_time'
    2. Internal format: Uses 'agent_id', 'task_name', 'model_ids', 'class', 'start_at', 'end_at'
    
    The agent will be saved to MongoDB and the runner will automatically pick it up on the next poll.
    """
    try:
        # Detect format by checking for 'id' field (web backend) or 'agent_id' (internal)
        if "id" in request_data and "name" in request_data:
            # Web backend format - parse and convert
            try:
                web_request = WebBackendAgentRequest(**request_data)
                internal_data = _convert_web_backend_to_internal(web_request, camera_repository)
                agent = agent_service.register_agent(**internal_data)
            except Exception as e:
                raise ValueError(f"Invalid web backend format: {e}")
        elif "agent_id" in request_data and "task_name" in request_data:
            # Internal format - parse and use directly
            try:
                # Extract rules before parsing to handle them manually if needed
                raw_rules = request_data.get("rules", [])
                logger.info(f"Received {len(raw_rules)} rules in request: {raw_rules}")
                
                # Try to parse the full request
                try:
                    internal_request = AgentCreateRequest(**request_data)
                    # Rules were successfully parsed as AgentRuleRequest objects
                    # Ensure labels are set for all rules and use 'class' alias
                    rules_to_store = []
                    for rule in internal_request.rules:
                        # Use dict() with by_alias=True to preserve 'class' key instead of 'class_name'
                        try:
                            rule_dict = rule.dict(by_alias=True) if hasattr(rule, 'dict') else rule.model_dump(by_alias=True)
                        except:
                            # Fallback for Pydantic v2
                            rule_dict = rule.model_dump(by_alias=True) if hasattr(rule, 'model_dump') else dict(rule)
                        
                        # Ensure label is set
                        if not rule_dict.get("label"):
                            class_name = rule_dict.get("class", "") or rule_dict.get("class_name", "")
                            rule_dict["label"] = f"{class_name} detection" if class_name else "Detection rule"
                        
                        # Ensure 'class' key exists (not 'class_name')
                        if "class_name" in rule_dict and "class" not in rule_dict:
                            rule_dict["class"] = rule_dict.pop("class_name")
                        
                        rules_to_store.append(rule_dict)
                    logger.info(f"Successfully parsed {len(rules_to_store)} rules: {rules_to_store}")
                except Exception as parse_error:
                    # If parsing fails, try to handle rules manually
                    # This handles cases where rules don't match AgentRuleRequest schema exactly
                    logger.warning(f"Failed to parse rules with AgentCreateRequest: {parse_error}. Attempting manual rule conversion.")
                    
                    # Try parsing without rules first
                    request_without_rules = {k: v for k, v in request_data.items() if k != "rules"}
                    internal_request = AgentCreateRequest(**request_without_rules)
                    
                    # Manually convert rules
                    rules_to_store = []
                    for rule in raw_rules:
                        if isinstance(rule, dict):
                            # Ensure required fields exist
                            rule_type = rule.get("type", "class_presence")
                            class_name = rule.get("class_name") or rule.get("class", "")
                            label = rule.get("label") or (f"{class_name} detection" if class_name else "Detection rule")
                            
                            if class_name:  # Only add rules with a class
                                converted_rule = {
                                    "type": rule_type,
                                    "class": class_name,
                                    "label": label,
                                }
                                # Handle min_count: normalize empty string to None
                                if "min_count" in rule:
                                    min_count_val = rule["min_count"]
                                    if min_count_val == "" or min_count_val is None:
                                        # Skip adding min_count if empty or None
                                        pass
                                    else:
                                        try:
                                            converted_rule["min_count"] = int(min_count_val)
                                        except (ValueError, TypeError):
                                            # Skip invalid min_count
                                            pass
                                rules_to_store.append(converted_rule)
                                logger.debug(f"Converted rule: {converted_rule}")
                            else:
                                logger.warning(f"Skipping rule without class_name: {rule}")
                        elif hasattr(rule, 'dict'):
                            # Pydantic model - use by_alias to preserve 'class' key
                            rule_dict = rule.dict(by_alias=True) if hasattr(rule, 'dict') else rule.model_dump(by_alias=True)
                            rules_to_store.append(rule_dict)
                        else:
                            # Already a dict or other format
                            rules_to_store.append(rule)
                    
                    logger.info(f"Manually converted {len(rules_to_store)} rules: {rules_to_store}")
                
                agent = agent_service.register_agent(
                    agent_id=internal_request.agent_id,
                    task_name=internal_request.task_name,
                    task_type=internal_request.task_type,
                    camera_id=internal_request.camera_id,
                    source_uri=internal_request.source_uri,
                    model_ids=internal_request.model_ids,
                    fps=internal_request.fps,
                    run_mode=internal_request.run_mode,
                    rules=rules_to_store,
                    start_at=internal_request.start_at,
                    end_at=internal_request.end_at,
                )
                logger.info(f"Agent {agent.agent_id} registered with {len(agent.rules)} rules")
            except Exception as e:
                logger.error(f"Error registering agent: {e}", exc_info=True)
                raise ValueError(f"Invalid internal format: {e}")
        else:
            raise ValueError("Request must be in either web backend format (with 'id' and 'name') or internal format (with 'agent_id' and 'task_name')")
        
        return AgentResponse(
            agent_id=agent.agent_id,
            task_name=agent.task_name,
            task_type=agent.task_type,
            camera_id=agent.camera_id,
            source_uri=agent.source_uri,
            model_ids=agent.model_ids,
            fps=agent.fps,
            run_mode=agent.run_mode,
            rules=[rule.dict(by_alias=True) if hasattr(rule, 'dict') else rule.model_dump(by_alias=True) for rule in agent.rules],
            status=agent.status,
            start_at=agent.start_at,
            end_at=agent.end_at,
            created_at=agent.created_at,
            updated_at=agent.updated_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to register agent: {e}")


@router.get(
    "/agents",
    response_model=List[AgentResponse],
    summary="List all agents",
    description="Get list of all registered agents, optionally filtered by camera_id or status."
)
async def list_agents_route(
    camera_id: Optional[str] = None,
    status_filter: Optional[str] = None,
    agent_service: AgentService = Depends(get_agent_service)
) -> List[AgentResponse]:
    """List all agents, optionally filtered by camera_id or status."""
    agents = agent_service.list_agents(camera_id, status_filter)
    return [
        AgentResponse(
            agent_id=agent.agent_id,
            task_name=agent.task_name,
            task_type=agent.task_type,
            camera_id=agent.camera_id,
            source_uri=agent.source_uri,
            model_ids=agent.model_ids,
            fps=agent.fps,
            run_mode=agent.run_mode,
            rules=[rule.dict(by_alias=True) if hasattr(rule, 'dict') else rule.model_dump(by_alias=True) for rule in agent.rules],
            status=agent.status,
            start_at=agent.start_at,
            end_at=agent.end_at,
            created_at=agent.created_at,
            updated_at=agent.updated_at,
        )
        for agent in agents
    ]


@router.get(
    "/agents/{agent_id}",
    response_model=AgentResponse,
    summary="Get agent by ID",
    description="Get details of a specific agent."
)
async def get_agent_route(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service)
) -> AgentResponse:
    """Get a specific agent by its ID."""
    agent = agent_service.get_agent_by_id(agent_id)
    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found")
    return AgentResponse(
        agent_id=agent.agent_id,
        task_name=agent.task_name,
        task_type=agent.task_type,
        camera_id=agent.camera_id,
        source_uri=agent.source_uri,
        model_ids=agent.model_ids,
        fps=agent.fps,
        run_mode=agent.run_mode,
        rules=[rule.dict() for rule in agent.rules],
        status=agent.status,
        start_at=agent.start_at,
        end_at=agent.end_at,
        created_at=agent.created_at,
        updated_at=agent.updated_at,
    )


@router.delete(
    "/agents/{agent_id}",
    response_model=AgentDeleteResponse,
    summary="Remove an agent",
    description="Remove an agent by setting its status to 'cancelled'. The runner will automatically stop processing."
)
async def remove_agent_route(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service)
) -> AgentDeleteResponse:
    """Remove (cancel) an agent."""
    if not agent_service.remove_agent(agent_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found")
    return AgentDeleteResponse(message=f"Agent '{agent_id}' has been marked as cancelled. Worker will stop on next poll.")


@router.get(
    "/agents/{agent_id}/stream-config",
    response_model=AgentStreamConfigResponse,
    summary="Get agent stream configuration",
    description="Get WebRTC configuration for viewing agent-specific live stream (with bounding boxes)."
)
async def get_agent_stream_config_route(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service)
) -> AgentStreamConfigResponse:
    """Get WebRTC configuration for agent stream."""
    try:
        config = agent_service.get_agent_stream_config(agent_id)
        return AgentStreamConfigResponse(**config)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get agent stream config: {e}")

