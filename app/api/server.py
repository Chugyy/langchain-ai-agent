from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid
import json
from typing import Dict, List, Optional, Any, Tuple

from app.agents.agent import AgentFactory
from app.utils.settings import get_settings
from app.utils.logging import get_logger
from app.utils.auth import APIKeyManager
from app.api.auth import get_api_key, verify_admin_key
from app.api.media import router as media_router
from app.schemas.media import MediaReference, MediaInfo
from app.tools.media_tools import fetch_media_from_url, media_registry

logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title="AI Conversational Agent API",
    description="API for interacting with an AI conversational agent",
    version="0.1.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the agent")
    session_id: Optional[str] = Field(None, description="Session ID for continuing a conversation")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional configuration overrides")
    media: Optional[List[MediaReference]] = Field(None, description="List of media to include with the message")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Agent's response message")
    session_id: str = Field(..., description="Session ID for the conversation")
    thinking: Optional[str] = Field(None, description="Agent's thinking process (if verbose mode)")
    media: Optional[List[MediaInfo]] = Field(None, description="List of media processed in this conversation")

class SessionConfigUpdate(BaseModel):
    temperature: Optional[float] = Field(None, description="LLM temperature setting")
    memory_type: Optional[str] = Field(None, description="Memory type (buffer/summary)")
    model_name: Optional[str] = Field(None, description="LLM model name")
    
class SessionResponse(BaseModel):
    session_id: str = Field(..., description="Session ID")
    messages: List[Dict[str, str]] = Field(..., description="List of messages in the session")
    config: Dict[str, Any] = Field(..., description="Session configuration")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")

class APIKeyRequest(BaseModel):
    scopes: List[str] = Field(default=["chat"], description="List of permission scopes")
    expires_in_days: int = Field(default=30, description="Number of days until key expiration")
    rate_limit: int = Field(default=100, description="Number of requests allowed per day")
    
class APIKeyResponse(BaseModel):
    key_id: str = Field(..., description="API key ID")
    api_key: str = Field(..., description="Full API key (only shown once)")
    scopes: List[str] = Field(..., description="Granted permission scopes")
    expires_at: str = Field(..., description="Expiration timestamp")
    rate_limit: int = Field(..., description="Rate limit (requests per day)")

# Modèles pour le débogage
class DebuggingResponse(BaseModel):
    """Réponse de débogage avec l'état complet de l'agent."""
    agent_state: Dict[str, Any] = Field(..., description="État de l'agent")
    session_data: Optional[SessionResponse] = Field(None, description="Données de session")
    last_steps: Optional[List[Dict[str, Any]]] = Field(None, description="Dernières étapes intermédiaires")

# Initialize agent factory once at startup
agent_factory = AgentFactory()

# Include the media router
app.include_router(media_router)

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, 
    background_tasks: BackgroundTasks,
    api_auth: Tuple[str, Dict] = Depends(get_api_key)
):
    """
    Process a chat message and return the agent's response.
    Creates a new session if session_id is not provided.
    
    Can include media references that will be processed along with the message.
    """
    try:
        # Extract key_id from auth
        key_id, _ = api_auth
        
        # Log incoming request
        logger.info(f"New chat message received", extra={
            "session_id": request.session_id,
            "prompt_length": len(request.message),
            "key_id": key_id,
            "has_media": request.media is not None and len(request.media) > 0
        })
        
        # Create or get existing session
        session_id = request.session_id or str(uuid.uuid4())
        
        # Process media if provided
        media_infos = []
        if request.media:
            logger.info(f"Processing {len(request.media)} media references", extra={"session_id": session_id})
            
            # Fetch all media 
            for media_ref in request.media:
                try:
                    metadata = fetch_media_from_url(str(media_ref.url), session_id)
                    if metadata:
                        # Les champs optionnels seront utilisés seulement s'ils sont fournis
                        if hasattr(media_ref, 'reference_id') and media_ref.reference_id:
                            metadata.reference_id = media_ref.reference_id
                        if hasattr(media_ref, 'title') and media_ref.title:
                            metadata.title = media_ref.title
                        if hasattr(media_ref, 'description') and media_ref.description:
                            metadata.description = media_ref.description
                            
                        # Convert to MediaInfo
                        media_info = MediaInfo(
                            media_id=metadata.media_id,
                            original_url=metadata.original_url,
                            media_type=metadata.media_type,
                            content_type=metadata.content_type,
                            size=metadata.size,
                            reference_id=getattr(metadata, "reference_id", None),
                            download_date=metadata.download_date,
                            processed=False,
                            title=getattr(metadata, "title", None),
                            description=getattr(metadata, "description", None)
                        )
                        media_infos.append(media_info)
                except Exception as e:
                    logger.error(f"Error processing media {media_ref.url}: {str(e)}", exc_info=True)
                    # Continue with other media
        
        # Modify message to include media references if any
        message = request.message
        if media_infos:
            # Add media context to the message
            media_context = "\n\nContexte média:\n"
            for i, media in enumerate(media_infos):
                ref_id = media.reference_id or f"media{i+1}"
                media_context += f"- {ref_id}: {media.media_type} ({media.content_type}), ID: {media.media_id}\n"
            
            # Ajouter une suggestion d'utilisation de l'outil extract_media_content
            media_context += "\nPour analyser ces médias, vous pouvez utiliser l'outil extract_media_content avec l'ID du média."
            
            # Append to original message
            message += media_context
        
        # Get agent instance for this session
        agent = agent_factory.get_agent(session_id, request.config)
        
        # Process message with agent
        logger.debug(f"Processing message with agent", extra={"session_id": session_id})
        response = agent.process_message(message)
        
        # Log completion info
        logger.info(f"Chat response generated", extra={
            "session_id": session_id,
            "response_length": len(response),
            "media_count": len(media_infos)
        })
        
        # Schedule session cleanup in background if needed
        if not request.session_id:  # New session
            logger.info(f"New session created", extra={"session_id": session_id})
            
        return ChatResponse(
            response=response,
            session_id=session_id,
            thinking=agent.get_thinking() if settings.debug_mode else None,
            media=media_infos if media_infos else None
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    api_auth: Tuple[str, Dict] = Depends(get_api_key)
):
    """
    Retrieve a session and its message history by session_id.
    """
    try:
        # Extract key_id from auth
        key_id, _ = api_auth
        
        logger.debug(f"Session retrieval request", extra={
            "session_id": session_id,
            "key_id": key_id
        })
        
        # Check if session exists and retrieve it
        if not agent_factory.session_exists(session_id):
            logger.warning(f"Session not found", extra={"session_id": session_id})
            raise HTTPException(status_code=404, detail="Session not found")
            
        # Get session data including messages and config
        session_data = agent_factory.get_session_data(session_id)
        
        logger.info(f"Session retrieved", extra={
            "session_id": session_id,
            "message_count": len(session_data["messages"])
        })
        
        return SessionResponse(**session_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session: {str(e)}", extra={"session_id": session_id}, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.patch("/sessions/{session_id}/config", response_model=Dict[str, Any])
async def update_session_config(
    session_id: str, 
    config_update: SessionConfigUpdate,
    api_auth: Tuple[str, Dict] = Depends(get_api_key)
):
    """
    Update configuration for a specific session.
    """
    try:
        # Extract key_id from auth
        key_id, _ = api_auth
        
        logger.debug(f"Session config update request", extra={
            "session_id": session_id,
            "key_id": key_id
        })
        
        # Check if session exists
        if not agent_factory.session_exists(session_id):
            logger.warning(f"Session not found for config update", extra={"session_id": session_id})
            raise HTTPException(status_code=404, detail="Session not found")
            
        # Convert Pydantic model to dict, removing None values
        config_dict = {k: v for k, v in config_update.dict().items() if v is not None}
        
        if not config_dict:
            logger.warning("No valid configuration changes provided")
            return {"message": "No changes made - empty update"}
            
        # Update session config
        updated_config = agent_factory.update_session_config(session_id, config_dict)
        
        logger.info(f"Session config updated", extra={
            "session_id": session_id,
            "updates": config_dict
        })
        
        return {"message": "Config updated successfully", "config": updated_config}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session config: {str(e)}", extra={"session_id": session_id}, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/auth/keys", response_model=APIKeyResponse)
async def create_api_key(
    request: APIKeyRequest,
    _: bool = Depends(verify_admin_key)
):
    """
    Create a new API key with the specified scopes and rate limit.
    Requires admin authentication.
    """
    try:
        # Initialize key manager
        key_manager = APIKeyManager()
        
        # Create new key with specified scopes and rate limit
        key_id, api_key, expires_at = key_manager.create_key(
            scopes=request.scopes,
            expires_in_days=request.expires_in_days,
            rate_limit=request.rate_limit
        )
        
        logger.info(f"New API key created", extra={
            "key_id": key_id,
            "scopes": request.scopes,
            "expires_in_days": request.expires_in_days,
            "rate_limit": request.rate_limit
        })
        
        return APIKeyResponse(
            key_id=key_id,
            api_key=api_key,  # Full key only returned once
            scopes=request.scopes,
            expires_at=expires_at,
            rate_limit=request.rate_limit
        )
        
    except Exception as e:
        logger.error(f"Error creating API key: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint to verify API is running.
    """
    return {
        "status": "healthy"
    }

# ============== ENDPOINTS DE DÉBOGAGE ===============
# Ces endpoints nécessitent l'authentification d'administrateur (admin key)
# et sont destinés au débogage et à l'analyse du comportement de l'agent

@app.get("/debug/sessions/{session_id}", response_model=DebuggingResponse)
async def debug_agent(
    session_id: str,
    _: bool = Depends(verify_admin_key)
):
    """
    Récupère l'état complet d'un agent pour débogage.
    Cette route est destinée aux développeurs et nécessite une clé admin.
    """
    try:
        logger.info(f"Demande de débogage pour la session {session_id}")
        
        # Vérifie si la session existe
        if not agent_factory.session_exists(session_id):
            logger.warning(f"Session {session_id} non trouvée")
            raise HTTPException(status_code=404, detail="Session non trouvée")
        
        # Récupère l'agent
        agent = agent_factory.get_agent(session_id=session_id)
        
        # Récupère l'état complet de l'agent
        agent_state = agent.dump_agent_state()
        
        # Récupère les données de session
        session_data = agent_factory.get_session_data(session_id)
        
        # Récupère les dernières étapes intermédiaires si disponibles
        last_steps = None
        if hasattr(agent, "last_intermediate_steps"):
            last_steps = []
            for i, (action, observation) in enumerate(agent.last_intermediate_steps):
                tool = getattr(action, "tool", "inconnu")
                tool_input = getattr(action, "tool_input", {})
                
                last_steps.append({
                    "step": i+1,
                    "tool": tool,
                    "tool_input": tool_input,
                    "observation": str(observation)
                })
        
        return DebuggingResponse(
            agent_state=agent_state,
            session_data=session_data,
            last_steps=last_steps
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du débogage de l'agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.put("/debug/sessions/{session_id}/verbose/{enable}")
async def set_verbose_mode(
    session_id: str, 
    enable: bool,
    _: bool = Depends(verify_admin_key)
):
    """
    Active ou désactive le mode verbeux pour une session.
    Cette route est destinée aux développeurs et nécessite une clé admin.
    """
    try:
        logger.info(f"Modification du mode verbeux pour la session {session_id}: {enable}")
        
        # Vérifie si la session existe
        if not agent_factory.session_exists(session_id):
            logger.warning(f"Session {session_id} non trouvée")
            raise HTTPException(status_code=404, detail="Session non trouvée")
        
        # Récupère l'agent
        agent = agent_factory.get_agent(session_id=session_id)
        
        # Active ou désactive le mode verbeux
        if hasattr(agent, "agent"):
            agent.agent.verbose = enable
            return {"status": "success", "message": f"Mode verbeux défini à {enable}"}
        else:
            return {"status": "error", "message": "Agent non compatible avec le mode verbeux"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la modification du mode verbeux: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.get("/debug/sessions/{session_id}/prompt")
async def dump_agent_prompt(
    session_id: str,
    _: bool = Depends(verify_admin_key)
):
    """
    Affiche le prompt template complet utilisé par l'agent.
    Cette route est destinée aux développeurs et nécessite une clé admin.
    """
    try:
        logger.info(f"Demande de dump du prompt pour la session {session_id}")
        
        # Vérifie si la session existe
        if not agent_factory.session_exists(session_id):
            logger.warning(f"Session {session_id} non trouvée")
            raise HTTPException(status_code=404, detail="Session non trouvée")
        
        # Récupère l'agent
        agent = agent_factory.get_agent(session_id=session_id)
        
        # Récupère le prompt template
        prompt_data = {}
        
        try:
            if hasattr(agent, "agent") and hasattr(agent.agent, "llm_chain"):
                prompt = agent.agent.llm_chain.prompt
                prompt_data = {
                    "template": getattr(prompt, "template", "Non disponible"),
                    "input_variables": getattr(prompt, "input_variables", []),
                    "template_format": getattr(prompt, "template_format", "Non disponible"),
                    "validate_template": getattr(prompt, "validate_template", True)
                }
                
                # Tente de récupérer les instructions de l'agent
                if hasattr(agent.agent, "agent"):
                    agent_obj = agent.agent.agent
                    prompt_data["prefix"] = getattr(agent_obj, "prefix", "Non disponible")
                    prompt_data["suffix"] = getattr(agent_obj, "suffix", "Non disponible")
                    prompt_data["format_instructions"] = getattr(agent_obj, "format_instructions", "Non disponible")
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du prompt: {str(e)}")
            prompt_data["error"] = str(e)
        
        return prompt_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du dump du prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.put("/debug/monitor/http/{session_id}/{enable}")
async def toggle_http_monitoring(
    session_id: str,
    enable: bool,
    _: bool = Depends(verify_admin_key)
):
    """
    Active ou désactive le monitoring HTTP pour une session spécifique.
    Permet de visualiser les requêtes et réponses envoyées à l'API OpenAI.
    Cette route est destinée aux développeurs et nécessite une clé admin.
    """
    try:
        # Récupère l'agent
        if not agent_factory.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session non trouvée")
            
        agent = agent_factory.get_agent(session_id=session_id)
        
        # Recréer l'agent avec le mode verbeux activé
        new_config = agent.config.copy() if hasattr(agent, "config") else {}
        
        # Créer un nouvel agent avec la configuration existante plus le mode verbeux
        agent = AgentFactory.create_agent(
            llm_settings=get_settings().llm,
            memory_settings=get_settings().memory,
            tools_settings=get_settings().tools,
            session_id=session_id,
            verbose=enable  # Active ou désactive le monitoring HTTP
        )
        
        return {
            "status": "success", 
            "monitoring": "enabled" if enable else "disabled",
            "message": "Monitoring HTTP " + ("activé" if enable else "désactivé") + " pour la session " + session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'activation du monitoring HTTP: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.get("/debug/sessions/{session_id}/http-logs")
async def get_http_logs(
    session_id: str,
    _: bool = Depends(verify_admin_key)
):
    """
    Récupère les logs HTTP pour une session spécifique si le monitoring est activé.
    Cette route est destinée aux développeurs et nécessite une clé admin.
    """
    try:
        # Cette fonctionnalité nécessite d'avoir implémenté un système de capture des logs HTTP
        # Pour l'instant, on renvoie un message indiquant que la fonctionnalité est en cours de développement
        return {
            "status": "warning",
            "message": "Cette fonctionnalité est disponible uniquement dans les logs du serveur pour le moment. Vérifiez les logs avec le niveau DEBUG activé."
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des logs HTTP: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}") 