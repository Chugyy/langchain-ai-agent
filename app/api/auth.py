from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Tuple, Dict, Optional

from app.utils.auth import APIKeyManager
from app.utils.logging import get_logger
from app.utils.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()

# Init security
security = HTTPBearer(auto_error=False)

def get_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Tuple[str, Dict]:
    """
    Vérifie et retourne la clé API.
    
    Returns:
        Tuple contenant key_id et key_data
    """
    # Mode développement - court-circuite l'authentification
    if settings.environment == "development" and settings.debug_mode:
        logger.debug("Mode développement activé, authentification court-circuitée")
        return ("dev-key", {"scopes": ["chat", "sessions"]})
    
    # Mode production standard - vérifie la clé API
    if not credentials:
        logger.warning("Tentative d'accès sans authentification")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    api_key = credentials.credentials
    if not api_key.startswith(settings.security.api_key_prefix):
        logger.warning("Format de clé API invalide")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    key_id, key_data = APIKeyManager.validate_key(api_key)
    if not key_id:
        logger.warning("Clé API invalide")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    logger.info(f"Authentification réussie avec la clé {key_id}")
    return key_id, key_data

def verify_admin_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> bool:
    """
    Vérifie que la clé API est une clé d'administration.
    
    Returns:
        True si valide
    """
    # Court-circuiter l'authentification en mode développement et debug
    if settings.environment == "development" and settings.debug_mode:
        logger.warning("Mode développement/debug: authentification admin court-circuitée")
        return True
    
    # S'assurer que les credentials existent
    if not credentials:
        logger.warning("Tentative d'accès admin sans authentification")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    api_key = credentials.credentials
    
    # Test direct contre la clé admin en environnement de développement
    if settings.environment == "development" and api_key == settings.admin_api_key:
        logger.info("Accès admin autorisé en mode développement")
        return True
        
    # Vérification dans la base en production
    key_id, key_data = APIKeyManager.validate_key(api_key)
    if not key_id or "admin" not in key_data.get("scopes", []):
        logger.warning("Tentative d'accès admin non autorisée")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    
    logger.info(f"Accès admin autorisé pour la clé {key_id}")
    return True 