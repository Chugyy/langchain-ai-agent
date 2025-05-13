import uuid
import secrets
import hashlib
import time
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta

from app.utils.settings import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# In-memory store for API keys (in a production environment, use a database)
# Structure: {key_hash: {key_data}}
API_KEYS = {}

class APIKeyManager:
    @staticmethod
    def generate_key() -> Tuple[str, str]:
        """
        Generate a new API key and its ID.
        Returns a tuple of (key_id, api_key).
        """
        key_id = str(uuid.uuid4())
        api_key = f"{key_id}.{secrets.token_urlsafe(32)}"
        return key_id, api_key
    
    @staticmethod
    def hash_key(api_key: str) -> str:
        """
        Create a secure hash of the API key for storage.
        """
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    @staticmethod
    def register_key(
        api_key: str, 
        scopes: List[str] = ["chat"], 
        expires_in_days: int = 30,
        rate_limit: int = 100
    ) -> Dict:
        """
        Register a new API key in the store with expiration and rate limiting.
        """
        key_hash = APIKeyManager.hash_key(api_key)
        key_id = api_key.split('.')[0]
        
        now = datetime.utcnow()
        expires_at = now + timedelta(days=expires_in_days)
        
        key_data = {
            "key_id": key_id,
            "scopes": scopes,
            "created_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "rate_limit": rate_limit,
            "request_count": 0,
            "last_reset": int(time.time()),
        }
        
        API_KEYS[key_hash] = key_data
        
        logger.info(f"New API key registered", extra={"key_id": key_id})
        return key_data
    
    @staticmethod
    def validate_key(api_key: str, scope: str = "chat") -> Tuple[bool, Optional[str], Dict]:
        """
        Validate an API key and check its permissions.
        Returns (is_valid, error_message, key_data).
        """
        if not api_key:
            return False, "No API key provided", {}
            
        try:
            # Vérification spéciale pour la clé admin en mode développement
            if settings.environment.lower() == "development" and api_key == settings.admin_api_key:
                logger.warning("Admin key used as API key (dev mode only)")
                # Créer des données fictives pour la clé admin
                admin_key_data = {
                    "key_id": "admin",
                    "scopes": ["chat", "sessions", "admin"],
                    "created_at": datetime.utcnow().isoformat(),
                    "expires_at": (datetime.utcnow() + timedelta(days=365)).isoformat(),
                    "rate_limit": 1000,
                    "request_count": 0,
                    "last_reset": int(time.time())
                }
                return True, None, admin_key_data
            
            key_hash = APIKeyManager.hash_key(api_key)
            
            if key_hash not in API_KEYS:
                logger.warning(f"Invalid API key attempted to be used", extra={"key_hash": key_hash[:8]})
                return False, "Invalid API key", {}
                
            key_data = API_KEYS[key_hash]
            
            # Check if key is expired
            expires_at = datetime.fromisoformat(key_data["expires_at"])
            if expires_at < datetime.utcnow():
                logger.warning(f"Expired API key used", extra={"key_id": key_data["key_id"]})
                return False, "API key expired", key_data
                
            # Check if scope is allowed
            if scope not in key_data["scopes"]:
                logger.warning(f"API key used with invalid scope", extra={
                    "key_id": key_data["key_id"],
                    "requested_scope": scope,
                    "allowed_scopes": key_data["scopes"]
                })
                return False, f"API key does not have {scope} permission", key_data
                
            # Check rate limit - simple time-based reset
            current_time = int(time.time())
            seconds_since_reset = current_time - key_data["last_reset"]
            
            # Reset counter if a day has passed
            if seconds_since_reset > 86400:  # 24 hours
                key_data["request_count"] = 0
                key_data["last_reset"] = current_time
                
            # Check if rate limit is exceeded
            if key_data["request_count"] >= key_data["rate_limit"]:
                logger.warning(f"API key rate limit exceeded", extra={"key_id": key_data["key_id"]})
                return False, "Rate limit exceeded", key_data
                
            # Update request count
            key_data["request_count"] += 1
            
            return True, None, key_data
            
        except Exception as e:
            logger.error(f"Error validating API key: {str(e)}", exc_info=True)
            return False, f"Error validating API key: {str(e)}", {}
    
    @staticmethod
    def revoke_key(api_key: str) -> bool:
        """
        Revoke an API key.
        """
        try:
            key_hash = APIKeyManager.hash_key(api_key)
            
            if key_hash in API_KEYS:
                key_id = API_KEYS[key_hash]["key_id"]
                del API_KEYS[key_hash]
                logger.info(f"API key revoked", extra={"key_id": key_id})
                return True
                
            return False
        except Exception as e:
            logger.error(f"Error revoking API key: {str(e)}", exc_info=True)
            return False 