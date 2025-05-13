import os
import logging
from typing import List, Dict, Any, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Charger les variables .env
load_dotenv()

name = os.getenv('LLM_NAME')
openai_api_key = os.getenv('OPENAI_API_KEY')
admin_api_key = os.getenv('ADMIN_API_KEY', 'admin-key-for-development')
environment = os.getenv('ENVIRONMENT', 'development')
debug_mode = os.getenv('DEBUG_MODE', 'true').lower() in ('true', '1', 't')

# Configuration basique du logging pour les messages de démarrage
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("settings")


# Fonction auxiliaire pour parser la liste d'outils
def parse_tools_list(tools_str: Optional[str]) -> List[str]:
    """
    Parse une chaîne de caractères séparée par des virgules en liste.
    
    Args:
        tools_str: chaîne de caractères (ex: "outil1,outil2,outil3")
        
    Returns:
        Liste de chaînes (ex: ["outil1", "outil2", "outil3"])
    """
    if not tools_str:
        return []
        
    # Supprimer les espaces et diviser par virgules
    tools = [t.strip() for t in tools_str.split(',')]
    # Filtrer les valeurs vides
    return [t for t in tools if t]


class LLMSettings(BaseSettings):
    """Configuration du modèle de langage."""
    name: str = name
    temperature: float = Field(0.0, env="TEMPERATURE")
    max_tokens: int = Field(1000, env="MAX_TOKENS")
    openai_api_key: str = openai_api_key

    model_config = SettingsConfigDict(
        env_prefix="LLM__",
        extra="ignore",
    )


class MemorySettings(BaseSettings):
    """Configuration de la mémoire de conversation."""
    type: str = Field("buffer", env="MEMORY_TYPE")
    max_message_count: int = Field(10, env="MAX_MESSAGE_COUNT")

    model_config = SettingsConfigDict(extra="ignore")


class SessionSettings(BaseSettings):
    """Configuration des sessions."""
    ttl_hours: int = Field(24, env="SESSION_TTL_HOURS")

    model_config = SettingsConfigDict(extra="ignore")


class ServerSettings(BaseSettings):
    """Configuration du serveur."""
    host: str = Field("0.0.0.0", env="SERVER_HOST")
    port: int = Field(8000, env="SERVER_PORT")
    workers: int = Field(4, env="SERVER_WORKERS")
    log_level: str = Field("debug", env="LOG_LEVEL")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], env="CORS_ORIGINS")

    model_config = SettingsConfigDict(extra="ignore")


class ToolsSettings(BaseSettings):
    """Paramètres des outils disponibles pour l'agent"""
    enabled: List[str] = Field(default_factory=lambda: [
        "shout", 
        "file_loader", 
        "youtube_transcript", 
        # Outils média simplifiés
        "load_media_from_url",
        "list_available_media",
        "extract_media_content",
        # Outils communication
        "whatsapp_send_message",
        "whatsapp_reply_to_chat",
        # Outils email
        "email_send",
        "email_retrieve",
        # Agenda Google
        "creer_evenement",
        "lister_evenements",
        "supprimer_evenement",
        "mettre_a_jour_evenement",
        "calculer_date"
    ], env="ENABLED_TOOLS")
    
    # Récupérer et parser la variable d'environnement ENABLED_TOOLS
    @validator('enabled', pre=True)
    def parse_enabled_tools(cls, value):
        # Si c'est déjà une liste, la retourner telle quelle
        if isinstance(value, list):
            return value
            
        # Si c'est une chaîne, la parser
        if isinstance(value, str):
            return parse_tools_list(value)
            
        return []

    model_config = SettingsConfigDict(
        env_prefix="",  # Pas de préfixe pour les variables d'environnement
        extra="ignore",
    )


class ApiKeysSettings(BaseSettings):
    """Configuration centralisée des clés API pour les différents services."""
    openai: Optional[str] = None
    rapid_api: Optional[str] = None  # Ajout du champ manquant
    # Ajoutez d'autres clés API selon vos besoins

    def get(self, service_name: str) -> Optional[str]:
        """Récupère une clé API par son nom de service."""
        return getattr(self, service_name, None)
    
    def all(self) -> Dict[str, Optional[str]]:
        """Renvoie toutes les clés API sous forme de dictionnaire."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def verify_required_keys(self, required_keys: List[str] = None) -> List[str]:
        """
        Vérifie quelles clés API requises sont manquantes.
        
        Args:
            required_keys: Liste des noms de clés API à vérifier
                          (par défaut toutes les clés sont vérifiées)
                          
        Returns:
            Liste des noms de clés API manquantes
        """
        missing_keys = []
        keys_to_check = required_keys or [k for k in self.__dict__ if not k.startswith('_')]
        
        for key in keys_to_check:
            value = getattr(self, key, None)
            if value is None or value == "":
                missing_keys.append(key)
                
        return missing_keys

    model_config = SettingsConfigDict(
        env_prefix="",  # Pas de préfixe pour les variables d'environnement
        extra="ignore"
    )


class EmailSettings(BaseSettings):
    """Configuration du service email."""
    smtp_host: str = Field("smtp.hostinger.com", env="EMAIL_SMTP_HOST")
    smtp_port: int = Field(465, env="EMAIL_SMTP_PORT")
    imap_host: str = Field("imap.hostinger.com", env="EMAIL_IMAP_HOST") 
    imap_port: int = Field(993, env="EMAIL_IMAP_PORT")
    username: Optional[str] = Field("contact@multimodal.digital", env="EMAIL_USERNAME")
    password: Optional[str] = Field("#ZiazkKu37BdukBh", env="EMAIL_PASSWORD")
    sender_name: Optional[str] = Field("Hugo Hoarau - Multimodal", env="EMAIL_SENDER_NAME")
    use_tls: bool = True

    model_config = SettingsConfigDict(extra="ignore")


class GoogleCalendarSettings(BaseSettings):
    """Configuration pour l'accès à Google Calendar."""
    credentials_file: str = Field("credentials.json", env="GCAL_CREDENTIALS_FILE")
    token_file: str = Field("token.json", env="GCAL_TOKEN_FILE")
    scopes_read_only: bool = Field(False, env="GCAL_READ_ONLY") 
    calendar_id: str = Field("primary", env="GCAL_CALENDAR_ID")

    model_config = SettingsConfigDict(extra="ignore")


class SecuritySettings(BaseSettings):
    """Configuration de sécurité."""
    api_key_prefix: str = Field("sk-", env="API_KEY_PREFIX")
    token_expiration_days: int = Field(30, env="TOKEN_EXPIRATION_DAYS")
    jwt_secret: str = Field("dev-jwt-secret-not-for-production", env="JWT_SECRET")
    api_key_salt: str = Field("dev-salt-not-for-production", env="API_KEY_SALT")
    model_config = SettingsConfigDict(extra="ignore")


class Settings(BaseSettings):
    """Configuration globale de l'application."""
    llm: LLMSettings         = Field(default_factory=LLMSettings)
    memory: MemorySettings    = Field(default_factory=MemorySettings)
    session: SessionSettings  = Field(default_factory=SessionSettings)
    server: ServerSettings    = Field(default_factory=ServerSettings)
    tools: ToolsSettings      = Field(default_factory=ToolsSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    api_keys: ApiKeysSettings = Field(default_factory=ApiKeysSettings)
    email: EmailSettings      = Field(default_factory=EmailSettings)
    calendar: GoogleCalendarSettings = Field(default_factory=GoogleCalendarSettings)
    api_host: str            = Field("0.0.0.0", env="API_HOST")
    api_port: int            = Field(8000, env="API_PORT")
    environment: str         = Field(environment, env="ENVIRONMENT")
    debug_mode: bool         = Field(debug_mode, env="DEBUG_MODE")
    admin_api_key: str       = Field(admin_api_key, env="ADMIN_API_KEY")
    unipile_dsn: str         = Field("api.unipile.com", env="UNIPILE_DSN")
    whatsapp_account_id: Optional[str] = Field(None, env="WA_ACCOUNTID")

    def check_api_keys(self) -> None:
        """
        Vérifie et affiche des informations sur les clés API manquantes.
        Utile pour le débogage au démarrage de l'application.
        """
        missing_keys = self.api_keys.verify_required_keys()
        
        if missing_keys:
            logger.warning(f"Les clés API suivantes sont manquantes ou vides: {', '.join(missing_keys)}")
            
            # Conseils spécifiques pour chaque clé
            if 'openai' in missing_keys:
                logger.warning("La clé OPENAI_API_KEY est requise pour le fonctionnement du LLM et les transcriptions audio/vidéo")
            else:
                logger.info("Toutes les clés API configurées sont présentes")
        
        # Vérifier les outils activés
        if not self.tools.enabled:
            logger.warning("Aucun outil n'est activé. Définissez ENABLED_TOOLS dans .env")
        else:
            logger.info(f"Outils activés: {', '.join(self.tools.enabled)}")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )


def get_settings() -> Settings:
    """Retourne une instance de Settings en ne se basant que sur .env et vars système."""
    # Récupérer d'abord la valeur de ENABLED_TOOLS directement
    enabled_tools_str = os.getenv('ENABLED_TOOLS')
    if enabled_tools_str:
        logger.debug(f"ENABLED_TOOLS raw value: {enabled_tools_str}")
    
    settings = Settings()
    
    # Charger explicitement ENABLED_TOOLS pour éviter les problèmes de préfixe
    if enabled_tools_str:
        settings.tools.enabled = parse_tools_list(enabled_tools_str)
    
    # Charger explicitement les clés API
    settings.api_keys.openai = os.getenv('OPENAI_API_KEY')
    
    # Charger le DSN Unipile
    unipile_dsn = os.getenv('UNIPILE_DSN')
    if unipile_dsn:
        settings.unipile_dsn = unipile_dsn
    
    # Charger l'ID de compte WhatsApp
    wa_account_id = os.getenv('WA_ACCOUNTID')
    if wa_account_id:
        settings.whatsapp_account_id = wa_account_id
    
    # Charger les paramètres email
    settings.email.username = os.getenv('EMAIL_USERNAME')
    settings.email.password = os.getenv('EMAIL_PASSWORD')
    settings.email.sender_name = os.getenv('EMAIL_SENDER_NAME')
    
    # Logger la valeur parsée
    logger.debug(f"ENABLED_TOOLS parsed: {settings.tools.enabled}")
    
    # Vérifier les clés API au chargement des paramètres
    if environment != "test":  # Ne pas afficher les avertissements pendant les tests
        settings.check_api_keys()
        
    return settings