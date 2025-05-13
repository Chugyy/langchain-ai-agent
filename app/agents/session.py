"""
Service de gestion des sessions pour l'agent IA.
Gère la création, récupération et suppression des sessions.
"""
import uuid
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from app.agents.agent import Agent, AgentFactory
from app.memory.manager import memory_storage
from app.utils.settings import get_settings
from app.utils.logging import get_logger

# Logger pour ce module
logger = get_logger(__name__)


class Session:
    """Représente une session de conversation avec l'agent."""
    
    def __init__(
        self,
        session_id: str,
        agent: Agent,
        created_at: Optional[datetime] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialise une session.
        
        Args:
            session_id: Identifiant unique de la session
            agent: Instance de l'agent associé à la session
            created_at: Date de création (par défaut: maintenant)
            config: Configuration de la session
        """
        self.session_id = session_id
        self.agent = agent
        self.created_at = created_at or datetime.utcnow()
        self.last_interaction = self.created_at
        self.config = config or {}
    
    def update_last_interaction(self):
        """Met à jour la date de dernière interaction."""
        self.last_interaction = datetime.utcnow()
    
    def is_expired(self, ttl_hours: int) -> bool:
        """
        Vérifie si la session a expiré.
        
        Args:
            ttl_hours: Durée de vie en heures
            
        Returns:
            True si la session a expiré, False sinon
        """
        expiration_time = self.last_interaction + timedelta(hours=ttl_hours)
        return datetime.utcnow() > expiration_time
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit la session en dictionnaire.
        
        Returns:
            Dictionnaire représentant la session
        """
        # Récupération de l'historique depuis la mémoire
        memory = self.agent.memory
        history = []
        
        # Tentative de récupération de l'historique (peut varier selon le type de mémoire)
        if hasattr(memory, "chat_memory") and hasattr(memory.chat_memory, "messages"):
            history = [
                {"role": msg.type, "content": msg.content}
                for msg in memory.chat_memory.messages
            ]
        
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_interaction": self.last_interaction.isoformat(),
            "config": self.config,
            "history": history
        }


class SessionManager:
    """Gestionnaire des sessions de l'agent."""
    
    def __init__(self, settings: Settings):
        """
        Initialise le gestionnaire de sessions.
        
        Args:
            settings: Configuration globale
        """
        self.settings = settings
        self._sessions: Dict[str, Session] = {}
        self._last_cleanup = datetime.utcnow()
        logger.info("Gestionnaire de sessions initialisé")
    
    def create_session(
        self,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[Session, str]:
        """
        Crée une nouvelle session.
        
        Args:
            config: Configuration spécifique pour cette session
            
        Returns:
            Tuple (session, session_id)
        """
        # Génération d'un identifiant unique
        session_id = str(uuid.uuid4())
        
        # Configuration de la session
        session_config = config or {}
        
        # Création de l'agent
        agent, _ = AgentFactory.create_agent(
            llm_settings=self.settings.llm,
            memory_settings=self.settings.memory,
            tools_settings=self.settings.tools,
            session_id=session_id,
            verbose=False
        )
        
        # Création de la session
        session = Session(
            session_id=session_id,
            agent=agent,
            config=session_config
        )
        
        # Enregistrement de la session
        self._sessions[session_id] = session
        logger.info(f"Nouvelle session créée: {session_id}")
        
        # Nettoyage périodique des sessions expirées
        self._cleanup_if_needed()
        
        return session, session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Récupère une session existante.
        
        Args:
            session_id: Identifiant de la session
            
        Returns:
            La session si elle existe, None sinon
        """
        session = self._sessions.get(session_id)
        
        if session:
            # Vérification de l'expiration
            if session.is_expired(self.settings.session.ttl_hours):
                logger.info(f"Tentative d'accès à une session expirée: {session_id}")
                self.delete_session(session_id)
                return None
            
            # Mise à jour de la date de dernière interaction
            session.update_last_interaction()
        
        return session
    
    def update_session_config(
        self,
        session_id: str,
        config: Dict[str, Any]
    ) -> Optional[Session]:
        """
        Met à jour la configuration d'une session.
        
        Args:
            session_id: Identifiant de la session
            config: Nouvelle configuration
            
        Returns:
            La session mise à jour ou None si elle n'existe pas
        """
        session = self.get_session(session_id)
        
        if not session:
            return None
        
        # Mise à jour de la configuration
        session.config.update(config)
        logger.info(f"Configuration de la session {session_id} mise à jour")
        
        return session
    
    def delete_session(self, session_id: str) -> bool:
        """
        Supprime une session.
        
        Args:
            session_id: Identifiant de la session
            
        Returns:
            True si la session a été supprimée, False sinon
        """
        if session_id in self._sessions:
            # Suppression de la mémoire associée
            memory_storage.delete(session_id)
            
            # Suppression de la session
            del self._sessions[session_id]
            logger.info(f"Session supprimée: {session_id}")
            return True
        
        return False
    
    def _cleanup_if_needed(self):
        """
        Nettoie les sessions expirées si nécessaire.
        Exécuté périodiquement lors de la création de sessions.
        """
        now = datetime.utcnow()
        # Nettoyage au plus une fois par heure
        if (now - self._last_cleanup).total_seconds() < 3600:
            return
        
        expired_sessions = [
            session_id for session_id, session in self._sessions.items()
            if session.is_expired(self.settings.session.ttl_hours)
        ]
        
        for session_id in expired_sessions:
            self.delete_session(session_id)
        
        if expired_sessions:
            logger.info(f"Nettoyage effectué: {len(expired_sessions)} sessions expirées supprimées")
        
        self._last_cleanup = now 