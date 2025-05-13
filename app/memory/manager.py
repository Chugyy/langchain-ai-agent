"""
Gestionnaire de mémoire pour l'agent IA.
Supporte différents types de mémoire (buffer, summary, etc.).
"""
from typing import Dict, List, Optional, Any
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory
)
from langchain.memory.chat_memory import BaseChatMemory
from langchain.llms.base import BaseLLM
from app.utils.logging import get_logger

# Logger pour ce module
logger = get_logger(__name__)


class MemoryManager:
    """Gestionnaire de mémoire pour les conversations de l'agent."""
    
    def __init__(
        self,
        type: str = "buffer",
        llm: Optional[BaseLLM] = None,
        max_message_count: int = 10,
        memory_key: str = "chat_history",
        return_messages: bool = True,
    ):
        """
        Initialise le gestionnaire de mémoire.
        
        Args:
            type: Type de mémoire ("buffer", "summary", "window")
            llm: Instance de LLM (nécessaire pour le type "summary")
            max_message_count: Nombre maximum de messages à conserver
            memory_key: Clé pour stocker l'historique dans le contexte
            return_messages: Si True, retourne les messages complets
        """
        self.type = type.lower().strip()
        self.llm = llm
        self.max_message_count = max_message_count
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.memory = self._create_memory()
        
        logger.info(f"Mémoire de type '{self.type}' initialisée")
    
    def _create_memory(self) -> BaseChatMemory:
        """
        Crée l'instance de mémoire selon le type demandé.
        
        Returns:
            Une instance de mémoire configurée
        
        Raises:
            ValueError: Si le type demandé n'est pas supporté
        """
        if self.type == "buffer":
            return ConversationBufferMemory(
                memory_key=self.memory_key,
                return_messages=self.return_messages
            )
        elif self.type == "summary":
            if not self.llm:
                raise ValueError("Un LLM est requis pour la mémoire de type 'summary'")
            return ConversationSummaryMemory(
                llm=self.llm,
                memory_key=self.memory_key,
                return_messages=self.return_messages
            )
        elif self.type == "window":
            return ConversationBufferWindowMemory(
                k=self.max_message_count,
                memory_key=self.memory_key,
                return_messages=self.return_messages
            )
        else:
            raise ValueError(f"Type de mémoire non supporté: {self.type}")
    
    def get_memory(self) -> BaseChatMemory:
        """
        Récupère l'instance de mémoire.
        
        Returns:
            L'instance de mémoire
        """
        return self.memory
    
    def clear(self) -> None:
        """Efface le contenu de la mémoire."""
        self.memory.clear()
        logger.debug("Mémoire effacée")


class MemoryStorage:
    """
    Gestionnaire de stockage des sessions de mémoire.
    Permet de conserver plusieurs sessions indépendantes.
    """
    
    def __init__(self):
        """Initialise le gestionnaire de stockage."""
        self._sessions: Dict[str, BaseChatMemory] = {}
        logger.info("Stockage de mémoire initialisé")
    
    def get_or_create(
        self,
        session_id: str,
        memory_type: str = "buffer",
        llm: Optional[BaseLLM] = None,
        max_message_count: int = 10
    ) -> BaseChatMemory:
        """
        Récupère ou crée une session de mémoire.
        
        Args:
            session_id: Identifiant de la session
            memory_type: Type de mémoire à créer
            llm: Instance de LLM
            max_message_count: Nombre maximum de messages
            
        Returns:
            L'instance de mémoire associée à la session
        """
        if session_id not in self._sessions:
            manager = MemoryManager(
                type=memory_type,
                llm=llm,
                max_message_count=max_message_count
            )
            self._sessions[session_id] = manager.get_memory()
            logger.info(f"Nouvelle session de mémoire créée: {session_id}")
        
        return self._sessions[session_id]
    
    def get(self, session_id: str) -> Optional[BaseChatMemory]:
        """
        Récupère une session de mémoire existante.
        
        Args:
            session_id: Identifiant de la session
            
        Returns:
            L'instance de mémoire ou None si la session n'existe pas
        """
        return self._sessions.get(session_id)
    
    def delete(self, session_id: str) -> bool:
        """
        Supprime une session de mémoire.
        
        Args:
            session_id: Identifiant de la session
            
        Returns:
            True si la session a été supprimée, False sinon
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session de mémoire supprimée: {session_id}")
            return True
        return False
    
    def clear_all(self) -> None:
        """Supprime toutes les sessions de mémoire."""
        self._sessions.clear()
        logger.info("Toutes les sessions de mémoire ont été supprimées") 