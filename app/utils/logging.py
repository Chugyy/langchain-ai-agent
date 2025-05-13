"""
Module de configuration du logging pour l'application.
Fournit un logging structuré et centralisé.
"""
import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional


class JSONFormatter(logging.Formatter):
    """Formatteur pour produire des logs au format JSON structuré."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Formate un enregistrement de log en JSON.
        
        Args:
            record: L'enregistrement de log à formater
            
        Returns:
            Le log formaté en JSON
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Ajout des données supplémentaires
        if hasattr(record, "data") and isinstance(record.data, dict):
            log_data.update(record.data)
        
        # Gestion des exceptions
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }
        
        return json.dumps(log_data)


def setup_logging(log_level: str = "DEBUG") -> None:
    """
    Configure le système de logging pour l'application.
    
    Args:
        log_level: Le niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Conversion du niveau de log
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configuration du logger root
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Suppression des handlers existants
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Ajout d'un handler pour la sortie standard
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root_logger.addHandler(handler)
    
    # Configuration du logger pour les librairies tierces
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Récupère un logger configuré pour un module spécifique.
    
    Args:
        name: Le nom du module
        
    Returns:
        Un logger configuré
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """Adaptateur permettant d'ajouter des données contextuelles aux logs."""
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any] = None):
        """
        Initialise l'adaptateur.
        
        Args:
            logger: Le logger à adapter
            extra: Données contextuelles à ajouter
        """
        super().__init__(logger, extra or {})
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Traite le message et les kwargs pour ajouter les données contextuelles.
        
        Args:
            msg: Le message à logger
            kwargs: Les arguments supplémentaires
            
        Returns:
            Tuple (message, kwargs) modifié
        """
        kwargs["extra"] = kwargs.get("extra", {})
        data = kwargs["extra"].get("data", {})
        data.update(self.extra)
        kwargs["extra"]["data"] = data
        return msg, kwargs


def get_contextualized_logger(name: str, **context) -> LoggerAdapter:
    """
    Récupère un logger avec contexte.
    
    Args:
        name: Le nom du module
        context: Contexte à ajouter aux logs
        
    Returns:
        Un adaptateur de logger
    """
    logger = get_logger(name)
    return LoggerAdapter(logger, context) 