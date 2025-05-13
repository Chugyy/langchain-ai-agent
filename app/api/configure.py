"""
Configuration de l'API et des outils disponibles.
"""
from typing import List, Dict, Any
from pydantic import BaseSettings, Field

# Outils disponibles par défaut
DEFAULT_TOOLS = [
    "youtube_transcript",
    "list_available_media", 
    "process_image",
    "process_document",
    "process_audio",
    "describe_media", 
    "load_media_from_url",
    "extract_media_content",
    "extract_video_audio",
    "extract_video_frames",
    "whatsapp_send_message",
    "whatsapp_reply_to_chat",
    "email_send",
    "email_retrieve",
    "creer_evenement",
    "lister_evenements",
    "supprimer_evenement",
    "mettre_a_jour_evenement",
    "calculer_date"  # Ajout du nouvel outil de date
]

# Liste des outils disponibles
AVAILABLE_TOOLS = DEFAULT_TOOLS

# Configurations par défaut
DEFAULT_LLM_NAME = "gpt-4o-mini"
DEFAULT_MEMORY_TYPE = "conversation_buffer"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SESSION_TTL = 24  # heures
DEFAULT_MAX_HISTORY = 100  # nombre de messages
DEFAULT_DEBUG_MODE = False  # si True, renvoie le processus de pensée
DEFAULT_VERBOSE = True  # si True, active les logs verbeux
DEFAULT_STREAM = False  # si True, active le streaming des réponses
# ... existing code ... 