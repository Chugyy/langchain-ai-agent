from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Union, Dict, Any
from datetime import datetime
from enum import Enum


class MediaType(str, Enum):
    """Types de médias supportés par l'API"""
    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    LINK = "link"  # URL générique


class MediaReference(BaseModel):
    """
    Référence à un média externe (URL) dans une requête
    """
    url: HttpUrl = Field(..., description="URL du média à charger")
    type: Optional[MediaType] = Field(None, description="Type de média (auto-détecté si non spécifié)")
    reference_id: Optional[str] = Field(None, description="Identifiant local pour référencer ce média dans le message")
    title: Optional[str] = Field(None, description="Titre optionnel du média")
    description: Optional[str] = Field(None, description="Description optionnelle du média")
    
    class Config:
        schema_extra = {
            "example": {
                "url": "https://example.com/image.jpg"
            }
        }


class MediaInfo(BaseModel):
    """
    Informations sur un média traité par le système
    """
    media_id: str = Field(..., description="ID unique du média")
    original_url: HttpUrl = Field(..., description="URL d'origine du média")
    media_type: MediaType = Field(..., description="Type de média")
    content_type: str = Field(..., description="Type MIME du média")
    size: int = Field(..., description="Taille en octets")
    reference_id: Optional[str] = Field(None, description="Identifiant de référence fourni par l'utilisateur")
    download_date: datetime = Field(..., description="Date de téléchargement")
    processed: bool = Field(default=False, description="Si le média a été traité")
    title: Optional[str] = Field(None, description="Titre du média")
    description: Optional[str] = Field(None, description="Description du média")
    
    class Config:
        schema_extra = {
            "example": {
                "media_id": "8f7d8f7d-8f7d-8f7d-8f7d-8f7d8f7d8f7d",
                "original_url": "https://example.com/image.jpg",
                "media_type": "image",
                "content_type": "image/jpeg",
                "size": 12345,
                "reference_id": "img1",
                "download_date": "2023-05-16T14:30:00",
                "processed": True,
                "title": "Mon image",
                "description": "Une belle image"
            }
        }


class ChatRequestMedia(BaseModel):
    """
    Extension du modèle de requête ChatRequest pour inclure des références de médias
    """
    message: str = Field(..., description="Message utilisateur")
    session_id: Optional[str] = Field(None, description="ID de session existante")
    config: Optional[Dict[str, Any]] = Field(None, description="Configuration optionnelle")
    media: Optional[List[MediaReference]] = Field(None, description="Liste des médias à traiter avec ce message")

    class Config:
        schema_extra = {
            "example": {
                "message": "Que vois-tu sur cette image ?",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "media": [
                    {
                        "url": "https://example.com/image.jpg",
                        "type": "image",
                        "reference_id": "img1"
                    }
                ]
            }
        }


class ChatResponseMedia(BaseModel):
    """
    Extension du modèle de réponse ChatResponse pour inclure des références aux médias traités
    """
    response: str = Field(..., description="Réponse de l'agent")
    session_id: str = Field(..., description="ID de session")
    thinking: Optional[str] = Field(None, description="Processus de réflexion de l'agent")
    media: Optional[List[MediaInfo]] = Field(None, description="Liste des médias traités dans cette conversation")

    class Config:
        schema_extra = {
            "example": {
                "response": "Je vois une image qui montre...",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "media": [
                    {
                        "media_id": "8f7d8f7d-8f7d-8f7d-8f7d-8f7d8f7d8f7d",
                        "original_url": "https://example.com/image.jpg",
                        "media_type": "image",
                        "content_type": "image/jpeg",
                        "size": 12345,
                        "reference_id": "img1",
                        "download_date": "2023-05-16T14:30:00",
                        "processed": True
                    }
                ]
            }
        } 