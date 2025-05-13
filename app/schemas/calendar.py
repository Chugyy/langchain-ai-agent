"""Schémas pour les outils de gestion de calendrier Google."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ListEventsSchema(BaseModel):
    """Schéma pour lister les événements du calendrier."""
    count: int = Field(10, description="Nombre d'événements à récupérer", ge=1, le=100)
    calendar_id: Optional[str] = Field(None, description="ID du calendrier (utilise les paramètres par défaut si None)")


class CreateEventSchema(BaseModel):
    """Schéma pour créer un événement dans le calendrier."""
    summary: str = Field(..., description="Titre de l'événement")
    start_time: str = Field(..., description="Date et heure de début (format ISO: 2023-12-24T15:00:00)")
    end_time: str = Field(..., description="Date et heure de fin (format ISO: 2023-12-24T16:00:00)")
    description: Optional[str] = Field(None, description="Description de l'événement")
    location: Optional[str] = Field(None, description="Lieu de l'événement")
    attendees: Optional[List[str]] = Field(None, description="Liste des emails des participants")
    calendar_id: Optional[str] = Field(None, description="ID du calendrier (utilise les paramètres par défaut si None)")


class UpdateEventSchema(BaseModel):
    """Schéma pour mettre à jour un événement existant."""
    event_id: str = Field(..., description="ID de l'événement à modifier")
    summary: Optional[str] = Field(None, description="Nouveau titre de l'événement")
    start_time: Optional[str] = Field(None, description="Nouvelle date/heure de début (format ISO)")
    end_time: Optional[str] = Field(None, description="Nouvelle date/heure de fin (format ISO)")
    description: Optional[str] = Field(None, description="Nouvelle description")
    location: Optional[str] = Field(None, description="Nouveau lieu")
    attendees: Optional[List[str]] = Field(None, description="Nouvelle liste des participants")
    calendar_id: Optional[str] = Field(None, description="ID du calendrier (utilise les paramètres par défaut si None)")


class DeleteEventSchema(BaseModel):
    """Schéma pour supprimer un événement."""
    event_id: str = Field(..., description="ID de l'événement à supprimer")
    calendar_id: Optional[str] = Field(None, description="ID du calendrier (utilise les paramètres par défaut si None)") 