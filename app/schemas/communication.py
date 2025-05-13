"""Schémas pour les outils de communication."""
from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr


class WhatsAppConnectSchema(BaseModel):
    """Schéma pour connecter un compte WhatsApp."""
    pass


class WhatsAppCheckStatusSchema(BaseModel):
    """Schéma pour vérifier le statut d'un compte WhatsApp."""
    account_id: str = Field(..., description="ID du compte WhatsApp")


class WhatsAppWaitConnectionSchema(BaseModel):
    """Schéma pour attendre la connexion d'un compte WhatsApp."""
    account_id: str = Field(..., description="ID du compte WhatsApp")
    max_minutes: int = Field(5, description="Temps maximum d'attente en minutes", ge=1, le=30)


class WhatsAppSendMessageSchema(BaseModel):
    """Schéma pour envoyer un message WhatsApp."""
    phone_number: str = Field(..., description="Numéro de téléphone du destinataire")
    message: str = Field(..., description="Texte du message")
    account_id: Optional[str] = Field(None, description="ID du compte WhatsApp (optionnel)")


class WhatsAppReplyToChatSchema(BaseModel):
    """Schéma pour répondre à un chat WhatsApp existant."""
    chat_id: str = Field(..., description="ID du chat WhatsApp")
    message: str = Field(..., description="Texte du message")


class EmailSendSchema(BaseModel):
    """Schéma pour envoyer un email."""
    to: List[str] = Field(..., description="Adresse(s) email du/des destinataire(s)")
    subject: str = Field(..., description="Sujet de l'email")
    body: str = Field(..., description="Corps du message (supporte le HTML)")
    cc: Optional[List[str]] = Field(None, description="Adresse(s) email en copie")
    bcc: Optional[List[str]] = Field(None, description="Adresse(s) email en copie cachée")
    attachments: Optional[List[str]] = Field(None, description="Chemins des pièces jointes")


class EmailRetrieveSchema(BaseModel):
    """Schéma pour récupérer des emails."""
    folder: str = Field("INBOX", description="Dossier à consulter (ex: INBOX, SENT)")
    limit: int = Field(10, description="Nombre maximum d'emails à récupérer", ge=1, le=50)
    unread_only: bool = Field(False, description="Ne récupérer que les emails non lus")
    search_query: Optional[str] = Field(None, description="Critère de recherche (ex: FROM someone@example.com)") 