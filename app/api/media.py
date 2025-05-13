"""
API pour la gestion des médias multimodaux.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import AnyHttpUrl
from typing import List, Dict, Optional, Any, Tuple

from app.schemas.media import MediaReference, MediaInfo, MediaType
from app.tools.media_tools import (
    fetch_media_from_url, 
    get_media_metadata, 
    list_media, 
    cleanup_old_media,
    media_registry
)
from app.utils.logging import get_logger
from app.api.auth import get_api_key

logger = get_logger(__name__)

# Router API
router = APIRouter(
    prefix="/media",
    tags=["media"]
)


@router.post("/load", response_model=MediaInfo)
async def load_media_from_url(
    media: MediaReference,
    api_auth: Tuple[str, Dict] = Depends(get_api_key)
):
    """
    Charge un média depuis une URL et le stocke pour traitement
    
    - **url**: URL du média à charger
    - **type**: Type de média (optionnel, auto-détecté si non spécifié)
    - **reference_id**: Identifiant local (optionnel)
    - **title**: Titre du média (optionnel)
    - **description**: Description du média (optionnel)
    
    Retourne les métadonnées du média chargé.
    """
    # Extraire l'ID de la clé API
    key_id, _ = api_auth
    
    try:
        # Charger le média depuis l'URL
        logger.info(f"Chargement de média demandé", extra={
            "url": str(media.url),
            "type": media.type.value if media.type else "auto",
            "key_id": key_id
        })
        
        # Récupérer ou créer la session_id à partir du contexte
        session_id = None  # À implémenter: récupérer la session de la requête
        
        # Charger le média
        metadata = fetch_media_from_url(str(media.url), session_id)
        if not metadata:
            raise HTTPException(status_code=400, detail=f"Impossible de charger le média depuis {media.url}")
        
        # Ajouter les champs optionnels s'ils sont fournis
        if media.reference_id:
            metadata.reference_id = media.reference_id
        if media.title:
            metadata.title = media.title
        if media.description:
            metadata.description = media.description
        
        # Convertir la classe MediaMetadata en Pydantic MediaInfo
        media_info = MediaInfo(
            media_id=metadata.media_id,
            original_url=metadata.original_url,
            media_type=metadata.media_type,
            content_type=metadata.content_type,
            size=metadata.size,
            reference_id=getattr(metadata, "reference_id", None),
            download_date=metadata.download_date,
            processed=metadata.processed,
            title=getattr(metadata, "title", None),
            description=getattr(metadata, "description", None)
        )
        
        logger.info(f"Média chargé avec succès", extra={
            "media_id": metadata.media_id,
            "media_type": metadata.media_type,
            "size": metadata.size
        })
        
        return media_info
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du média: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement du média: {str(e)}")


@router.get("/list", response_model=List[MediaInfo])
async def list_available_media(
    session_id: Optional[str] = None,
    api_auth: Tuple[str, Dict] = Depends(get_api_key)
):
    """
    Liste tous les médias disponibles, filtré par session_id si fourni
    
    - **session_id**: ID de session pour filtrer (optionnel)
    
    Retourne la liste des médias correspondants.
    """
    try:
        media_list = list_media(session_id)
        
        # Convertir les objets MediaMetadata en Pydantic MediaInfo
        result = []
        for metadata in media_list:
            media_info = MediaInfo(
                media_id=metadata.media_id,
                original_url=metadata.original_url,
                media_type=metadata.media_type,
                content_type=metadata.content_type,
                size=metadata.size,
                reference_id=getattr(metadata, "reference_id", None),
                download_date=metadata.download_date,
                processed=metadata.processed,
                title=getattr(metadata, "title", None),
                description=getattr(metadata, "description", None)
            )
            result.append(media_info)
            
        return result
        
    except Exception as e:
        logger.error(f"Erreur lors de la liste des médias: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la liste des médias: {str(e)}")


@router.get("/{media_id}", response_model=MediaInfo)
async def get_media_info(
    media_id: str,
    api_auth: Tuple[str, Dict] = Depends(get_api_key)
):
    """
    Récupère les métadonnées d'un média par son ID
    
    - **media_id**: ID unique du média
    
    Retourne les métadonnées du média.
    """
    try:
        metadata = get_media_metadata(media_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Média non trouvé: {media_id}")
            
        # Convertir l'objet MediaMetadata en Pydantic MediaInfo
        media_info = MediaInfo(
            media_id=metadata.media_id,
            original_url=metadata.original_url,
            media_type=metadata.media_type,
            content_type=metadata.content_type,
            size=metadata.size,
            reference_id=getattr(metadata, "reference_id", None),
            download_date=metadata.download_date,
            processed=metadata.processed,
            title=getattr(metadata, "title", None),
            description=getattr(metadata, "description", None)
        )
        
        return media_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des infos du média: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des infos du média: {str(e)}")


@router.delete("/{media_id}")
async def delete_media(
    media_id: str,
    api_auth: Tuple[str, Dict] = Depends(get_api_key)
):
    """
    Supprime un média et ses métadonnées
    
    - **media_id**: ID unique du média à supprimer
    
    Retourne un message de confirmation.
    """
    try:
        metadata = get_media_metadata(media_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Média non trouvé: {media_id}")
            
        # Supprimer le fichier
        import os
        if os.path.exists(metadata.local_path):
            os.remove(metadata.local_path)
            
        # Supprimer du registre
        if media_id in media_registry:
            del media_registry[media_id]
            
        logger.info(f"Média supprimé avec succès", extra={"media_id": media_id})
        return {"message": f"Média {media_id} supprimé avec succès"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la suppression du média: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la suppression du média: {str(e)}")


@router.post("/cleanup")
async def cleanup_media(
    max_age_hours: int = 24,
    api_auth: Tuple[str, Dict] = Depends(get_api_key)
):
    """
    Nettoie les médias plus anciens que max_age_hours
    
    - **max_age_hours**: Âge maximum en heures (défaut: 24)
    
    Retourne le nombre de médias supprimés.
    """
    try:
        count = cleanup_old_media(max_age_hours)
        logger.info(f"Nettoyage des médias effectué", extra={"count": count, "max_age_hours": max_age_hours})
        return {"message": f"{count} médias supprimés", "count": count}
        
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage des médias: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors du nettoyage des médias: {str(e)}") 