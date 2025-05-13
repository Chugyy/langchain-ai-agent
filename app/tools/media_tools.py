#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Outils pour gérer et traiter des médias multimodaux (images, vidéos, sons, documents).
"""
import os
import uuid
import requests
from datetime import datetime
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse
import mimetypes
import tempfile
import hashlib
from io import BytesIO
import openai

from app.tools.registry import register
from app.utils.settings import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Configuration
MEDIA_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../media_cache"))
os.makedirs(MEDIA_CACHE_DIR, exist_ok=True)

# Types de médias supportés
SUPPORTED_MIME_TYPES = {
    "image": ["image/jpeg", "image/png", "image/gif", "image/webp"],
    "document": ["application/pdf", "text/plain", "text/csv", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
    "audio": ["audio/mpeg", "audio/wav", "audio/ogg"],
    "video": ["video/mp4", "video/mpeg", "video/webm"]
}

# Structure pour stocker les métadonnées des médias
class MediaMetadata:
    def __init__(self, media_id: str, original_url: str, local_path: str, 
                 media_type: str, content_type: str, size: int,
                 session_id: Optional[str] = None):
        self.media_id = media_id
        self.original_url = original_url
        self.local_path = local_path
        self.media_type = media_type  # image, document, audio, video
        self.content_type = content_type
        self.size = size
        self.session_id = session_id
        self.processed = False
        self.processed_content = None
        self.download_date = datetime.now()

# Registre des médias en mémoire (à remplacer par un système de cache persistant)
media_registry: Dict[str, MediaMetadata] = {}

def url_to_media_type(url: str, content_type: Optional[str] = None) -> str:
    """
    Détermine le type de média à partir de l'URL et/ou du content-type
    
    Args:
        url: URL du média
        content_type: Content-type HTTP (optionnel)
        
    Returns:
        Le type de média (image, document, audio, video)
    """
    if content_type:
        for media_type, mime_types in SUPPORTED_MIME_TYPES.items():
            if any(content_type.startswith(mime) for mime in mime_types):
                return media_type
    
    # Si pas de content-type ou non reconnu, on essaie avec l'extension de l'URL
    path = urlparse(url).path
    ext = os.path.splitext(path)[1].lower()
    
    if ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
        return "image"
    elif ext in ['.pdf', '.txt', '.doc', '.docx']:
        return "document"
    elif ext in ['.mp3', '.wav', '.ogg']:
        return "audio"
    elif ext in ['.mp4', '.mpeg', '.webm']:
        return "video"
    
    # Par défaut, on considère que c'est un document
    return "document"

def fetch_media_from_url(url: str, session_id: Optional[str] = None) -> Optional[MediaMetadata]:
    """
    Télécharge un média depuis une URL et le stocke localement
    
    Args:
        url: URL du média à télécharger
        session_id: ID de session associé (optionnel)
        
    Returns:
        Métadonnées du média téléchargé ou None en cas d'échec
    """
    try:
        # Vérifier si l'URL est valide
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            logger.error(f"URL invalide: {url}")
            return None
            
        # Calculer un hash unique pour cette URL
        url_hash = hashlib.md5(url.encode()).hexdigest()
        media_id = str(uuid.uuid4())
        
        # Télécharger le contenu
        logger.info(f"Téléchargement du média depuis {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Déterminer le type de contenu
        content_type = response.headers.get('Content-Type', '').split(';')[0]
        media_type = url_to_media_type(url, content_type)
        
        # Vérifier si le type est supporté
        is_supported = False
        for supported_types in SUPPORTED_MIME_TYPES.values():
            if any(content_type.startswith(mime) for mime in supported_types):
                is_supported = True
                break
                
        if not is_supported:
            logger.warning(f"Type de média non supporté: {content_type} ({url})")
            # On continue quand même, on sera plus permissif que restrictif
        
        # Déterminer l'extension du fichier
        ext = mimetypes.guess_extension(content_type) or os.path.splitext(parsed_url.path)[1]
        if not ext:
            ext = ".bin"  # Extension par défaut
            
        # Créer un chemin de fichier local
        filename = f"{url_hash}{ext}"
        file_path = os.path.join(MEDIA_CACHE_DIR, filename)
        
        # Télécharger et sauvegarder le contenu
        content = response.content
        with open(file_path, "wb") as f:
            f.write(content)
            
        size = len(content)
        
        # Créer les métadonnées
        metadata = MediaMetadata(
            media_id=media_id,
            original_url=url,
            local_path=file_path,
            media_type=media_type,
            content_type=content_type,
            size=size,
            session_id=session_id
        )
        
        # Stocker dans le registre
        media_registry[media_id] = metadata
        
        logger.info(f"Média téléchargé avec succès: {media_id} ({media_type}, {size} octets)")
        return metadata
        
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement du média {url}: {str(e)}", exc_info=True)
        return None

def get_media_metadata(media_id: str) -> Optional[MediaMetadata]:
    """
    Récupère les métadonnées d'un média par son ID
    
    Args:
        media_id: Identifiant unique du média
        
    Returns:
        Métadonnées du média ou None si non trouvé
    """
    return media_registry.get(media_id)

def list_media(session_id: Optional[str] = None) -> List[MediaMetadata]:
    """
    Liste tous les médias disponibles, filtré par session_id si fourni
    
    Args:
        session_id: ID de session pour filtrer (optionnel)
        
    Returns:
        Liste des médias correspondants
    """
    if session_id:
        return [m for m in media_registry.values() if m.session_id == session_id]
    return list(media_registry.values())

def cleanup_old_media(max_age_hours: int = 24) -> int:
    """
    Nettoie les médias plus anciens que max_age_hours
    
    Args:
        max_age_hours: Âge maximum en heures
        
    Returns:
        Nombre de médias supprimés
    """
    now = datetime.now()
    to_delete = []
    
    for media_id, metadata in media_registry.items():
        age = (now - metadata.download_date).total_seconds() / 3600
        if age > max_age_hours:
            to_delete.append(media_id)
    
    count = 0
    for media_id in to_delete:
        metadata = media_registry[media_id]
        try:
            if os.path.exists(metadata.local_path):
                os.remove(metadata.local_path)
            del media_registry[media_id]
            count += 1
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage du média {media_id}: {str(e)}")
    
    return count

# ===== Fonctions d'extraction de contenu =====

def extract_text_from_image(file_path: str) -> str:
    """
    Extrait le texte d'une image (OCR)
    
    Args:
        file_path: Chemin vers le fichier image
        
    Returns:
        Texte extrait de l'image
    """
    try:
        # Import lazy pour éviter de charger pytesseract si pas nécessaire
        import pytesseract
        from PIL import Image
        
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        
        if not text.strip():
            return "[Aucun texte détecté dans l'image]"
            
        return f"[Texte extrait de l'image]\n\n{text}"
    except ImportError:
        return "[Extraction d'OCR impossible: pytesseract n'est pas installé]"
    except Exception as e:
        return f"[Erreur lors de l'extraction de texte de l'image: {str(e)}]"

def extract_text_from_pdf(file_path: str, max_pages: int = 10) -> str:
    """
    Extrait le texte d'un fichier PDF
    
    Args:
        file_path: Chemin vers le fichier PDF
        max_pages: Nombre maximum de pages à extraire
        
    Returns:
        Texte extrait du PDF
    """
    try:
        import PyPDF2
        
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            pages_to_read = min(num_pages, max_pages)
            
            text = f"[Document PDF contenant {num_pages} pages, extrait des {pages_to_read} premières pages]\n\n"
            
            for i in range(pages_to_read):
                page = reader.pages[i]
                text += f"--- Page {i+1} ---\n"
                text += page.extract_text() or "[Page sans texte extractible]"
                text += "\n\n"
                
            if num_pages > max_pages:
                text += f"[...{num_pages - max_pages} pages supplémentaires non extraites...]"
                
            return text
    except ImportError:
        return "[Extraction de PDF impossible: PyPDF2 n'est pas installé]"
    except Exception as e:
        return f"[Erreur lors de l'extraction du PDF: {str(e)}]"

def extract_audio_transcription(file_path: str, model: str = "whisper-1") -> str:
    """
    Extrait la transcription d'un fichier audio en utilisant OpenAI Whisper
    
    Args:
        file_path: Chemin vers le fichier audio
        model: Modèle Whisper à utiliser (default: "whisper-1")
        
    Returns:
        Transcription audio
    """
    try:
        # Vérifier que la clé API est disponible
        api_key = os.getenv("OPENAI_API_KEY") or settings.openai_api_key
        if not api_key:
            return "[Erreur: Clé API OpenAI non disponible pour la transcription audio]"
            
        openai.api_key = api_key
        
        # Ouvrir le fichier audio en mode binaire
        with open(file_path, "rb") as audio_file:
            # Appel à l'API Whisper
            logger.info(f"Transcription du fichier audio {file_path} avec le modèle {model}")
            response = openai.Audio.transcribe(model, audio_file)
            
        # Extraire le texte transcrit
        transcription = response["text"]
        
        if not transcription.strip():
            return "[Aucun texte détecté dans l'audio]"
            
        return f"[Transcription audio via Whisper]\n\n{transcription}"
    except ImportError:
        return "[Transcription audio impossible: la bibliothèque openai n'est pas installée]"
    except Exception as e:
        logger.error(f"Erreur lors de la transcription audio: {str(e)}", exc_info=True)
        return f"[Erreur lors de la transcription audio: {str(e)}]"

def extract_video_audio(file_path: str) -> Optional[str]:
    """
    Extrait la piste audio d'une vidéo et la sauvegarde dans un fichier temporaire
    
    Args:
        file_path: Chemin vers le fichier vidéo
        
    Returns:
        Chemin vers le fichier audio extrait ou None en cas d'échec
    """
    try:
        import moviepy.editor as mp
        
        # Créer un nom de fichier temporaire pour l'audio
        audio_path = tempfile.mktemp(suffix='.mp3', dir=MEDIA_CACHE_DIR)
        
        # Charger la vidéo
        logger.info(f"Extraction de l'audio à partir de la vidéo {file_path}")
        video = mp.VideoFileClip(file_path)
        
        # Extraire l'audio
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        
        return audio_path
    except ImportError:
        logger.error("Extraction audio impossible: la bibliothèque moviepy n'est pas installée")
        return None
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction audio de la vidéo: {str(e)}", exc_info=True)
        return None

# ===== Outils LangChain (exposés à l'agent) =====

@register(name="load_media_from_url")
def load_media_from_url(url: str, session_id: Optional[str] = None) -> str:
    """
    Charge un média depuis une URL et retourne son ID pour traitement ultérieur.
    
    Cette fonction télécharge un média (image, document, audio, vidéo) depuis une URL,
    le stocke localement et retourne un ID unique qui permettra de le traiter par la suite
    avec d'autres fonctions comme extract_media_content.
    
    Args:
        url: URL complète du média à télécharger (http://example.com/media.mp4)
        session_id: Identifiant de session optionnel pour associer le média à une conversation
        
    Returns:
        Message de confirmation avec l'ID du média et ses métadonnées, ou message d'erreur
        
    Exemple:
        "Média chargé avec succès! ID: 550e8400-e29b-41d4-a716-446655440000, Type: video, Format: video/mp4"
    """
    metadata = fetch_media_from_url(url, session_id)
    if not metadata:
        return f"Erreur: Impossible de charger le média depuis {url}"
    
    media_info = (
        f"Média chargé avec succès!\n"
        f"ID: {metadata.media_id}\n"
        f"Type: {metadata.media_type}\n"
        f"Format: {metadata.content_type}\n"
        f"Taille: {metadata.size} octets\n\n"
        f"Pour analyser ce média, utilisez l'outil extract_media_content avec l'ID du média."
    )
    
    return media_info

@register(name="list_available_media")
def list_available_media(session_id: Optional[str] = None) -> str:
    """
    Liste tous les médias disponibles téléchargés précédemment.
    
    Cette fonction permet de voir la liste des médias qui ont été téléchargés
    et qui sont disponibles pour être analysés. Vous pouvez filtrer par session si nécessaire.
    
    Args:
        session_id: Optionnel. Si fourni, filtre les médias pour cette session spécifique.
        
    Returns:
        Liste formatée des médias disponibles avec leurs métadonnées, ou message si aucun média.
        
    Exemple:
        "Médias disponibles:
        
        ID: 550e8400-e29b-41d4-a716-446655440000
        Type: image
        URL d'origine: https://example.com/photo.jpg
        Format: image/jpeg
        Taille: 52400 octets
        Date: 2023-05-10 15:30:45"
    """
    media_list = list_media(session_id)
    if not media_list:
        return "Aucun média disponible."
    
    result = "Médias disponibles:\n\n"
    for media in media_list:
        result += (
            f"ID: {media.media_id}\n"
            f"Type: {media.media_type}\n"
            f"URL d'origine: {media.original_url}\n"
            f"Format: {media.content_type}\n"
            f"Taille: {media.size} octets\n"
            f"Date: {media.download_date}\n\n"
        )
    
    return result

@register(name="extract_media_content")
def extract_media_content(media_id: str, max_pages: int = 10) -> str:
    """
    Extrait le contenu textuel de n'importe quel type de média (universel).
    
    Cette fonction détecte automatiquement le type de média et applique 
    le traitement approprié pour en extraire du texte:
    - Documents: extrait le texte avec mise en forme
    - Images: applique l'OCR pour extraire le texte visible
    - Audio: transcrit le contenu audio en texte via Whisper
    - Vidéo: extrait l'audio puis le transcrit en texte
    
    Args:
        media_id: Identifiant unique du média (obtenu via load_media_from_url)
        max_pages: Pour les PDF, nombre maximum de pages à extraire
        
    Returns:
        Contenu textuel extrait du média avec métadonnées, adapté selon son type
        
    Exemple:
        "[Informations sur le média]
        Type: document
        Format: application/pdf
        Taille: 1254000 octets
        URL d'origine: https://example.com/document.pdf
        
        [Document PDF contenant 15 pages, extrait des 10 premières pages]
        
        --- Page 1 ---
        Contenu textuel de la page..."
    """
    media = get_media_metadata(media_id)
    if not media:
        return f"Erreur: Média non trouvé avec l'ID {media_id}"
    
    # Vérifier si déjà traité (cache)
    if media.processed and media.processed_content:
        logger.info(f"Utilisation du contenu en cache pour le média {media_id}")
        return media.processed_content
    
    content = ""
    
    # Traitement selon le type de média
    if media.media_type == "document" or media.content_type.startswith('application/pdf') or media.content_type.startswith('text/'):
        # Pour les documents
        if media.content_type.startswith('application/pdf'):
            content = extract_text_from_pdf(media.local_path, max_pages)
        elif media.content_type.startswith('text/'):
            # Pour les fichiers texte, lire directement
            try:
                with open(media.local_path, 'r', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                content = f"Erreur lors de la lecture du fichier texte: {str(e)}"
    
    elif media.media_type == "image":
        # Pour les images
        content = extract_text_from_image(media.local_path)
        content += "\n\n[Description de l'image: Ce média est une image.]"
    
    elif media.media_type == "audio":
        # Pour les fichiers audio
        content = extract_audio_transcription(media.local_path)
        content += "\n\n[Fichier audio]"
    
    elif media.media_type == "video":
        # Pour les vidéos - utiliser le nouvel extracteur vidéo
        audio_path = extract_video_audio(media.local_path)
        if audio_path:
            content = f"[Média vidéo: {media.content_type}]\n\n"
            content += "Transcription audio de la vidéo:\n\n"
            content += extract_audio_transcription(audio_path)
            
            # Nettoyer le fichier temporaire
            try:
                os.remove(audio_path)
            except:
                pass
        else:
            content = f"[Média vidéo: {media.content_type}]\n\n"
            content += "Impossible d'extraire l'audio de cette vidéo pour transcription."
    
    else:
        content = f"[Type de média non reconnu: {media.media_type} / {media.content_type}]\n\n"
        content += "Le fichier a été téléchargé mais son format n'est pas reconnu pour l'extraction de contenu."
    
    # Limiter la taille du contenu extrait
    if len(content) > 15000:
        content = content[:15000] + "...\n[Contenu tronqué pour rester dans les limites]"
    
    # Ajouter des métadonnées au contenu
    metadata_info = (
        f"[Informations sur le média]\n"
        f"Type: {media.media_type}\n"
        f"Format: {media.content_type}\n"
        f"Taille: {media.size} octets\n"
        f"URL d'origine: {media.original_url}\n\n"
    )
    
    result = metadata_info + content
    
    # Mettre en cache
    media.processed = True
    media.processed_content = result
    
    return result 