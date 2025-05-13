## Système existant et structure

Votre système actuel utilise un registre d'outils (`registry.py`) qui permet l'enregistrement automatique des outils via le décorateur `@register`. La fonction `ensure_tools_imported()` importe dynamiquement tous les modules Python dans le dossier `app/tools/`.

## Comment ajouter facilement de nouveaux outils

### 1. Créer un nouveau fichier d'outil

Pour ajouter un nouvel outil, créez simplement un fichier Python dans le dossier `app/tools/`. Par exemple:

```
app/tools/weather_tools.py
```

### 2. Définir la validation des arguments avec Pydantic

Avant de créer l'outil, il est recommandé de définir un schéma de validation pour les arguments:

```python
# Dans un fichier schemas/weather.py
from typing import Optional
from pydantic import BaseModel, Field

class GetWeatherSchema(BaseModel):
    """Schéma pour récupérer la météo."""
    location: str = Field(..., description="Nom de la ville ou coordonnées")
    units: str = Field("metric", description="Unités (metric/imperial)")
    language: Optional[str] = Field("fr", description="Code de langue")
```

### 3. Créer la fonction d'outil et l'enregistrer

Dans votre fichier d'outil, importez le décorateur `@register` et le schéma, puis définissez votre fonction:

```python
from app.tools.registry import register
from app.utils.settings import get_settings
from app.utils.logging import get_logger
from app.schemas.weather import GetWeatherSchema

logger = get_logger(__name__)
settings = get_settings()

@register(name="get_weather", args_schema=GetWeatherSchema)
def get_weather(location: str, units: str = "metric", language: str = "fr") -> str:
    """
    Récupère les informations météo pour une localisation donnée.
    
    Args:
        location: Nom de la ville ou coordonnées
        units: Unités de mesure (metric/imperial)
        language: Code de langue pour la réponse
        
    Returns:
        Les informations météo ou un message d'erreur
    """
    logger.info(f"Récupération de la météo pour {location}")
    
    try:
        # Implémentation de l'outil...
        return f"Météo pour {location}: ..."
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la météo: {str(e)}", exc_info=True)
        return f"Erreur: {str(e)}"
```

### 4. Utiliser des mécanismes de retry pour les opérations réseau

Pour les outils qui dépendent d'API externes ou d'opérations réseau, il est recommandé d'utiliser un décorateur de retry:

```python
from app.tools.utils import with_retry

@register(name="get_weather", args_schema=GetWeatherSchema)
@with_retry(max_retries=3, delay=1)
def get_weather(location: str, units: str = "metric", language: str = "fr") -> str:
    # Implémentation...
```

Exemple de décorateur de retry:

```python
import functools
import time
import requests
from app.utils.logging import get_logger

logger = get_logger(__name__)

def with_retry(max_retries: int = 3, delay: float = 1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            last_exception = None
            
            while True:
                try:
                    return func(*args, **kwargs)
                except requests.HTTPError as e:
                    # Ne pas réessayer pour certains codes d'erreur
                    status_code = e.response.status_code if hasattr(e, 'response') else 0
                    
                    # Ne pas réessayer pour les erreurs d'authentification ou de validation
                    if status_code in (400, 401, 403, 422):
                        logger.error(f"Erreur HTTP {status_code} non réessayable: {e}")
                        raise
                    
                    retries += 1
                    last_exception = e
                    
                    if retries > max_retries:
                        logger.error(f"Échec après {max_retries} tentatives - Erreur HTTP {status_code}: {e}")
                        raise
                        
                    logger.warning(f"Tentative {retries}/{max_retries} échouée - Erreur HTTP {status_code}: {e}")
                    # Attendre plus longtemps entre les tentatives (backoff exponentiel)
                    time.sleep(delay * (2 ** (retries - 1)))
                    
                except (requests.ConnectionError, requests.Timeout) as e:
                    # Toujours réessayer les erreurs de connexion
                    retries += 1
                    last_exception = e
                    
                    if retries > max_retries:
                        logger.error(f"Échec après {max_retries} tentatives - Erreur de connexion: {e}")
                        raise
                        
                    logger.warning(f"Tentative {retries}/{max_retries} échouée - Erreur de connexion: {e}")
                    time.sleep(delay * (2 ** (retries - 1)))
                    
                except Exception as e:
                    # Pour les autres exceptions
                    retries += 1
                    last_exception = e
                    
                    if retries > max_retries:
                        logger.error(f"Échec après {max_retries} tentatives: {e}")
                        raise
                        
                    logger.warning(f"Tentative {retries}/{max_retries} échouée: {e}")
                    time.sleep(delay)
                    
        return wrapper
    return decorator
```

### 5. Ajouter le nom de l'outil aux outils activés

Dans `settings.py`, la configuration de base des outils activés est:

```python
enabled: List[str] = Field(default_factory=lambda: [
    "shout", 
    "file_loader", 
    "youtube_transcript",
    # Votre nouvel outil
    "get_weather"
], env="ENABLED_TOOLS")
```

Vous pouvez:
- Ajouter votre nouvel outil à cette liste par défaut
- Le configurer via une variable d'environnement `ENABLED_TOOLS`
- L'activer dynamiquement lors de la création d'une session

### 6. Tester votre outil

Créez un test simple pour vérifier que votre outil fonctionne correctement:

```python
# tests/test_weather_tool.py
from app.tools.weather_tools import get_weather

def test_get_weather():
    result = get_weather(location="Paris")
    assert "Météo pour Paris" in result
    assert "Erreur" not in result
```

## Bonnes pratiques pour la création d'outils

### Gestion des erreurs robuste

Toujours implémenter une gestion d'erreur complète:

```python
try:
    # Code principal
    return "Résultat"
except SpecificError as e:
    logger.error(f"Erreur spécifique: {str(e)}")
    return f"Erreur spécifique: {str(e)}"
except Exception as e:
    logger.error(f"Erreur inattendue: {str(e)}", exc_info=True)
    return f"Une erreur inattendue s'est produite: {str(e)}"
```

### Logging approprié

Utilisez différents niveaux de logging:
- `logger.debug()` pour les informations détaillées de débogage
- `logger.info()` pour les informations générales sur l'exécution
- `logger.warning()` pour les situations anormales mais non critiques
- `logger.error()` pour les erreurs qui empêchent le fonctionnement normal
- `logger.critical()` pour les erreurs graves qui peuvent causer un arrêt du système

### Validation des configurations

Pour les outils qui nécessitent des configurations externes (API, connexions), validez leur présence:

```python
def my_tool():
    api_key = settings.api_keys.get("service_name")
    if not api_key:
        logger.error("Clé API manquante pour service_name")
        return "Erreur: Configuration incomplète. Définissez SERVICE_API_KEY dans les variables d'environnement."
```

## Amélioration pour une gestion encore plus scalable

Pour rendre le système encore plus flexible pour vous et l'IA, voici quelques recommandations:

### 1. Organiser les outils par catégorie

Créez des sous-dossiers thématiques dans `app/tools/`:
```
app/tools/
  ├── media/
  │   ├── youtube_tools.py
  │   ├── twitter_tools.py
  │   └── ...
  ├── data/
  │   ├── file_tools.py
  │   └── database_tools.py
  └── web/
      ├── search_tools.py
      └── weather_tools.py
```

### 2. Implémenter un système de métadonnées pour les outils

Modifiez le décorateur `@register` pour accepter des métadonnées supplémentaires:

```python
@register(
    name="youtube_transcript",
    category="media",
    requires_api_key=True,
    cost_per_call=0.01
)
```

### 3. Créer une interface de gestion des outils

Développez un endpoint d'API qui liste tous les outils disponibles avec leurs métadonnées, permettant:
- De voir quels outils sont disponibles
- De connaître les paramètres requis pour chaque outil
- D'activer/désactiver des outils par catégorie

### 4. Automatiser la documentation des outils

Générez automatiquement une documentation à partir des docstrings et métadonnées des outils:
```
GET /tools/documentation
```

## Guide pour l'IA - Comment ajouter un nouvel outil

Pour que l'IA puisse facilement ajouter de nouveaux outils, voici un template simple à suivre:

1. Identifier le besoin d'un nouvel outil
2. Déterminer la catégorie appropriée (média, data, web, etc.)
3. Créer les schémas de validation des arguments dans le dossier `app/schemas/`
4. Créer un fichier dans le dossier correspondant avec un nom descriptif (`<catégorie>_tools.py`)
5. Implémenter la fonction avec:
   - Des paramètres clairs et typés
   - Une docstring complète (description, paramètres, valeur de retour)
   - Gestion des erreurs robuste avec try/except
   - Logging approprié à chaque étape
   - Mécanisme de retry pour les opérations réseau si nécessaire
6. Décorer la fonction avec `@register` et référencer le schéma des arguments

## Exemple concret d'ajout d'un outil complexe

Voici un exemple complet d'un outil d'envoi d'email:

```python
# app/schemas/communication.py
from typing import Optional, List
from pydantic import BaseModel, Field

class EmailSendSchema(BaseModel):
    """Schéma pour envoyer un email."""
    to: List[str] = Field(..., description="Adresse(s) email du/des destinataire(s)")
    subject: str = Field(..., description="Sujet de l'email")
    body: str = Field(..., description="Corps du message (supporte le HTML)")
    cc: Optional[List[str]] = Field(None, description="Adresse(s) email en copie")
    bcc: Optional[List[str]] = Field(None, description="Adresse(s) email en copie cachée")
    attachments: Optional[List[str]] = Field(None, description="Chemins des pièces jointes")
```

```python
# app/tools/communication_tools.py
from app.tools.registry import register
from app.utils.settings import get_settings
from app.utils.logging import get_logger
from app.schemas.communication import EmailSendSchema
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.utils import formataddr
import os
from typing import List, Optional, Dict, Any

logger = get_logger(__name__)
settings = get_settings()

class EmailClient:
    """Client pour l'envoi d'emails via SMTP."""
    
    def __init__(
        self, 
        username: str,
        password: str,
        smtp_host: str,
        smtp_port: int,
        sender_name: Optional[str],
        use_tls: bool
    ):
        self.username = username
        self.password = password
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.sender_name = sender_name
        self.use_tls = use_tls
        
    def get_formatted_sender(self) -> str:
        """Retourne l'adresse d'expéditeur correctement formatée."""
        if self.sender_name:
            return formataddr((self.sender_name, self.username))
        return self.username
    
    def send_email(
        self, 
        to: List[str], 
        subject: str, 
        body: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        attachments: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Envoie un email via SMTP."""
        if not to:
            logger.error("Aucun destinataire spécifié")
            return {"success": False, "error": "Aucun destinataire spécifié"}
        
        try:
            # Création du message
            msg = MIMEMultipart()
            msg["From"] = self.get_formatted_sender()
            msg["To"] = ", ".join(to)
            msg["Subject"] = subject
            
            if cc:
                msg["Cc"] = ", ".join(cc)
                
            # Détection du type de contenu (HTML ou texte)
            if "<html" in body.lower() or "<body" in body.lower() or "<div" in body.lower() or "<br" in body.lower():
                msg.attach(MIMEText(body, "html"))
            else:
                msg.attach(MIMEText(body, "plain"))
            
            # Ajout des pièces jointes
            if attachments:
                for file_path in attachments:
                    if not os.path.exists(file_path):
                        logger.warning(f"Pièce jointe introuvable: {file_path}")
                        continue
                    
                    filename = os.path.basename(file_path)
                    with open(file_path, "rb") as file:
                        part = MIMEApplication(file.read(), Name=filename)
                    
                    part["Content-Disposition"] = f'attachment; filename="{filename}"'
                    msg.attach(part)
            
            # Liste complète des destinataires
            all_recipients = to[:]
            if cc: all_recipients.extend(cc)
            if bcc: all_recipients.extend(bcc)
            
            # Connexion au serveur SMTP avec le bon protocole selon le port
            if self.smtp_port == 465:  # Port SSL standard
                with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=30) as server:
                    server.login(self.username, self.password)
                    server.send_message(msg, from_addr=self.username, to_addrs=all_recipients)
            else:
                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as server:
                    if self.use_tls:
                        server.starttls()
                    server.login(self.username, self.password)
                    server.send_message(msg, from_addr=self.username, to_addrs=all_recipients)
            
            return {
                "success": True, 
                "to": to,
                "cc": cc or [],
                "bcc": bcc or [],
                "subject": subject
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de l'email: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

def _get_email_client():
    """Instancie le client Email avec la configuration."""
    # Validation des paramètres obligatoires
    if not settings.email.username or not settings.email.password:
        raise ValueError("Identifiants email manquants")
    
    return EmailClient(
        username=settings.email.username,
        password=settings.email.password,
        smtp_host=settings.email.smtp_host,
        smtp_port=settings.email.smtp_port,
        sender_name=settings.email.sender_name,
        use_tls=settings.email.use_tls
    )

@register(name="email_send", args_schema=EmailSendSchema)
@with_retry(max_retries=2, delay=1)
def email_send(
    to: List[str],
    subject: str,
    body: str,
    cc: Optional[List[str]] = None,
    bcc: Optional[List[str]] = None,
    attachments: Optional[List[str]] = None
) -> str:
    """
    Envoie un email via SMTP.
    Args:
        to: Liste des adresses email des destinataires.
        subject: Sujet de l'email.
        body: Corps du message (peut contenir du HTML).
        cc: Liste des adresses email en copie.
        bcc: Liste des adresses email en copie cachée.
        attachments: Liste des chemins vers les pièces jointes.
    Returns:
        Confirmation ou message d'erreur.
    """
    # Validation des paramètres
    if not to:
        return "Erreur: Au moins un destinataire est requis."
    
    if not subject or not body:
        return "Erreur: Le sujet et le corps du message sont requis."
    
    try:
        client = _get_email_client()
        result = client.send_email(to, subject, body, cc, bcc, attachments)
        
        if result["success"]:
            return f"Email envoyé avec succès à {len(to)} destinataire(s)."
        else:
            return f"Erreur: {result['error']}"
            
    except ValueError as e:
        return f"Erreur de configuration: {str(e)}"
        
    except Exception as e:
        return f"Erreur lors de l'envoi de l'email: {str(e)}"
```

Cette approche vous permet d'ajouter facilement autant d'outils que vous le souhaitez, tout en maintenant une architecture propre et évolutive.