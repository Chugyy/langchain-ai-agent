# AI Conversational Agent API

Une API REST pour un agent conversationnel IA facilement intégrable dans des applications tierces.

## Fonctionnalités

- 💬 Conversation avec un agent IA utilisant LangChain
- 🧠 Support pour différents types de mémoire (buffer, summary)
- 🔑 Authentification par clé API et limitation de débit
- 🛠️ Sessions configurables avec différents paramètres de LLM
- 📊 Logging et monitoring complets
- 🐳 Containerisation Docker pour déploiement facile

## Guide de démarrage rapide

### Prérequis

- Python 3.9 ou supérieur
- Une clé API OpenAI

### Installation locale (étape par étape pour débutants)

#### 1. Cloner le dépôt

```bash
git clone https://github.com/yourusername/ai-agent.git
cd ai-agent
```

#### 2. Créer un environnement virtuel

```bash
# Sur macOS/Linux
python -m venv venv
source venv/bin/activate

# Sur Windows
python -m venv venv
venv\Scripts\activate
```

#### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

#### 4. Configurer les variables d'environnement

Créez un fichier `.env` à la racine du projet avec le contenu suivant:

```
# Configuration du LLM
OPENAI_API_KEY=sk-votre-clé-api-openai
LLM_NAME=gpt-4o-mini
TEMPERATURE=0.0
MAX_TOKENS=1000

# Configuration de la mémoire
MEMORY_TYPE=buffer
SESSION_TTL_HOURS=24

# Outils activés (séparés par des virgules, sans espaces)
ENABLED_TOOLS=shout,file_loader,youtube_transcript

# Configuration du serveur
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

> **Important**: Remplacez `sk-votre-clé-api-openai` par votre véritable clé API OpenAI.

#### 5. Démarrer le serveur

```bash
# Mode développement
python -m app.main

# Ou avec uvicorn directement
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 6. Tester l'API

Vous pouvez maintenant tester l'API avec curl ou un outil comme Postman:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer admin-key-for-development" \
  -d '{"message": "Bonjour, comment vas-tu?"}'
```

Le serveur devrait répondre avec quelque chose comme:

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "reply": "Bonjour! Je suis un assistant IA. Comment puis-je vous aider aujourd'hui?",
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 15,
    "total_tokens": 40
  }
}
```

#### 7. Explorer la documentation de l'API

Une fois l'application lancée, visitez `http://localhost:8000/docs` dans votre navigateur pour accéder à la documentation Swagger UI.

### Déploiement avec Docker

#### 1. Construire l'image Docker

```bash
docker build -t ai-agent .
```

#### 2. Lancer le conteneur

```bash
docker run -p 8000:8000 --env-file .env ai-agent
```

Alternativement, utilisez docker-compose:

```bash
docker-compose up -d
```

## Tutoriel: Premier projet d'intégration

Ce tutoriel vous guide dans la création d'une application Python simple qui interagit avec l'API de l'agent IA.

### 1. Créer un nouveau dossier pour votre projet client

```bash
mkdir mon-client-agent-ia
cd mon-client-agent-ia
```

### 2. Créer un environnement virtuel pour ce client

```bash
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sur Windows
```

### 3. Installer les bibliothèques nécessaires

```bash
pip install requests
```

### 4. Créer un fichier Python pour le client

Créez un fichier `client.py` avec le contenu suivant:

```python
import requests
import json

# Configuration
API_URL = "http://localhost:8000"
API_KEY = "admin-key-for-development"  # Utilisez votre vraie clé API en production

# En-têtes pour les requêtes
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def chat_with_agent(message, session_id=None):
    """
    Envoie un message à l'agent et retourne sa réponse.
    
    Args:
        message (str): Le message à envoyer à l'agent
        session_id (str, optional): ID de session pour continuer une conversation
        
    Returns:
        dict: Réponse complète de l'API
    """
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id
        
    response = requests.post(
        f"{API_URL}/chat", 
        headers=headers,
        data=json.dumps(payload)
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Erreur: {response.status_code}")
        print(response.text)
        return None

def main():
    print("Client de démonstration pour l'Agent IA")
    print("Tapez 'quit' pour quitter")
    
    session_id = None  # Pour stocker l'ID de session
    
    while True:
        user_input = input("\nVous: ")
        if user_input.lower() == 'quit':
            break
            
        result = chat_with_agent(user_input, session_id)
        if result:
            # Extraire et stocker l'ID de session pour la continuité
            session_id = result.get("session_id")
            print(f"\nAgent: {result.get('reply')}")
            
            # Afficher des informations sur l'utilisation des tokens (optionnel)
            if "usage" in result:
                usage = result["usage"]
                print(f"\n[Info: {usage.get('total_tokens')} tokens utilisés]")

if __name__ == "__main__":
    main()
```

### 5. Exécuter le client

```bash
python client.py
```

Vous pouvez maintenant discuter avec l'agent IA via cette interface en ligne de commande. Le client maintient automatiquement la session de conversation.

### 6. Améliorations possibles

- Ajouter une gestion des erreurs plus robuste
- Implémenter un mécanisme de sauvegarde des conversations
- Créer une interface graphique avec Tkinter ou une application web avec Flask/FastAPI

## Documentation de l'API

Les principaux endpoints sont:

- `POST /chat` - Envoyer un message à l'agent IA
- `GET /sessions/{session_id}` - Récupérer l'historique d'une session
- `PATCH /sessions/{session_id}/config` - Mettre à jour la configuration d'une session
- `POST /auth/keys` - Générer une nouvelle clé API (admin seulement)
- `GET /health` - Endpoint de contrôle de santé

Pour une documentation complète, consultez les fichiers dans le dossier `/docs` ou l'interface Swagger UI (`/docs`).

## Configuration

La configuration peut être gérée via des variables d'environnement ou le fichier `.env`.

### Options principales

- `OPENAI_API_KEY` - Votre clé API OpenAI
- `LLM_NAME` - Le modèle LLM à utiliser (ex: "gpt-4o-mini")
- `TEMPERATURE` - Réglage de température pour la génération
- `MEMORY_TYPE` - Type de mémoire à utiliser ("buffer" ou "summary")
- `SESSION_TTL_HOURS` - Durée de vie des sessions inactives
- `ENABLED_TOOLS` - Liste des outils activés (séparés par des virgules)
- `ADMIN_API_KEY` - Clé API admin pour les opérations administratives

## Structure du projet

```
app/
├── agents/          # Implémentation de l'agent
├── api/             # Endpoints API
├── config/          # Configuration
├── llm/             # Factory LLM
├── memory/          # Gestion de la mémoire
├── tools/           # Outils de l'agent
└── utils/           # Utilitaires (settings, logging, etc.)
```

## Tests

Pour exécuter les tests:

```bash
pytest
```

## Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.