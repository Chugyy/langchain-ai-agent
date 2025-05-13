# Stack Technique et Composants Essentiels

## 1. Environnement et dépendances  
- **Python ≥ 3.9**  
- **LangChain** (>= 0.0.x)  
- **langchain-community** pour `ChatOpenAI`  
- **FastAPI** (ou Flask) pour l'API REST  
- **Uvicorn** (ou Gunicorn + Uvloop) comme serveur ASGI  
- **Pydantic** pour la validation (schemas, settings)  
- **python-dotenv** pour charger `.env`  
- **pytest** (et `pytest-asyncio`) pour les tests  
- **Docker & docker-compose** pour le déploiement  

**Dépendances principales**  
```text
langchain>=0.0.x
langchain-community>=0.1.x
fastapi>=0.85
uvicorn[standard]>=0.18
pydantic>=1.10
pydantic-settings>=2.0
python-dotenv>=0.21
pytest>=7.2
httpx>=0.24.0  # Pour les requêtes HTTP
```

**Dépendances externes**  
```text
# Pour les différents outils
google-auth>=2.22.0  # Authentification Google Calendar
google-api-python-client>=2.0.0  # API Google
unipile>=0.1.0  # Intégration WhatsApp
python-multipart>=0.0.5  # Traitement de fichiers
```

---

## 2. Configuration

* **Variables d'environnement** via `.env`
* Chargement via `pydantic_settings` et `BaseSettings` dans `app/utils/settings.py`
* Champs essentiels :

  ```
  # LLM
  OPENAI_API_KEY=sk-xxx
  LLM_NAME=gpt-4o-mini
  TEMPERATURE=0.0
  MAX_TOKENS=1000
  
  # Mémoire
  MEMORY_TYPE=buffer
  SESSION_TTL_HOURS=24
  
  # Outils
  ENABLED_TOOLS=shout,file_loader,youtube_transcript,whatsapp_send_message,email_send,creer_evenement
  
  # API Externes  
  RAPID_API_KEY=xxx  # Pour YouTube
  UNIPILE_API_KEY=xxx  # Pour WhatsApp
  
  # Email
  EMAIL_SMTP_HOST=smtp.example.com
  EMAIL_USERNAME=user@example.com
  EMAIL_PASSWORD=xxx
  
  # Serveur
  SERVER_HOST=0.0.0.0
  SERVER_PORT=8000
  ```

---

## 3. Structure des modules LangChain

| Module                | Rôle                                                            |
| --------------------- | --------------------------------------------------------------- |
| **llm/factory.py**    | Création dynamique du LLM (`ChatOpenAI`, `OpenAI`, etc.)        |
| **tools/registry.py** | Enregistrement et auto-chargement des fonctions outil           |
| **memory/manager.py** | Instanciation de la mémoire (buffer, summary, vectorstore)      |
| **agents/agent.py**   | Classes d'agent (initialisation, boucle ReAct, gestion mémoire) |
| **api/server.py**     | Exposition des endpoints REST (FastAPI)                         |
| **utils/settings.py** | Chargement des configurations via Pydantic                      |
| **utils/logging.py**  | Configuration du logging structuré                              |
| **schemas/**          | Schémas Pydantic pour validation des entrées/sorties            |

---

## 4. Composants clés pour l'agent LangChain

### 4.1 Instanciation du LLM

```python
from app.llm.factory import get_llm, get_llm_from_settings
from app.utils.settings import get_settings

# Méthode 1: Via settings
settings = get_settings()
llm = get_llm_from_settings(settings.llm)

# Méthode 2: Directement
llm = get_llm(
    name="gpt-4o-mini",
    temperature=0.0,
    max_tokens=1000,
    openai_api_key="sk-xxx"
)
```

### 4.2 Chargement des outils

```python
from app.tools.registry import load_tools, load_all_tools

# Charger tous les outils
all_tools = load_all_tools()

# Charger une sélection d'outils
selected_tools = load_tools([
    "shout", 
    "youtube_transcript", 
    "whatsapp_send_message"
])
```

### 4.3 Gestion de la mémoire

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key='chat_history',
    input_key='input',
    return_messages=True,
    output_key='output'
)
```

### 4.4 Initialisation de l'agent

```python
from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    max_iterations=20,
    early_stopping_method="generate",
    return_intermediate_steps=True
)

# Utilisation
result = agent.invoke({
    "input": "Quel temps fait-il à Paris?",
    "chat_history": [],
    "agent_scratchpad": ""
})
```

---

## 5. Points de vigilance

* **Clés et quotas** : chargez `OPENAI_API_KEY` via `.env` ou secrets manager
* **Sécurité** : validez et restreignez les outils exposés
* **Timeout / Retry** : configurez les timeouts HTTP et les stratégies de retry
* **Observabilité** : loggez chaque appel (prompt, tokens, latence)
* **Tests** : mockez le LLM (`langchain.llms.OpenAI`) pour tests unitaires
* **Rate limiting** : implémentez un rate limiting côté client pour éviter les dépassements de quota