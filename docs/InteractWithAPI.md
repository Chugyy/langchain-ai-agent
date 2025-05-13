D'après le fichier `app/api/server.py`, voici les principales commandes pour interagir avec le LLM via l'API:

## Commandes API pour interagir avec le LLM

### 1. Envoyer un message à l'agent
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <VOTRE_CLE_API>" \
  -d '{
    "message": "Votre message ici",
    "session_id": "id-session-optionnel"
  }'
```

### 2. Consulter l'historique d'une session
```bash
curl -X GET http://localhost:8000/sessions/<SESSION_ID> \
  -H "Authorization: Bearer <VOTRE_CLE_API>"
```

### 3. Modifier la configuration d'une session
```bash
curl -X PATCH http://localhost:8000/sessions/<SESSION_ID>/config \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <VOTRE_CLE_API>" \
  -d '{
    "temperature": 0.7,
    "model_name": "gpt-4o",
    "memory_type": "buffer"
  }'
```

### 4. Créer une nouvelle clé API (admin uniquement)
```bash
curl -X POST http://localhost:8000/auth/keys \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <CLE_ADMIN>" \
  -d '{
    "scopes": ["chat", "sessions:read"],
    "expires_in_days": 30,
    "rate_limit": 100
  }'
```

### 5. Vérifier la santé de l'API
```bash
curl -X GET http://localhost:8000/health
```

### Commandes de débogage (admin uniquement)

#### Obtenir l'état complet d'un agent
```bash
curl -X GET http://localhost:8000/debug/sessions/<SESSION_ID> \
  -H "Authorization: Bearer <CLE_ADMIN>"
```

#### Activer/désactiver le mode verbeux
```bash
curl -X PUT http://localhost:8000/debug/sessions/<SESSION_ID>/verbose/true \
  -H "Authorization: Bearer <CLE_ADMIN>"
```

#### Consulter le prompt template de l'agent
```bash
curl -X GET http://localhost:8000/debug/sessions/<SESSION_ID>/prompt \
  -H "Authorization: Bearer <CLE_ADMIN>"
```

#### Activer/désactiver le monitoring HTTP
```bash
curl -X PUT http://localhost:8000/debug/monitor/http/<SESSION_ID>/true \
  -H "Authorization: Bearer <CLE_ADMIN>"
```

Pour toutes ces commandes, remplacez `<VOTRE_CLE_API>`, `<CLE_ADMIN>` et `<SESSION_ID>` par vos valeurs réelles.
