# Documentation de l'API – Agent IA Conversationnel

## Table des matières
1. [Introduction](#introduction)  
2. [Authentification](#authentification)  
3. [Gestion des sessions](#gestion-des-sessions)  
4. [Endpoints REST](#endpoints-rest)  
   - [POST /chat](#1-post-chat)  
   - [GET /sessions/{session_id}](#2-get-sessionssession_id)  
   - [PATCH /sessions/{session_id}/config](#3-patch-sessionssession_idconfig)  
   - [POST /auth/keys](#4-post-authkeys)  
5. [Configuration du LLM et des outils](#configuration-du-llm-et-des-outils)  
6. [Bonnes pratiques d'intégration](#bonnes-pratiques-dintégration)  
7. [Outils disponibles](#outils-disponibles)
8. [Gestion des erreurs](#gestion-des-erreurs)

---

## Introduction
Cette API expose un agent IA conversationnel simple à intégrer.  
- **Objectif** : permettre à une application tierce d'envoyer des messages à l'agent et de recevoir des réponses.  
- **Formats** : JSON en requête et en réponse.  
- **Authentification** : clé API via en-tête HTTP `Authorization: Bearer <API_KEY>`.

---

## Authentification
Toutes les requêtes (hors enregistrement de clés) nécessitent un en-tête HTTP :

```http
Authorization: Bearer <VOTRE_CLE_API>
Content-Type: application/json
````

* **Erreur 401** : clé manquante ou invalide.
* **Erreur 403** : accès refusé (quota atteint ou clé désactivée).

---

## Gestion des sessions

L'agent conserve le contexte de la conversation par **session**.

* Chaque session dispose d'un identifiant UUID.
* Vous pouvez fournir :

  * Tout l'historique dans chaque appel (champ `history`),
  * Ou uniquement le `session_id`, laissant le serveur gérer le contexte stored.

### Cycle de vie

1. **Démarrage** : premier appel à `/chat` renvoie un `session_id`.
2. **Conversation** : appels ultérieurs à `/chat` incluent le même `session_id`.
3. **Consultation** : récupérer l'état et l'historique via `GET /sessions/{session_id}`.
4. **Configuration** : ajuster modèle et paramètres via `PATCH /sessions/{session_id}/config`.
5. **Fermeture** : le serveur purge automatiquement les sessions inactives après X heures, configurable.

---

## Endpoints REST

### 1. POST `/chat`

Échanger un message avec l'agent.

* **URL** : `/chat`

* **Méthode** : `POST`

* **Headers** :

  ```http
  Authorization: Bearer <API_KEY>
  Content-Type: application/json
  ```

* **Corps (JSON)** :

  | Champ       | Type      | Obligatoire | Description                                                   |
  | ----------- | --------- | ----------- | ------------------------------------------------------------- |
  | message     | string    | oui         | Texte envoyé à l'agent.                                       |
  | session\_id | string    | non         | UUID de la session existante (si omis, nouvelle session).     |
  | temperature | float     | non         | Température du LLM pour cet appel (écrase la config session). |
  | tools       | \[string] | non         | Liste d'outils à charger pour cet appel.                      |

* **Exemple de requête** :

  ```http
  POST /chat HTTP/1.1
  Authorization: Bearer abc123
  Content-Type: application/json

  {
    "message": "Bonjour, pouvez-vous me résumer la réunion d'hier ?"
  }
  ```

* **Exemple de réponse** :

  ```json
  {
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "reply": "Bien sûr ! Hier, nous avons discuté : ...",
    "usage": {
      "prompt_tokens": 42,
      "completion_tokens": 85,
      "total_tokens": 127
    }
  }
  ```

* **Codes de statut** :

  * `200 OK`
  * `400 Bad Request` (payload manquant ou mal formé)
  * `401 Unauthorized`
  * `429 Too Many Requests` (rate limit)

---

### 2. GET `/sessions/{session_id}`

Récupérer l'historique et la configuration d'une session.

* **URL** : `/sessions/{session_id}`
* **Méthode** : `GET`
* **Headers** : `Authorization`
* **Réponse (200)** :

  ```json
  {
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "created_at": "2025-05-07T08:00:00Z",
    "last_interaction": "2025-05-07T08:15:30Z",
    "config": {
      "model": "gpt-4o-mini",
      "temperature": 0.0,
      "tools": ["shout", "file_loader", "youtube_transcript"]
    },
    "history": [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Bonjour !"}
    ]
  }
  ```

---

### 3. PATCH `/sessions/{session_id}/config`

Modifier les paramètres d'une session en cours.

* **URL** : `/sessions/{session_id}/config`

* **Méthode** : `PATCH`

* **Corps (JSON)** :

  | Champ       | Type      | Description                                  |
  | ----------- | --------- | -------------------------------------------- |
  | model       | string    | Nom du LLM (`gpt-4o-mini`, `gpt-3.5-turbo`…) |
  | temperature | float     | Température (0.0–1.0)                        |
  | tools       | \[string] | Liste d'outils disponibles                   |

* **Exemple** :

  ```http
  PATCH /sessions/550e8400-e29b-41d4-a716-446655440000/config HTTP/1.1
  Authorization: Bearer abc123
  Content-Type: application/json

  {
    "temperature": 0.7,
    "tools": ["shout", "youtube_transcript"]
  }
  ```

* **Réponse (200)** : retourne la nouvelle configuration.

---

### 4. POST `/auth/keys`

Gérer vos clés API (création).

* **URL** : `/auth/keys`

* **Méthode** : `POST`

* **Corps (JSON)** :

  | Champ  | Type      | Description                    |
  | ------ | --------- | ------------------------------ |
  | name   | string    | Nom descriptif de la clé       |
  | scopes | \[string] | Permissions (ex. `chat:write`) |

* **Exemple** :

  ```http
  POST /auth/keys HTTP/1.1
  Content-Type: application/json

  {
    "name": "IntegrationApp",
    "scopes": ["chat:write", "sessions:read"]
  }
  ```

* **Réponse (201)** :

  ```json
  {
    "key_id": "key_01F...",
    "api_key": "sk-xxxx",
    "scopes": ["chat:write","sessions:read"]
  }
  ```

---

## Configuration du LLM et des outils

* **Par défaut** : paramètres dans `settings.py` ou variables d'environnement ;
* **À la volée** : via le champ `temperature` ou `tools` dans `/chat` ou `PATCH /sessions/.../config`.
* **Outils externes** : pré-enregistrer vos fonctions (ex. `shout`, `file_loader`) lors de l'initialisation du serveur puis exposer leur nom aux clients.

### Configuration des modèles de langage

Les modèles suivants sont supportés par défaut:

| Type | Modèles disponibles |
|------|---------------------|
| Chat | gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini, gpt-3.5-turbo, gpt-3.5-turbo-16k |
| Completion | text-davinci-003, text-davinci-002 |

Pour ajouter des modèles supplémentaires, consultez la documentation [`AddLLM.md`](/docs/AddLLM.md).

---

## Bonnes pratiques d'intégration

1. **Gestion d'erreurs** : parsez `error.code` et `error.message` pour réagir (retry, back-off).
2. **Rate limiting client** : throttle vos appels pour éviter `429`.
3. **Pool de sessions** : réutilisez `session_id` pour conserver le contexte.
4. **Logs et monitoring** : instrumentez chaque appel (latence, tokens consommés).
5. **Sécurité** : protégez votre clé API côté client, utilisez HTTPS.
6. **Timeout et retry** : paramétrez un timeout adapté, retry en cas d'erreur transitoire.

---

## Outils disponibles

L'API propose plusieurs outils intégrés qui peuvent être activés par session:

### shout
Transforme un texte en majuscules et ajoute des points d'exclamation.
- **Paramètres** : `text` (string) - Le texte à transformer
- **Exemple** : "hello" → "HELLO!!!"

### file_loader
Charge le contenu d'un fichier texte.
- **Paramètres** : `file_path` (string) - Chemin vers le fichier à charger

### youtube_transcript
Extrait la transcription d'une vidéo YouTube.
- **Paramètres** : `video_url` (string) - URL de la vidéo YouTube

### extract_media_content
Analyse le contenu d'un fichier média (document, image, audio, vidéo).
- **Paramètres** : `media_id` (string) - Identifiant du média à analyser

### whatsapp_send_message
Envoie un message WhatsApp.
- **Paramètres** : 
  - `recipient` (string) - Numéro du destinataire
  - `message` (string) - Contenu du message

### email_send
Envoie un email.
- **Paramètres** :
  - `to` (string) - Adresse email du destinataire
  - `subject` (string) - Objet de l'email
  - `body` (string) - Corps de l'email

### creer_evenement
Crée un événement dans Google Calendar.
- **Paramètres** :
  - `summary` (string) - Titre de l'événement
  - `start_time` (string) - Date et heure de début (format ISO)
  - `end_time` (string) - Date et heure de fin (format ISO)
  - `description` (string, optionnel) - Description de l'événement

Pour voir la liste complète des outils et apprendre à en créer de nouveaux, consultez [`CreateATool.md`](/docs/CreateATool.md).

## Gestion des erreurs

### Codes d'erreur communs

| Code HTTP | Signification | Cause possible | Action recommandée |
|-----------|---------------|----------------|-------------------|
| 400 | Bad Request | Requête mal formée | Vérifier le format JSON et les champs obligatoires |
| 401 | Unauthorized | Clé API manquante ou invalide | Vérifier la clé API |
| 403 | Forbidden | Accès refusé (permissions) | Vérifier les scopes de la clé API |
| 404 | Not Found | Session inexistante | Vérifier le session_id |
| 429 | Too Many Requests | Rate limit dépassé | Implémenter exponential backoff |
| 500 | Internal Server Error | Erreur côté serveur | Reporter le problème et réessayer plus tard |

### Format des erreurs

Les erreurs sont retournées au format JSON:

```json
{
  "error": {
    "code": "invalid_session_id",
    "message": "La session spécifiée n'existe pas ou a expiré",
    "status": 404
  }
}
```

### Bonnes pratiques

1. **Retry automatique** pour les erreurs 429 et 5xx avec exponential backoff
2. **Logging** des erreurs côté client pour faciliter le débogage
3. **Gestion gracieuse** des erreurs côté utilisateur final