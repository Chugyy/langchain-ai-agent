# Guide d'ajout d'un nouveau modèle LLM

Ce guide explique comment ajouter et configurer un nouveau modèle de langage (LLM) dans le système.

## Structure existante

Le système utilise une factory pattern (`app/llm/factory.py`) pour créer dynamiquement les instances de LLM. Cette approche permet:
- Un changement flexible de modèle
- La configuration centralisée des paramètres
- La gestion unifiée des erreurs

## Étapes pour ajouter un nouveau LLM

### 1. Mettre à jour la liste des modèles supportés

Dans `app/llm/factory.py`, repérez les listes de modèles supportés et ajoutez votre nouveau modèle dans la liste appropriée:

```python
# Pour les modèles de chat (ChatOpenAI)
openai_chat_models = [
    "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
    "gpt-3.5-turbo", "gpt-3.5-turbo-16k",
    "votre-nouveau-modele"  # Ajoutez votre modèle ici
]

# Pour les modèles de complétion (OpenAI)
openai_completion_models = [
    "text-davinci-003", "text-davinci-002",
    "votre-nouveau-modele-completion"  # Ou ici pour un modèle de complétion
]
```

### 2. Ajouter une nouvelle classe d'implémentation (si nécessaire)

Si votre nouveau modèle provient d'un fournisseur différent (pas OpenAI), ajoutez l'intégration dans `app/llm/factory.py`:

```python
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms.openai import OpenAI
# Ajoutez l'import pour votre nouveau provider
from langchain_community.chat_models import ChatAnthropic  # Exemple avec Anthropic

# Plus bas dans le code, ajoutez la condition
elif name.startswith("claude-"):
    return ChatAnthropic(
        model_name=name,
        temperature=temperature or 0.0,
        max_tokens=max_tokens or 1000,
        anthropic_api_key=anthropic_api_key,
        **kwargs
    )
```

### 3. Mettre à jour les paramètres de configuration

Dans `app/utils/settings.py`, ajoutez la nouvelle clé API si nécessaire:

```python
class ApiKeysSettings(BaseSettings):
    """Configuration centralisée des clés API pour les différents services."""
    openai: Optional[str] = None
    anthropic: Optional[str] = None  # Ajoutez la nouvelle clé API
    # ...
```

### 4. Mettre à jour le chargement des variables d'environnement

Toujours dans `app/utils/settings.py`, dans la fonction `get_settings()`:

```python
def get_settings() -> Settings:
    # ...
    settings.api_keys.openai = os.getenv('OPENAI_API_KEY')
    settings.api_keys.anthropic = os.getenv('ANTHROPIC_API_KEY')  # Ajoutez la nouvelle clé
    # ...
```

### 5. Mettre à jour la fonction get_llm

Si nécessaire, modifiez la signature de la fonction `get_llm` pour accepter le nouveau paramètre:

```python
def get_llm(
    name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,  # Nouveau paramètre
    **kwargs
) -> BaseLanguageModel:
```

### 6. Tester le nouveau modèle

Créez un test simple pour valider l'intégration:

```python
def test_new_llm_integration():
    # Configurer l'environnement pour le test
    os.environ['ANTHROPIC_API_KEY'] = 'test-key'
    
    # Tester la création du modèle
    llm = get_llm(name="claude-3-opus")
    
    # Vérifier le type
    assert isinstance(llm, ChatAnthropic)
    
    # Vérifier les paramètres
    assert llm.model_name == "claude-3-opus"
    assert llm.temperature == 0.0
```

## Exemple complet d'ajout d'un nouveau provider

Voici un exemple complet pour intégrer les modèles d'Anthropic:

```python
from langchain_community.chat_models import ChatAnthropic

# Ajouter la liste des modèles supportés
anthropic_models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]

# Dans la fonction get_llm:
elif name in anthropic_models:
    # Vérifier que la clé API est disponible
    if not anthropic_api_key:
        raise ValueError("Anthropic API key is required for Claude models")
        
    return ChatAnthropic(
        model_name=name,
        temperature=temperature or 0.0,
        max_tokens=max_tokens or 1000,
        anthropic_api_key=anthropic_api_key,
        **kwargs
    )
```

## Considérations importantes

1. **Compatibilité des interfaces**: Assurez-vous que le nouveau modèle implémente correctement l'interface `BaseLanguageModel` de LangChain
2. **Gestion des erreurs**: Implémentez une gestion appropriée des erreurs spécifiques au nouveau fournisseur
3. **Documentation**: Mettez à jour la documentation pour inclure le nouveau modèle
4. **Variables d'environnement**: Documentez les nouvelles variables d'environnement requises
