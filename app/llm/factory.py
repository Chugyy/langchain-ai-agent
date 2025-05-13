"""
Factory pour la création dynamique de LLMs.
Supporte différents modèles et providers.
"""
from typing import Dict, Any, Optional, Union
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms.openai import OpenAI
from langchain.schema.language_model import BaseLanguageModel
from app.utils.settings import LLMSettings, Settings


class UnsupportedLLMError(Exception):
    """Exception levée lorsqu'un LLM demandé n'est pas supporté."""
    pass


def get_llm(
    name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    openai_api_key: Optional[str] = None,
    **kwargs
) -> BaseLanguageModel:
    """
    Récupère une instance de LLM configurée.
    
    Args:
        name: Nom du modèle LLM (ex: "gpt-4o-mini", "gpt-3.5-turbo")
        temperature: Valeur de température (0.0-1.0)
        max_tokens: Nombre maximum de tokens à générer
        openai_api_key: Clé d'API OpenAI
        **kwargs: Arguments supplémentaires à passer au constructeur
        
    Returns:
        Une instance configurée de LLM
        
    Raises:
        UnsupportedLLMError: Si le LLM demandé n'est pas supporté
    """
    # Normalisation du nom du modèle
    name = name or "gpt-4o-mini"
    name = name.lower().strip()
    
    # Modèles compatibles avec ChatOpenAI (modèle de chat)
    openai_chat_models = [
        "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
        "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
    ]
    
    # Modèles compatibles avec OpenAI (completion)
    openai_completion_models = [
        "text-davinci-003", "text-davinci-002"
    ]
    
    # Configuration du LLM
    if name in openai_chat_models:
        return ChatOpenAI(
            model_name=name,
            temperature=temperature or 0.0,
            max_tokens=max_tokens or 1000,
            openai_api_key=openai_api_key,
            **kwargs
        )
    elif name in openai_completion_models:
        return OpenAI(
            model_name=name,
            temperature=temperature or 0.0,
            max_tokens=max_tokens or 1000,
            openai_api_key=openai_api_key,
            **kwargs
        )
    else:
        raise UnsupportedLLMError(f"Le modèle '{name}' n'est pas pris en charge.")


def get_llm_from_settings(settings: LLMSettings) -> BaseLanguageModel:
    """
    Crée un LLM à partir des paramètres de configuration.
    
    Args:
        settings: Configuration du LLM
        
    Returns:
        Une instance configurée de LLM
    """
    return get_llm(
        name=settings.name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        openai_api_key=settings.api_keys.openai
    ) 