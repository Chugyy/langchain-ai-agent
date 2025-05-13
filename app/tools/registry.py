"""
Registre d'outils pour l'agent IA.
Permet d'enregistrer et charger dynamiquement des outils.
"""
from typing import Dict, List, Callable, Optional, Any, Type
import inspect
import importlib
import os
from functools import wraps
from langchain.tools import BaseTool, StructuredTool
from pydantic import BaseModel, create_model
from app.utils.logging import get_logger

# Logger pour ce module
logger = get_logger(__name__)

# Registre global des outils
_TOOLS_REGISTRY: Dict[str, BaseTool] = {}

# Mapping des schémas pour chaque outil
_SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register(name: Optional[str] = None, description: Optional[str] = None, args_schema: Optional[Type[BaseModel]] = None):
    """
    Décorateur pour enregistrer une fonction comme outil.
    
    Args:
        name: Nom de l'outil (par défaut: nom de la fonction)
        description: Description de l'outil (par défaut: docstring de la fonction)
        args_schema: Schéma Pydantic pour valider les paramètres de l'outil
        
    Returns:
        Décorateur qui enregistre la fonction
    """
    def decorator(func: Callable) -> Callable:
        # Détermination du nom et de la description
        tool_name = name or func.__name__
        tool_description = description or inspect.getdoc(func) or "Aucune description disponible"
        
        # Création d'un schéma par défaut si aucun n'est fourni
        schema_to_use = args_schema
        
        if not schema_to_use:
            # Créer un schéma vide ou basé sur la signature de la fonction
            sig = inspect.signature(func)
            params = {
                name: (
                    param.annotation if param.annotation is not inspect.Parameter.empty else str,
                    ... if param.default is inspect.Parameter.empty else param.default
                )
                for name, param in sig.parameters.items()
                if name != 'self'
            }
            
            # Créer dynamiquement un schéma Pydantic
            schema_name = f"{tool_name.capitalize()}Schema"
            try:
                schema_to_use = create_model(schema_name, **params)
                logger.debug(f"Schéma créé dynamiquement pour '{tool_name}': {schema_name}")
            except Exception as e:
                logger.warning(f"Impossible de créer un schéma dynamique pour '{tool_name}': {e}")
                # Créer un schéma vide comme fallback
                schema_to_use = create_model(schema_name)
        
        # Création de l'outil comme StructuredTool
        logger.debug(f"Création d'un StructuredTool pour '{tool_name}' avec schema: {schema_to_use}")
        tool = StructuredTool.from_function(
            func=func,
            name=tool_name,
            description=tool_description,
            args_schema=schema_to_use
        )
        
        # Enregistrer le schéma pour référence future
        _SCHEMA_REGISTRY[tool_name] = schema_to_use
        
        # Enregistrement dans le registre
        _TOOLS_REGISTRY[tool_name] = tool
        logger.info(f"Outil '{tool_name}' enregistré avec succès comme StructuredTool")
        
        # Retourne la fonction d'origine
        return func
    
    return decorator


def get_tool(name: str) -> Optional[BaseTool]:
    """
    Récupère un outil par son nom.
    
    Args:
        name: Nom de l'outil à récupérer
        
    Returns:
        L'outil demandé ou None s'il n'existe pas
    """
    return _TOOLS_REGISTRY.get(name)


def get_schema(name: str) -> Optional[Type[BaseModel]]:
    """
    Récupère le schéma d'un outil par son nom.
    
    Args:
        name: Nom de l'outil
        
    Returns:
        Le schéma Pydantic associé à l'outil ou None
    """
    return _SCHEMA_REGISTRY.get(name)


def load_tools(tool_names: List[str]) -> List[BaseTool]:
    """
    Charge une liste d'outils à partir de leurs noms.
    
    Args:
        tool_names: Liste des noms d'outils à charger
        
    Returns:
        Liste des outils chargés
    """
    # S'assurer que tous les modules d'outils sont importés
    ensure_tools_imported()
    
    tools = []
    for name in tool_names:
        tool = get_tool(name)
        if tool:
            tools.append(tool)
        else:
            logger.warning(f"Outil '{name}' non trouvé dans le registre")
    
    if not tools:
        logger.warning(f"Aucun outil n'a été chargé parmi {tool_names}")
    else:
        logger.info(f"Outils chargés: {[t.name for t in tools]}")
    
    return tools


def load_all_tools() -> List[BaseTool]:
    """
    Charge tous les outils enregistrés.
    
    Returns:
        Liste de tous les outils
    """
    # S'assurer que tous les modules d'outils sont importés
    ensure_tools_imported()
    
    tools = list(_TOOLS_REGISTRY.values())
    logger.info(f"Tous les outils chargés: {len(tools)} outils")
    return tools


def clear_registry() -> None:
    """
    Efface le registre d'outils.
    Utile pour les tests.
    """
    _TOOLS_REGISTRY.clear()
    _SCHEMA_REGISTRY.clear()
    logger.debug("Registre d'outils effacé")


def ensure_tools_imported() -> None:
    """
    S'assure que tous les modules d'outils sont importés pour enregistrer les outils.
    Cherche et importe dynamiquement tous les fichiers dans le dossier tools.
    """
    if _TOOLS_REGISTRY:
        return  # Les outils sont déjà chargés
    
    # Chemin du dossier des outils
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Parcourir tous les fichiers Python dans le dossier tools
    for filename in os.listdir(tools_dir):
        if filename.endswith('.py') and filename != '__init__.py' and filename != 'registry.py':
            module_name = filename[:-3]  # Enlever l'extension .py
            module_path = f"app.tools.{module_name}"
            
            try:
                # Importer le module pour enregistrer les outils
                importlib.import_module(module_path)
                logger.debug(f"Module d'outils importé: {module_path}")
            except Exception as e:
                logger.error(f"Erreur lors de l'importation du module {module_path}: {str(e)}")
    
    # Log des outils enregistrés
    logger.info(f"Outils enregistrés: {list(_TOOLS_REGISTRY.keys())}")


# Importer automatiquement les outils au chargement du module
ensure_tools_imported() 