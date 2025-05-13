"""
Module principal de l'agent IA conversationnel.
Initialise et gère l'agent avec LangChain.
"""
import uuid
import json
import inspect
from typing import Dict, List, Optional, Any
from langchain.agents import initialize_agent, AgentType
from langchain.schema.language_model import BaseLanguageModel
from langchain.memory.chat_memory import BaseChatMemory
from langchain.tools import BaseTool
from pprint import pprint
from app.llm.factory import get_llm_from_settings
from app.tools.registry import load_tools, load_all_tools
from app.memory.manager import MemoryManager, MemoryStorage
from app.utils.settings import LLMSettings, MemorySettings, ToolsSettings, get_settings
from app.utils.logging import get_logger
from datetime import datetime, timedelta
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import traceback
import logging
from types import MethodType
from langchain.callbacks.manager import CallbackManagerForChainRun

# Activer le logger LangChain en DEBUG pour obtenir les traces détaillées
logging.getLogger("langchain").setLevel(logging.DEBUG)

# Logger pour ce module
logger = get_logger(__name__)

# Stockage global des sessions
memory_storage = MemoryStorage()

# Stockage des agents par session_id
AGENT_INSTANCES = {}
# Stockage des configurations de session
SESSION_CONFIGS = {}
# Stockage des métadonnées de session
SESSION_META = {}

settings = get_settings()

class Agent:
    """
    Agent conversationnel avec mémoire et outils.
    Gère une conversation continue avec l'utilisateur.
    """
    
    def __init__(self, session_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialise un agent avec une configuration spécifique.
        
        Args:
            session_id: Identifiant de la session
            config: Configuration personnalisée (écrase les valeurs par défaut)
        """
        self.session_id = session_id
        self.config = config or {}
        
        # Initialisation du modèle de langage
        self._init_llm()
        
        # Chargement des outils
        self._init_tools()
        
        # Initialisation de la mémoire
        self._init_memory()
        
        # Création de l'agent
        self._init_agent()
        
        logger.info(f"Agent initialisé pour la session {session_id}")
        
    def _init_llm(self):
        """Initialise le modèle de langage."""
        model_name = self.config.get('model_name', settings.llm.name)
        temperature = self.config.get('temperature', settings.llm.temperature)
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=settings.api_keys.openai
        )
        
        logger.debug(f"LLM initialisé: {model_name}, température: {temperature}")
        
    def _init_tools(self):
        """Charge les outils disponibles pour l'agent."""
        # Charge tous les outils disponibles ou une liste spécifique
        tool_names = self.config.get('tools', settings.tools.enabled)
        
        # Vérifier si tool_names est déjà une liste d'objets Tool
        if tool_names and isinstance(tool_names, list) and all(isinstance(item, str) for item in tool_names):
            # C'est une liste de noms, utiliser load_tools
            self.tools = load_tools(tool_names)
            logger.debug(f"Outils chargés par nom: {', '.join(tool_names)}")
        elif tool_names and isinstance(tool_names, list) and hasattr(tool_names[0], 'name'):
            # C'est déjà une liste d'outils, utiliser directement
            self.tools = tool_names
            logger.debug(f"Outils déjà chargés: {', '.join([t.name for t in self.tools])}")
        else:
            # Charger tous les outils
            self.tools = load_all_tools()
            logger.debug("Tous les outils ont été chargés")
            
        # Important: logger les outils disponibles pour le débogage
        tool_names = [tool.name for tool in self.tools]
        logger.info(f"Outils disponibles pour l'agent: {tool_names}")

        # Log détaillé des descriptions d'outils
        tool_details = [{
            "name": tool.name,
            "description": tool.description,
            "parameters": getattr(tool, "args_schema", "Non disponible")
        } for tool in self.tools]
        logger.debug(f"Détails des outils: {json.dumps(tool_details, indent=2, default=str)}")
        
    def _init_memory(self):
        """Initialise la mémoire de l'agent."""
        memory_type = self.config.get('memory_type', settings.memory.type)
        
        # Pour l'instant, on utilise uniquement ConversationBufferMemory
        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            input_key='input',
            return_messages=True,
            output_key='output'
        )
        
        logger.debug(f"Mémoire initialisée: {memory_type}")
        logger.debug(f"Configuration mémoire: memory_key='chat_history', input_key='input', return_messages=True, output_key='output'")
        
    def _init_agent(self):
        """Initialise l'agent LangChain."""
        # Prefix de prompt personnalisé pour suggérer l'utilisation des outils et le multi-step
        prefix_template = """Tu es un assistant IA conversationnel.
        
Pour analyser les fichiers et médias, utilise de préférence l'outil 'extract_media_content' qui fonctionne pour tout type de fichier (document, image, audio, vidéo) plutôt que les outils spécifiques comme process_document ou process_image.

Utilise extract_media_content avec l'ID du média à analyser comme première étape avant d'essayer d'autres outils.

IMPORTANT: Tu es un agent qui peut appeler plusieurs outils successivement.
Après chaque ACTION et OBSERVATION, réfléchis à la suite.
N'arrête pas tant que l'objectif de l'utilisateur n'est pas entièrement atteint.
Pour les messages WhatsApp, tu dois t'assurer que le compte est connecté avant d'envoyer un message.
"""

        # Suffix nécessaire pour inclure le scratchpad et permettre le multi-étapes
        suffix_template = """\
{chat_history}
Tools disponibles : {tool_names}
Question: {input}

Pensées de l'agent (pour planifier les actions):
{agent_scratchpad}"""

        # Log des templates de prompts
        logger.debug(f"Prompt prefix: {prefix_template}")
        logger.debug(f"Prompt suffix: {suffix_template}")
        
        # Configuration de l'agent
        max_iterations = 20
        early_stopping_method = "generate"
        
        # Créer un agent temporaire pour extraire format_instructions
        logger.info("Création d'un agent temporaire pour extraire format_instructions")
        temp_agent = initialize_agent(
            tools=[],  # Aucun outil nécessaire pour cette étape
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False
        )
        
        # Extraire les instructions de formatage
        format_instructions = temp_agent.agent.output_parser.get_format_instructions()
        logger.info("Format instructions récupérées avec succès")
        print("=== FORMAT INSTRUCTIONS RÉCUPÉRÉES ===")
        print(format_instructions)
        
        # Créer les template messages en utilisant les classes MessagePromptTemplate
        from langchain.prompts import (
            SystemMessagePromptTemplate,
            HumanMessagePromptTemplate,
            ChatPromptTemplate,
        )
        
        # Échapper les quadruples accolades de format_instructions avec Jinja2
        raw_fi = "{% raw %}\n" + format_instructions + "\n{% endraw %}"
        
        # Créer les messages du prompt avec le bon format de template
        prefix = SystemMessagePromptTemplate.from_template(
            prefix_template,
            template_format="jinja2"
        )
        format_ins = SystemMessagePromptTemplate.from_template(
            raw_fi,
            template_format="jinja2"
        )
        suffix = HumanMessagePromptTemplate.from_template(
            suffix_template,
            template_format="jinja2"
        )
        
        # Préparer la liste des noms d'outils
        tool_names_list = [tool.name for tool in self.tools]
        
        # Créer le template de prompt directement avec partial_variables
        prompt = ChatPromptTemplate(
            input_variables=[
                "chat_history",
                "input",
                "agent_scratchpad"
            ],
            messages=[prefix, format_ins, suffix],
            partial_variables={"tool_names": tool_names_list}
        )
        
        # Inspecter le prompt final
        print("=== PROMPT MESSAGES ===")
        for i, msg in enumerate(prompt.messages):
            print(f"[{i}] {type(msg).__name__}")
            if hasattr(msg, 'prompt') and hasattr(msg.prompt, '_template'):
                print(msg.prompt._template.strip(), "\n---")
            else:
                print("(Structure non standard, impossible d'afficher le template)")
                print("---")
        print("Variables d'entrée:", prompt.input_variables)
        
        # Initialiser l'agent final avec le prompt personnalisé
        logger.info("Initialisation de l'agent final avec le prompt personnalisé")
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            prompt=prompt,
            memory=self.memory,
            verbose=True,
            max_iterations=max_iterations,
            early_stopping_method=early_stopping_method,
            return_intermediate_steps=True
        )
        
        # Log détaillé de la configuration de l'agent
        agent_config = {
            "type": AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            "max_iterations": max_iterations,
            "early_stopping_method": early_stopping_method,
            "return_intermediate_steps": True,
            "memory_type": type(self.memory).__name__,
            "memory_config": {
                "memory_key": "chat_history",
                "return_messages": True,
                "output_key": "output"
            },
            "tools_count": len(self.tools)
        }
        
        logger.info(f"Agent créé avec configuration: {json.dumps(agent_config, indent=2)}")
        
        # Vérification détaillée du prompt template
        try:
            if hasattr(self.agent, 'agent') and hasattr(self.agent.agent, 'llm_chain'):
                prompt = self.agent.agent.llm_chain.prompt
                logger.info("=== Variables utilisées pour le prompt ===")
                if hasattr(prompt, 'input_variables'):
                    logger.info(f"Variables d'entrée: {prompt.input_variables}")
                    # Vérifier si agent_scratchpad est dans les variables
                    if 'agent_scratchpad' in prompt.input_variables:
                        logger.info("✓ agent_scratchpad est bien présent dans les variables d'entrée")
                    else:
                        logger.error("ALERTE: agent_scratchpad n'est pas dans les variables d'entrée!")
                else:
                    logger.warning("Impossible de récupérer input_variables")
        except Exception as e:
            logger.error(f"Erreur lors de l'examen du template: {str(e)}")
        
        # Afficher le scratchpad à chaque itération (MODIFIED SECTION FOR DEBUG_CALL)
        if hasattr(self.agent, 'agent') and hasattr(self.agent.agent, 'llm_chain'):
            llm_chain = self.agent.agent.llm_chain
            original_llm_chain_call = llm_chain._call # This is the bound method

            def patched_debug_call(self_llm_chain, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None):
                # self_llm_chain is the llm_chain instance
                try:
                    raw_prompt = self_llm_chain.prompt.format(**inputs)
                    logger.info("=== Prompt envoyé à l'API ===")
                    logger.info(raw_prompt)
                    scratchpad = inputs.get('agent_scratchpad', '')
                    logger.info("=== Contenu du scratchpad ===")
                    logger.info(scratchpad)
                except Exception as e:
                    logger.error(f"Erreur lors de la capture du prompt: {str(e)}")
                
                # Call the original bound method correctly
                return original_llm_chain_call(inputs, run_manager=run_manager)

            llm_chain._call = MethodType(patched_debug_call, llm_chain)
            logger.info("Patched LLMChain._call with MethodType for debug logging.")
        
    def process_message(self, message: str) -> str:
        """
        Traite un message utilisateur et renvoie la réponse de l'agent.
        
        Args:
            message: Message de l'utilisateur
            
        Returns:
            Réponse de l'agent
        """
        # → Pour debugging : partir d'un contexte vierge à chaque appel
        self.memory.clear()
        logger.debug("Mémoire effacée pour nouveau contexte")
        
        logger.debug(f"Traitement du message: {message[:50]}...")
        logger.info(f"[Session {self.session_id}] Début du traitement du message")
        
        try:
            # Log du type LLM et de la version de LangChain
            import langchain
            import langchain_community
            logger.debug(f"Version LangChain: {langchain.__version__}")
            logger.debug(f"Version LangChain Community: {langchain_community.__version__}")
            logger.debug(f"Type LLM: {type(self.llm).__name__}")
            
            # AJOUT: Capturer le scratchpad AVANT l'appel
            # current_scratch = getattr(self.agent.agent, "agent_scratchpad", None) # This might not be relevant anymore or might be part of 'inputs'
            # print("SCRATCHPAD AVANT →", current_scratch) # Keep for now, or remove if redundant with new debug log

            # Load memory variables and prepare all inputs for invoke
            memory_vars = self.memory.load_memory_variables({})
            chat_history_messages = memory_vars.get("chat_history", [])
            
            invoke_inputs = {
                "input": message,
                "chat_history": chat_history_messages,
                "agent_scratchpad": "" # Provide initial empty scratchpad
            }
            
            logger.debug(f"Invocation de l'agent avec inputs: { {k: v if k != 'chat_history' else '[messages]' for k, v in invoke_inputs.items()} }") # Log inputs without large history
            result = self.agent.invoke(invoke_inputs)
            
            # AJOUT: Afficher les étapes intermédiaires brutes juste après invoke
            # print("INTERMEDIATE_STEPS →", result.get("intermediate_steps")) # Keep for now
            
            # AJOUT: Afficher le scratchpad APRÈS l'appel (This might be captured by the debug log now)
            # current_scratch_after = getattr(self.agent.agent, "agent_scratchpad", None)
            # print("SCRATCHPAD APRÈS →", current_scratch_after) # Keep for now

            # Restaurer la méthode originale - NO LONGER NEEDED WITH MethodType if we don't unpatch
            # if hasattr(self.agent, 'agent') and hasattr(self.agent.agent, 'llm_chain') and hasattr(llm_chain, '_call_original_for_debug'):
            #    llm_chain._call = llm_chain._call_original_for_debug # Restore
            #    delattr(llm_chain, '_call_original_for_debug')

            # Log du résultat complet (pour le débogage)
            logger.debug(f"Résultat complet: {json.dumps(result, indent=2, default=str)}")
            
            # Examiner les intermediate_steps bruts
            if "intermediate_steps" in result:
                logger.info("=== Intermediate steps bruts ===")
                steps_str = json.dumps(result["intermediate_steps"], indent=2, default=str)
                logger.info(steps_str)
                
                # Extraire les étapes intermédiaires pour le débogage et l'historique
                steps = result["intermediate_steps"]
                self._log_intermediate_steps(steps)
                # Stocker les étapes pour référence ultérieure
                self.last_intermediate_steps = steps
                
                # Log du scratchpad après chaque itération
                logger.debug("Contenu du scratchpad après exécution:")
                scratchpad_content = self._format_scratchpad(steps)
                logger.debug(f"SCRATCHPAD:\n{scratchpad_content}")
                
                # Vérifier s'il y a eu des erreurs ou des conditions d'arrêt dans les observations
                self._check_tool_errors(steps)
            else:
                logger.warning("Aucune étape intermédiaire trouvée dans le résultat")
            
            # Vérifier le finish_reason s'il est disponible
            finish_reason = result.get("finish_reason", None)
            if finish_reason:
                logger.info(f"Finish reason: {finish_reason}")
                if finish_reason == "length":
                    logger.warning("L'agent s'est arrêté car la limite de tokens a été atteinte!")
                elif finish_reason == "stop":
                    logger.info("L'agent s'est arrêté normalement avec un token d'arrêt")
                    
            # Analyser les messages supplémentaires retournés par OpenAI
            if "llm_output" in result:
                llm_output = result["llm_output"]
                logger.debug(f"LLM output: {json.dumps(llm_output, indent=2, default=str)}")
                if isinstance(llm_output, dict) and "token_usage" in llm_output:
                    token_usage = llm_output["token_usage"]
                    logger.info(f"Utilisation de tokens: {token_usage}")
            
            # Extraire la réponse finale
            output = result.get("output", "Je n'ai pas de réponse à cette question.")
            
            # Analyser l'output pour extraire action_input si c'est un dictionnaire ou JSON
            response = output
            try:
                if isinstance(output, str) and (output.strip().startswith("{") and output.strip().endswith("}")):
                    output_dict = json.loads(output)
                    if output_dict.get("action") == "Final Answer":
                        response = output_dict.get("action_input", "Je n'ai pas de réponse à cette question.")
                        logger.info(f"Réponse extraite du champ action_input: {response[:50]}...")
                elif isinstance(output, dict) and output.get("action") == "Final Answer":
                    response = output.get("action_input", "Je n'ai pas de réponse à cette question.")
                    logger.info(f"Réponse extraite directement du dictionnaire: {response[:50]}...")
            except Exception as e:
                logger.warning(f"Erreur lors de l'extraction de la réponse: {str(e)}")
                # Conserver la réponse telle quelle
            
            # Si la réponse indique que l'agent s'est arrêté à cause de la limite d'itérations,
            # formuler une réponse basée sur les observations obtenues
            if "Agent stopped due to iteration limit" in response:
                logger.warning(f"Agent arrêté en raison de la limite d'itérations ({self.agent.max_iterations})")
                if hasattr(self, "last_intermediate_steps") and self.last_intermediate_steps:
                    # Récupérer la dernière observation utile
                    for action, observation in reversed(self.last_intermediate_steps):
                        if "calculer_date" in str(action) and "weekday" in str(action):
                            observation_str = str(observation)
                            if "prochain lundi" in observation_str:
                                # Extraire la date du prochain lundi à partir de l'observation
                                logger.info(f"Utilisation de l'observation de calculer_date: {observation_str}")
                                response = observation_str
                                break
                    
                    # Si aucune observation utile n'a été trouvée, construire une réponse 
                    # à partir de la première observation pertinente
                    if "Agent stopped due to iteration limit" in response:
                        for action, observation in self.last_intermediate_steps:
                            if "calculer_date" in str(action) and "weekday" in str(action) and "prochain lundi" in str(observation):
                                logger.info(f"Utilisation de la première observation de calculer_date: {observation}")
                                response = str(observation)
                                break
            
            logger.info(f"[Session {self.session_id}] Réponse générée: {response[:50]}...")
            return response
            
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Erreur lors du traitement du message: {str(e)}")
            logger.error(f"Stacktrace: {tb}")
            return f"Une erreur s'est produite: {str(e)}"
    
    def _check_tool_errors(self, steps):
        """
        Vérifie si des outils ont renvoyé des erreurs ou des conditions d'arrêt.
        
        Args:
            steps: Liste des étapes intermédiaires (action, observation)
        """
        for i, (action, observation) in enumerate(steps):
            # Convertir observation en chaîne pour la recherche
            observation_str = str(observation)
            
            # Rechercher des motifs d'erreur courants
            if "error" in observation_str.lower() or "exception" in observation_str.lower():
                logger.warning(f"L'outil #{i+1} ({getattr(action, 'tool', 'inconnu')}) a renvoyé une erreur: {observation_str[:100]}...")
            
            # Rechercher des motifs d'arrêt
            if "stop" in observation_str.lower() or "unable to continue" in observation_str.lower():
                logger.warning(f"L'outil #{i+1} ({getattr(action, 'tool', 'inconnu')}) a renvoyé une condition d'arrêt: {observation_str[:100]}...")
    
    def _format_scratchpad(self, steps):
        """
        Formate les étapes intermédiaires en un format de scratchpad pour le débogage.
        
        Args:
            steps: Liste des étapes intermédiaires (action, observation)
            
        Returns:
            Texte formaté du scratchpad
        """
        if not steps:
            return "Scratchpad vide"
            
        scratchpad = []
        for i, (action, observation) in enumerate(steps):
            tool = getattr(action, "tool", "inconnu")
            tool_input = getattr(action, "tool_input", {})
            
            scratchpad.append(f"Thought: Je dois utiliser l'outil {tool}")
            scratchpad.append(f"Action: {tool}")
            scratchpad.append(f"Action Input: {json.dumps(tool_input, default=str)}")
            scratchpad.append(f"Observation: {observation}")
            
            # Simuler le format de scratchpad qui serait normalement généré
            if i < len(steps) - 1:
                scratchpad.append("Thought: Je devrais utiliser un autre outil")
            else:
                scratchpad.append("Thought: J'ai maintenant toutes les informations")
        
        return "\n".join(scratchpad)
    
    def _log_intermediate_steps(self, steps):
        """
        Journalise les étapes intermédiaires de l'agent pour le débogage.
        
        Args:
            steps: Liste des étapes intermédiaires
        """
        if not steps:
            logger.debug("Aucune étape intermédiaire à journaliser")
            return
            
        logger.debug(f"Nombre d'étapes intermédiaires: {len(steps)}")
        
        # Log détaillé de chaque étape
        for i, (action, observation) in enumerate(steps):
            tool = getattr(action, "tool", "inconnu")
            tool_input = getattr(action, "tool_input", {})
            
            # Log plus détaillé pour les développeurs
            logger.debug(f"Étape {i+1}:")
            logger.debug(f"  Type d'action: {type(action).__name__}")
            logger.debug(f"  Outil: {tool}")
            logger.debug(f"  Entrée complète: {json.dumps(tool_input, default=str)}")
            logger.debug(f"  Type d'observation: {type(observation).__name__}")
            logger.debug(f"  Observation complète: {observation}")
            
            # Récupérer et logger l'état interne de l'agent après chaque étape
            try:
                # Cette partie peut dépendre de l'implémentation spécifique de l'agent
                if hasattr(self.agent, 'agent') and hasattr(self.agent.agent, 'llm_chain'):
                    current_input = self.agent.agent.llm_chain.prompt.input_variables
                    logger.debug(f"  Variables d'entrée du prompt: {current_input}")
            except Exception as e:
                logger.debug(f"  Impossible de récupérer l'état interne: {str(e)}")
    
    def get_thinking(self) -> Optional[str]:
        """
        Récupère les étapes de réflexion de l'agent.
        
        Returns:
            Texte contenant les étapes de réflexion
        """
        if hasattr(self, "last_intermediate_steps"):
            # Formater les étapes pour une meilleure lisibilité
            result = []
            for i, (action, observation) in enumerate(self.last_intermediate_steps):
                tool = getattr(action, "tool", "inconnu")
                tool_input = getattr(action, "tool_input", {})
                
                result.append(f"Étape {i+1}:")
                result.append(f"  Outil: {tool}")
                result.append(f"  Entrée: {json.dumps(tool_input, default=str)}")
                result.append(f"  Observation: {observation}")
                result.append("")
            
            return "\n".join(result)
        elif hasattr(self.agent, "intermediate_steps"):
            return str(self.agent.intermediate_steps)
        return None
    
    def dump_agent_state(self) -> Dict[str, Any]:
        """
        Récupère l'état complet de l'agent pour le débogage.
        
        Returns:
            Dictionnaire contenant l'état de l'agent
        """
        state = {
            "session_id": self.session_id,
            "config": self.config,
            "llm_type": type(self.llm).__name__,
            "llm_config": {
                "model_name": getattr(self.llm, "model_name", "non disponible"),
                "temperature": getattr(self.llm, "temperature", "non disponible")
            },
            "tools": [t.name for t in self.tools],
            "memory_type": type(self.memory).__name__,
            "memory_config": {
                "memory_key": getattr(self.memory, "memory_key", "non disponible"),
                "return_messages": getattr(self.memory, "return_messages", "non disponible"),
                "output_key": getattr(self.memory, "output_key", "non disponible")
            },
            "agent_type": type(self.agent).__name__,
            "agent_config": {
                "max_iterations": getattr(self.agent, "max_iterations", "non disponible"),
                "early_stopping_method": getattr(self.agent, "early_stopping_method", "non disponible")
            }
        }
        
        # Récupérer les détails du prompt si disponible
        try:
            if hasattr(self.agent, 'agent') and hasattr(self.agent.agent, 'llm_chain'):
                prompt = self.agent.agent.llm_chain.prompt
                state["prompt"] = {
                    "input_variables": prompt.input_variables,
                    "template": getattr(prompt, "template", "non disponible")
                }
        except Exception as e:
            state["prompt_error"] = str(e)
        
        return state


class AgentFactory:
    """
    Factory pour créer et gérer des instances d'agent.
    """
    
    def __init__(self):
        """
        Initialise la factory avec les paramètres par défaut.
        """
        self.settings = get_settings()
        self.cleanup_interval = timedelta(hours=self.settings.session.ttl_hours)
    
    def get_agent(self, session_id: str, config_override: Optional[Dict[str, Any]] = None) -> Agent:
        """
        Récupère un agent existant ou en crée un nouveau pour la session.
        
        Args:
            session_id: Identifiant de la session
            config_override: Paramètres de configuration à écraser
            
        Returns:
            Une instance d'agent
        """
        # Vérifier si l'agent existe déjà
        if session_id in AGENT_INSTANCES:
            logger.debug(f"Réutilisation d'un agent existant pour la session {session_id}")
            
            # Mettre à jour la configuration si nécessaire
            if config_override:
                self.update_session_config(session_id, config_override)
            
            # Mettre à jour l'horodatage
            SESSION_META[session_id]["updated_at"] = datetime.utcnow().isoformat()
            
            return AGENT_INSTANCES[session_id]
        
        # Créer un nouvel agent
        logger.info(f"Création d'un nouvel agent pour la session {session_id}")
        
        # Préparer la configuration
        config = {}
        
        # Ajouter les configurations de base
        config["model_name"] = self.settings.llm.name
        config["temperature"] = self.settings.llm.temperature
        config["memory_type"] = self.settings.memory.type
        config["tools"] = self.settings.tools.enabled
        
        # Appliquer les overrides de configuration
        if config_override:
            for key, value in config_override.items():
                config[key] = value
        
        # Stocker la configuration de session
        SESSION_CONFIGS[session_id] = config
        
        # Créer directement un nouvel agent avec la configuration
        agent = Agent(session_id=session_id, config=config)
        
        # Stocker l'agent
        AGENT_INSTANCES[session_id] = agent
        
        # Stocker les métadonnées de session
        now = datetime.utcnow().isoformat()
        SESSION_META[session_id] = {
            "created_at": now,
            "updated_at": now
        }
        
        return agent
    
    def session_exists(self, session_id: str) -> bool:
        """
        Vérifie si une session existe.
        """
        return session_id in AGENT_INSTANCES
    
    def get_session_data(self, session_id: str) -> Dict[str, Any]:
        """
        Récupère les données d'une session, y compris l'historique des messages.
        """
        if not self.session_exists(session_id):
            return None
        
        # Récupérer l'historique des messages
        memory = AGENT_INSTANCES[session_id].memory
        messages = []
        
        # Récupérer les messages à partir de la mémoire
        # Cette implémentation dépend du type de mémoire utilisé
        try:
            chat_history = memory.chat_memory.messages
            for msg in chat_history:
                messages.append({
                    "role": "user" if msg.type == "human" else "assistant",
                    "content": msg.content
                })
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des messages: {str(e)}")
        
        # Construire la réponse
        return {
            "session_id": session_id,
            "messages": messages,
            "config": SESSION_CONFIGS.get(session_id, {}),
            "created_at": SESSION_META.get(session_id, {}).get("created_at", ""),
            "updated_at": SESSION_META.get(session_id, {}).get("updated_at", "")
        }
    
    def update_session_config(self, session_id: str, config_update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Met à jour la configuration d'une session existante.
        """
        if not self.session_exists(session_id):
            return None
        
        # Récupérer la configuration actuelle
        current_config = SESSION_CONFIGS.get(session_id, {})
        
        # Appliquer les mises à jour
        if "temperature" in config_update:
            current_config["llm"].temperature = config_update["temperature"]
        if "model_name" in config_update:
            current_config["llm"].name = config_update["model_name"]
        if "memory_type" in config_update:
            current_config["memory"].type = config_update["memory_type"]
            # Ici, il faudrait réinitialiser la mémoire, mais c'est complexe
        if "tools" in config_update:
            current_config["tools"].enabled = config_update["tools"]
        
        # Mettre à jour la configuration stockée
        SESSION_CONFIGS[session_id] = current_config
        
        # Mettre à jour l'horodatage
        SESSION_META[session_id]["updated_at"] = datetime.utcnow().isoformat()
        
        return current_config
    
    @staticmethod
    def create_agent(
        llm_settings: LLMSettings,
        memory_settings: MemorySettings,
        tools_settings: ToolsSettings,
        session_id: Optional[str] = None,
        verbose: bool = False
    ) -> tuple[Agent, str]:
        """
        Crée une instance d'agent configurée.
        
        Args:
            llm_settings: Configuration du LLM
            memory_settings: Configuration de la mémoire
            tools_settings: Configuration des outils
            session_id: ID de session existant (optionnel)
            verbose: Si True, active le mode verbeux
            
        Returns:
            Tuple (agent, session_id)
        """
        # Génération d'un ID de session si nécessaire
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Initialisation du LLM
        llm = get_llm_from_settings(llm_settings)
        
        # Création de l'agent avec juste les noms des outils, pas les objets
        agent = Agent(
            session_id=session_id,
            config={
                "model_name": llm_settings.name,
                "temperature": llm_settings.temperature,
                "memory_type": memory_settings.type,
                "tools": tools_settings.enabled  # Passer les noms des outils, pas les objets
            }
        )
        
        logger.info(f"Agent créé pour la session {session_id}")
        
        # Ajouter une méthode pour capturer et journaliser les requêtes HTTP
        if verbose:
            try:
                # Tentative d'ajout d'un hook pour capturer les requêtes HTTP
                import httpx
                original_transport_send = httpx.HTTPTransport.handle_request
                
                def logging_transport_send(self, request):
                    logger.debug(f"Requête HTTP: {request.method} {request.url}")
                    logger.debug(f"Headers: {request.headers}")
                    if request.content:
                        try:
                            # Tenter de parser le corps comme JSON pour un affichage plus lisible
                            body = json.loads(request.content.decode('utf-8'))
                            # Masquer la clé API si elle est présente
                            if 'api_key' in body:
                                body['api_key'] = '***'
                            logger.debug(f"Body: {json.dumps(body, indent=2)}")
                        except:
                            # Si ce n'est pas du JSON, afficher en texte brut
                            logger.debug(f"Body: {request.content.decode('utf-8')}")
                    
                    response = original_transport_send(self, request)
                    
                    logger.debug(f"Réponse HTTP: {response.status_code}")
                    logger.debug(f"Headers: {response.headers}")
                    try:
                        response_json = response.json()
                        logger.debug(f"Body: {json.dumps(response_json, indent=2)}")
                        if 'finish_reason' in response_json:
                            logger.info(f"finish_reason: {response_json['finish_reason']}")
                    except:
                        logger.debug("Impossible de parser la réponse comme JSON")
                    
                    return response
                
                # Remplacer la méthode d'origine par la version avec logging
                httpx.HTTPTransport.handle_request = logging_transport_send
                logger.info("Monitoring HTTP activé pour les requêtes API")
            except Exception as e:
                logger.warning(f"Impossible d'activer le monitoring HTTP: {str(e)}")
        
        return agent, session_id 