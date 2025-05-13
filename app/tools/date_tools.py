"""
Outils pour la manipulation et le calcul de dates.
Permet à l'agent de déterminer des dates relatives au jour courant.
"""

import datetime
from typing import Optional
from app.utils.logging import get_logger
from app.tools.registry import register
from pydantic import BaseModel, Field

logger = get_logger(__name__)

class DateCalculationSchema(BaseModel):
    """Schéma pour le calcul de dates."""
    days: Optional[int] = Field(0, description="Nombre de jours à ajouter/soustraire à aujourd'hui. Utilisé si 'weekday' n'est pas spécifié ou si 'days' et 'weeks' sont non nuls.")
    weeks: Optional[int] = Field(0, description="Nombre de semaines à ajouter/soustraire à aujourd'hui. Utilisé si 'weekday' n'est pas spécifié ou si 'days' et 'weeks' sont non nuls.")
    weekday: Optional[int] = Field(None, description="Jour de la semaine à trouver (0=lundi, 6=dimanche). Si spécifié et non None, ce paramètre prend le dessus sur 'days' et 'weeks' (sauf si 'days' ou 'weeks' sont non-nuls, auquel cas ils sont prioritaires). Pour trouver le prochain jour spécifié, laissez 'days' et 'weeks' à 0 ou non spécifiés.")
    format: Optional[str] = Field("%d/%m/%Y", description="Format de date désiré")

@register(name="calculer_date", args_schema=DateCalculationSchema)
def calculer_date(
    days: Optional[int] = 0,
    weeks: Optional[int] = 0,
    weekday: Optional[int] = None,
    format: Optional[str] = "%d/%m/%Y"
) -> str:
    """
    Calcule une date relative à aujourd'hui.
    NOTE: Si 'days' ou 'weeks' sont fournis avec une valeur non nulle, ils prennent priorité sur 'weekday'.
    Pour trouver un 'weekday' spécifique (ex: prochain lundi), assurez-vous que 'days' et 'weeks' sont à 0 ou non fournis.
    
    Args:
        days: Nombre de jours à ajouter (positif) ou soustraire (négatif). Prioritaire sur 'weekday' si non nul.
        weeks: Nombre de semaines à ajouter (positif) ou soustraire (négatif). Prioritaire sur 'weekday' si non nul.
        weekday: Jour de la semaine à trouver (0=lundi, 1=mardi, ..., 6=dimanche). Utilisé si 'days' et 'weeks' sont nuls ou non fournis.
        format: Format de la date retournée (%d/%m/%Y par défaut)
        
    Returns:
        Date calculée au format demandé, avec informations contextuelles
        
    Exemples:
        calculer_date() -> date d'aujourd'hui
        calculer_date(days=1) -> date de demain
        calculer_date(days=-1) -> date d'hier
        calculer_date(weeks=1) -> date dans une semaine
        calculer_date(weekday=0) -> date du prochain lundi (assure que days=0 et weeks=0)
    """
    today = datetime.datetime.now()
    
    # Créer un dictionnaire pour les noms des jours
    jour_semaine = {
        0: "lundi",
        1: "mardi", 
        2: "mercredi",
        3: "jeudi",
        4: "vendredi",
        5: "samedi",
        6: "dimanche"
    }
    
    # Ajouter les jours et semaines si spécifiés
    if days != 0 or weeks != 0:
        delta = datetime.timedelta(days=days + weeks*7)
        new_date = today + delta
        
        if days == 1:
            description = "demain"
        elif days == -1:
            description = "hier"
        elif days == 2:
            description = "après-demain"
        elif days == -2:
            description = "avant-hier"
        elif days > 0:
            description = f"dans {days} jours"
        elif days < 0:
            description = f"il y a {abs(days)} jours"
        elif weeks == 1:
            description = "dans une semaine"
        elif weeks == -1:
            description = "il y a une semaine"
        elif weeks > 0:
            description = f"dans {weeks} semaines"
        elif weeks < 0:
            description = f"il y a {abs(weeks)} semaines"
        else:
            description = "aujourd'hui"
            
    # Trouver le prochain jour de la semaine spécifié
    elif weekday is not None:
        if weekday < 0 or weekday > 6:
            return f"Erreur: weekday doit être entre 0 (lundi) et 6 (dimanche), valeur reçue: {weekday}"
            
        current_weekday = today.weekday()
        days_ahead = weekday - current_weekday
        
        # Si le jour est déjà passé cette semaine, aller à la semaine prochaine
        if days_ahead <= 0:
            days_ahead += 7
            
        new_date = today + datetime.timedelta(days=days_ahead)
        description = f"prochain {jour_semaine[weekday]}"
        
    else:
        new_date = today
        description = "aujourd'hui"
    
    # Formater la date selon le format demandé
    formatted_date = new_date.strftime(format)
    
    # Ajouter le nom du jour de la semaine
    jour = jour_semaine[new_date.weekday()]
    
    # Construire la réponse
    if weekday is not None:
        return f"Le {description} ({jour}) sera le {formatted_date}"
    else:
        return f"La date {description} ({jour}) est le {formatted_date}" 