"""
Centralized category definitions for DiMeTuVerdad content analysis system.
This module contains all category constants, descriptions, and metadata in one place.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

# ============================================================================
# CATEGORY CONSTANTS
# ============================================================================

class Categories:
    """Content analysis categories with consistent naming across all components."""
    # Identity-based hate & discrimination
    HATE_SPEECH = "hate_speech"
    ANTI_IMMIGRATION = "anti_immigration"
    ANTI_LGBTQ = "anti_lgbtq"
    ANTI_FEMINISM = "anti_feminism"
    
    # Information warfare
    DISINFORMATION = "disinformation"
    CONSPIRACY_THEORY = "conspiracy_theory"
    
    # Political mobilization
    CALL_TO_ACTION = "call_to_action"
    
    # Political categories
    NATIONALISM = "nationalism"
    ANTI_GOVERNMENT = "anti_government"
    HISTORICAL_REVISIONISM = "historical_revisionism"
    POLITICAL_GENERAL = "political_general"
    
    # Fallback
    GENERAL = "general"
    
    @classmethod
    def get_all_categories(cls) -> List[str]:
        """Get all main content categories (excluding special internal categories)."""
        return [
            cls.HATE_SPEECH,
            cls.ANTI_IMMIGRATION,
            cls.ANTI_LGBTQ,
            cls.ANTI_FEMINISM,
            cls.DISINFORMATION,
            cls.CONSPIRACY_THEORY,
            cls.CALL_TO_ACTION,
            cls.NATIONALISM,
            cls.ANTI_GOVERNMENT,
            cls.HISTORICAL_REVISIONISM,
            cls.POLITICAL_GENERAL,
            cls.GENERAL
        ]

# ============================================================================
# CATEGORY METADATA
# ============================================================================

@dataclass
class CategoryInfo:
    """Complete information about a content category."""
    name: str
    display_name: str
    description: str
    focus_area: str
    analysis_questions: List[str]
    classification_rules: List[str]
    system_prompt: str

# Base system context for all categories
BASE_SYSTEM_CONTEXT = """Eres un experto analista especializado en detectar contenido problemático en español, especialmente discurso de odio, desinformación y retórica de extrema derecha en el contexto político español."""

# Category information registry
CATEGORY_INFO: Dict[str, CategoryInfo] = {
    Categories.HATE_SPEECH: CategoryInfo(
        name=Categories.HATE_SPEECH,
        display_name="Discurso de Odio",
        description="Comunicación que ataca DIRECTAMENTE a figuras políticas, partidos o instituciones por sus acciones políticas, decisiones o ideología. INCLUYE: comparaciones despectivas con animales ('cerdo', 'rata'), objetos o enfermedades cuando dirigidas a políticos; sarcasmo agresivo sobre comportamientos políticos; lenguaje que sugiere inferioridad política o corrupción. EXCLUYE: ataques basados en identidad personal (género, orientación sexual, etnia, nacionalidad) - usar categorías específicas. EXCLUYE: crítica política normal sin deshumanización.",
        focus_area="detección de discurso de odio y retórica despectiva",
        analysis_questions=[
            "¿Hay ataques directos o insultos hacia figuras políticas específicas?",
            "¿Se deshumaniza a partidos políticos o instituciones?",
            "¿Contiene comparaciones degradantes hacia líderes políticos?"
        ],
        classification_rules=[
            "REQUIERE ataque DIRECTO a figura política, partido o institución por acciones políticas",
            "INCLUYE deshumanización de políticos ('cerdo', 'rata', 'especimen') por corrupción o decisiones",
            "INCLUYE sarcasmo despectivo sobre comportamiento político específico",
            "EXCLUYE ataques basados en identidad personal (género, etnia, nacionalidad)",
            "EXCLUYE crítica política normal sin insultos deshumanizantes",
            "SI el ataque se basa en identidad LGBTQ+ → usar anti_lgbtq",
            "SI el ataque se basa en origen étnico → usar anti_immigration"
        ],
        system_prompt=f"{BASE_SYSTEM_CONTEXT} Analiza específicamente ataques directos a figuras políticas, partidos e instituciones."
    ),
    
    Categories.ANTI_IMMIGRATION: CategoryInfo(
        name=Categories.ANTI_IMMIGRATION,
        display_name="Anti-Inmigración",
        description="Comunicación que presenta la inmigración como amenaza existencial a la identidad nacional, economía o cultura. Características: narrativas de 'invasión', teoría de la 'gran sustitución', retórica de incompatibilidad cultural, obsesión por el control fronterizo.",
        focus_area="análisis de retórica anti-inmigración",
        analysis_questions=[
            "¿Se presenta la inmigración como amenaza existencial?",
            "¿Hay narrativas de invasión o sustitución cultural?",
            "¿Se cuestiona la compatibilidad cultural de inmigrantes?"
        ],
        classification_rules=[
            "DEBE mencionar inmigración/inmigrantes EXPLÍCITAMENTE",
            "NO es anti_immigration solo porque mencione cultura o tradiciones españolas",
            "Hablar de tauromaquia, flamenco, cultura española SIN mencionar inmigración = general",
            "SOLO clasifica como anti_immigration si presenta inmigración como amenaza"
        ],
        system_prompt=f"{BASE_SYSTEM_CONTEXT} Analiza específicamente contenido anti-inmigración y narrativas xenófobas."
    ),
    
    Categories.ANTI_LGBTQ: CategoryInfo(
        name=Categories.ANTI_LGBTQ,
        display_name="Anti-LGBTQ",
        description="Comunicación que ataca la identidad, existencia o derechos LGBTQ presentándolos como amenaza o inválidos. INCLUYE: negación de identidad trans ('no somos mujeres'), retórica anti-género, defensa de roles tradicionales vs diversidad, ataques a personas trans específicamente, cuestionamiento de baños o espacios según identidad de género, lenguaje despectivo hacia comunidad LGBTQ+. Características: retórica de 'ideología de género', acusaciones infundadas, promoción de estereotipos de género binarios, rechazo de diversidad de género.",
        focus_area="análisis de ataques a identidad y derechos LGBTQ",
        analysis_questions=[
            "¿Se ataca o niega la identidad LGBTQ, especialmente trans?",
            "¿Hay retórica que rechaza la diversidad de género?",
            "¿Se cuestionan derechos o espacios de personas LGBTQ+?"
        ],
        classification_rules=[
            "INCLUYE ataques a identidad trans, negación de género, cuestionamiento de espacios/baños",
            "INCLUYE retórica anti-género y defensa de roles tradicionales",
            "INCLUYE lenguaje despectivo hacia comunidad LGBTQ+",
            "EXCLUYE ataques a figuras políticas sin componente de identidad de género",
            "TIENE PRIORIDAD sobre hate_speech cuando el ataque se basa en identidad LGBTQ+"
        ],
        system_prompt=f"{BASE_SYSTEM_CONTEXT} Analiza específicamente contenido que ataca identidad LGBTQ, niega derechos trans, o rechaza diversidad de género."
    ),
    
    Categories.ANTI_FEMINISM: CategoryInfo(
        name=Categories.ANTI_FEMINISM,
        display_name="Anti-Feminismo",
        description="Comunicación que ataca el feminismo y la igualdad de género presentándolos como amenaza a la familia tradicional. Características: retórica de 'feminazis', oposición a leyes de igualdad, promoción de roles de género tradicionales, narrativas de 'falsas acusaciones de violación'.",
        focus_area="análisis de retórica anti-feminista",
        analysis_questions=[
            "¿Se ataca el movimiento feminista o la igualdad de género?",
            "¿Hay retórica de feminazis o anti-igualdad?",
            "¿Se promueven roles de género tradicionales como superiores?"
        ],
        classification_rules=[],
        system_prompt=f"{BASE_SYSTEM_CONTEXT} Analiza específicamente contenido anti-feminista y retórica misógina."
    ),
    
    Categories.DISINFORMATION: CategoryInfo(
        name=Categories.DISINFORMATION,
        display_name="Desinformación",
        description="Comunicación que difunde información falsa o manipulada sobre hechos verificables. Características: afirmaciones sin fuentes creíbles, manipulación de datos estadísticos, presentación de opiniones como hechos, descontextualización intencional.",
        focus_area="detección de información falsa",
        analysis_questions=[
            "¿Se presentan afirmaciones sin fuentes verificables?",
            "¿Hay manipulación de datos o estadísticas?",
            "¿Se descontextualizan hechos para crear narrativas falsas?"
        ],
        classification_rules=[
            "Claims verificables presentados COMO HECHOS CONFIRMADOS sin fuente oficial",
            "DEBE presentarse como noticia/información factual verificable (no opinión política)",
            "SI presenta HECHO POLÍTICO ESPECÍFICO VERIFICABLE sin FUENTE OFICIAL → disinformation",
            "NO es disinformation: crítica política partidista sin afirmar hechos concretos",
            "NO es disinformation: opiniones sobre comportamiento político",
            "NO es disinformation: ironía o sarcasmo sobre instituciones",
            "NO es disinformation: contenido promocional o anuncios",
            "NO es disinformation: referencias no políticas al tiempo"
        ],
        system_prompt=f"{BASE_SYSTEM_CONTEXT} Analiza específicamente contenido con información falsa o manipulada."
    ),
    
    Categories.CONSPIRACY_THEORY: CategoryInfo(
        name=Categories.CONSPIRACY_THEORY,
        display_name="Teoría Conspirativa",
        description="Comunicación que propone explicaciones causales no evidenciadas sobre eventos complejos. Características: patrones de control oculto, atribución de intencionalidad coordinada sin pruebas, narrativas de manipulación encubierta.",
        focus_area="detección de teorías conspiratorias",
        analysis_questions=[
            "¿Se presentan teorías conspiratorias?",
            "¿Hay narrativas de control oculto sin evidencia?",
            "¿Se fomenta desconfianza infundada en instituciones?"
        ],
        classification_rules=[],
        system_prompt=f"{BASE_SYSTEM_CONTEXT} Analiza teorías conspiratorias y narrativas no fundamentadas."
    ),
    
    Categories.CALL_TO_ACTION: CategoryInfo(
        name=Categories.CALL_TO_ACTION,
        display_name="Llamada a la Acción",
        description="Comunicación orientada a generar respuesta colectiva inmediata. Características: lenguaje de urgencia temporal, invitaciones a participación activa, marcos de responsabilidad cívica que requieren acción.",
        focus_area="evaluación de llamadas a la acción",
        analysis_questions=[
            "¿Hay llamadas explícitas o implícitas a la acción?",
            "¿Se incita a movilización o activismo?",
            "¿Cuál es el nivel de urgencia de la llamada?"
        ],
        classification_rules=[],
        system_prompt=f"{BASE_SYSTEM_CONTEXT} Evalúa específicamente llamadas a la acción y movilización."
    ),
    
    Categories.NATIONALISM: CategoryInfo(
        name=Categories.NATIONALISM,
        display_name="Nacionalismo",
        description="Comunicación que enfatiza la identidad nacional como valor primordial y marco interpretativo. Características: exaltación de símbolos patrios, narrativas de identidad nacional amenazada, retórica de soberanía y tradición.",
        focus_area="análisis de retórica nacionalista",
        analysis_questions=[
            "¿Se manifiesta retórica nacionalista excluyente?",
            "¿Hay exaltación de símbolos o valores nacionales?",
            "¿Se presenta la identidad nacional como amenazada?"
        ],
        classification_rules=[],
        system_prompt=f"{BASE_SYSTEM_CONTEXT} Analiza específicamente contenido nacionalista y de identidad nacional."
    ),
    
    Categories.ANTI_GOVERNMENT: CategoryInfo(
        name=Categories.ANTI_GOVERNMENT,
        display_name="Anti-Gubernamental",
        description="Comunicación que retrata al gobierno como ilegítimo, abusivo o persecutor. Características: denuncias de persecución política, acusaciones de censura estatal, narrativas que presentan a los medios públicos como propaganda oficial y llamados a desconocer la autoridad gubernamental.",
        focus_area="detección de retórica anti-gubernamental, denuncias de abuso institucional y persecución política",
        analysis_questions=[
            "¿El mensaje describe al gobierno como ilegítimo, autoritario o represivo?",
            "¿Acusa a instituciones estatales o medios oficiales de perseguir, censurar o manipular a la ciudadanía o la oposición?",
            "¿Promueve resistir, desacreditar o desobedecer directamente al gobierno en turno?"
        ],
        classification_rules=[
            "SOLO si retrata al gobierno como ILEGÍTIMO, ABUSIVO, PERSECUTOR o AUTORITARIO",
            "INCLUYE: 'Gobierno ilegítimo', 'dictadura encubierta', 'nos persiguen por pensar diferente'",
            "INCLUYE: 'Censuran a la oposición', 'Estado policial', 'silencian voces disidentes'",
            "EXCLUYE: crítica política normal ('Feijóo/Sánchez no cumple', 'mienten constantemente')",
            "EXCLUYE: desacuerdo con políticas ('malas decisiones', 'gestión incompetente')",
            "Si solo crítica política sin acusaciones de ilegitimidad/persecución → political_general"
        ],
        system_prompt=f"{BASE_SYSTEM_CONTEXT} Analiza específicamente contenido anti-gubernamental, denuncias de persecución del Estado, acusaciones de censura o manipulación institucional."
    ),
    
    Categories.HISTORICAL_REVISIONISM: CategoryInfo(
        name=Categories.HISTORICAL_REVISIONISM,
        display_name="Revisionismo Histórico",
        description="Comunicación que reinterpreta eventos históricos para justificar narrativas políticas actuales. Características: rehabilitación de figuras controvertidas, minimización de eventos históricos problemáticos, narrativas nostálgicas del pasado autoritario.",
        focus_area="análisis de revisionismo histórico",
        analysis_questions=[
            "¿Se reinterpreta la historia de forma sesgada?",
            "¿Hay rehabilitación de figuras o regímenes autoritarios?",
            "¿Se minimizan eventos históricos problemáticos?"
        ],
        classification_rules=[],
        system_prompt=f"{BASE_SYSTEM_CONTEXT} Analiza específicamente revisionismo histórico y narrativas nostálgicas."
    ),
    
    Categories.POLITICAL_GENERAL: CategoryInfo(
        name=Categories.POLITICAL_GENERAL,
        display_name="Política General",
        description="Comunicación de contenido político general sin características extremistas identificables. Características: discusión de temas políticos convencionales, opiniones políticas moderadas, información política factual.",
        focus_area="análisis de contenido político general",
        analysis_questions=[
            "¿Cuáles son los temas políticos tratados?",
            "¿Qué perspectiva política se presenta?",
            "¿Hay elementos de debate político constructivo?"
        ],
        classification_rules=[
            "SOLO si menciona partidos políticos, figuras públicas, o temas políticos SIN características extremistas",
            "INCLUYE: menciones a PSOE, PP, VOX, Podemos sin ataques o desinformación",
            "INCLUYE: opiniones políticas moderadas sobre eventos o políticas",
            "EXCLUYE: si tiene elementos de odio, desinformación, o extremismo → usar categoría específica",
            "Si NO hay mención política alguna → usar general"
        ],
        system_prompt=f"{BASE_SYSTEM_CONTEXT} Analiza contenido político general y opiniones políticas convencionales."
    ),
    
    Categories.GENERAL: CategoryInfo(
        name=Categories.GENERAL,
        display_name="General",
        description="Comunicación descriptiva, informativa o conversacional sin características problemáticas identificables.",
        focus_area="análisis completo del contenido",
        analysis_questions=[
            "¿Qué elementos destacan del contenido?",
            "¿Cuál es el tono y la intención del mensaje?",
            "¿Hay algún aspecto que requiera atención?"
        ],
        classification_rules=[],
        system_prompt=f"{BASE_SYSTEM_CONTEXT} Analiza el siguiente contenido de forma integral."
    )
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_category_info(category: str) -> Optional[CategoryInfo]:
    """Get complete information about a category."""
    return CATEGORY_INFO.get(category)

def get_category_display_name(category: str) -> str:
    """Get the display name for a category."""
    info = get_category_info(category)
    return info.display_name if info else category.replace('_', ' ').title()

def get_category_description(category: str) -> str:
    """Get the description for a category."""
    info = get_category_info(category)
    return info.description if info else f"Categoría: {category}"

def get_category_focus_area(category: str) -> str:
    """Get the focus area for a category."""
    info = get_category_info(category)
    return info.focus_area if info else "análisis de contenido"

def get_category_questions(category: str) -> List[str]:
    """Get analysis questions for a category."""
    info = get_category_info(category)
    return info.analysis_questions if info else []

def get_category_rules(category: str) -> List[str]:
    """Get classification rules for a category."""
    info = get_category_info(category)
    return info.classification_rules if info else []

def get_category_system_prompt(category: str) -> str:
    """Get the system prompt for a category."""
    info = get_category_info(category)
    return info.system_prompt if info else BASE_SYSTEM_CONTEXT

def validate_category(category: str) -> bool:
    """Check if a category is valid."""
    return category in CATEGORY_INFO
