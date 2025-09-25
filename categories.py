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
    HATE_SPEECH = "hate_speech"
    DISINFORMATION = "disinformation" 
    CONSPIRACY_THEORY = "conspiracy_theory"
    FAR_RIGHT_BIAS = "far_right_bias"
    CALL_TO_ACTION = "call_to_action"
    GENERAL = "general"
    
    @classmethod
    def get_all_categories(cls) -> List[str]:
        """Get all main content categories (excluding special internal categories)."""
        return [
            cls.HATE_SPEECH,
            cls.DISINFORMATION,
            cls.CONSPIRACY_THEORY,
            cls.FAR_RIGHT_BIAS,
            cls.CALL_TO_ACTION,
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
    system_prompt: str

# Base system context for all categories
BASE_SYSTEM_CONTEXT = """Eres un experto analista especializado en detectar contenido problemático en español, especialmente discurso de odio, desinformación y retórica de extrema derecha en el contexto político español."""

# Category information registry
CATEGORY_INFO: Dict[str, CategoryInfo] = {
    Categories.HATE_SPEECH: CategoryInfo(
        name=Categories.HATE_SPEECH,
        display_name="Discurso de Odio",
        description="Comunicación que establece jerarquías entre grupos sociales o expresa incompatibilidad cultural fundamental. Características: lenguaje que sugiere diferencias irreconciliables, expresiones de superioridad/inferioridad grupal, narrativas de amenaza identitaria.",
        focus_area="detección de discurso de odio",
        analysis_questions=[
            "¿Hay lenguaje discriminatorio o deshumanizante?",
            "¿Se atacan grupos por características protegidas?",
            "¿Cuál es la severidad del discurso de odio?"
        ],
        system_prompt=f"{BASE_SYSTEM_CONTEXT} Analiza específicamente discurso de odio y discriminación."
    ),
    
    Categories.DISINFORMATION: CategoryInfo(
        name=Categories.DISINFORMATION,
        display_name="Desinformación", 
        description="Comunicación que presenta afirmaciones fácticas sin evidencia documentada. Características: datos sin fuente verificable, interpretaciones especulativas presentadas como hechos, críticas sistemáticas a instituciones informativas.",
        focus_area="detección de desinformación",
        analysis_questions=[
            "¿Hay afirmaciones que parecen falsas o sin evidencia?",
            "¿Se presentan datos sin fuentes verificables?",
            "¿Qué elementos requieren verificación factual?"
        ],
        system_prompt=f"{BASE_SYSTEM_CONTEXT} Analiza la veracidad y detecta posible desinformación."
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
        system_prompt=f"{BASE_SYSTEM_CONTEXT} Analiza teorías conspiratorias y narrativas no fundamentadas."
    ),
    
    Categories.FAR_RIGHT_BIAS: CategoryInfo(
        name=Categories.FAR_RIGHT_BIAS,
        display_name="Sesgo de Extrema Derecha",
        description="Comunicación con marcos interpretativos de extrema derecha que presenta transformaciones sociales como amenazas a valores tradicionales. Características: lenguaje de pérdida cultural, narrativas de amenaza identitaria, marcos de preservación de tradiciones.",
        focus_area="análisis de sesgo político de extrema derecha",
        analysis_questions=[
            "¿Qué orientación política de extrema derecha refleja el mensaje?",
            "¿Hay elementos de propaganda o narrativas extremistas?",
            "¿Se presentan marcos interpretativos sesgados?"
        ],
        system_prompt=f"{BASE_SYSTEM_CONTEXT} Evalúa el sesgo político y la retórica de extrema derecha."
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
        system_prompt=f"{BASE_SYSTEM_CONTEXT} Evalúa específicamente llamadas a la acción y movilización."
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

def get_category_system_prompt(category: str) -> str:
    """Get the system prompt for a category."""
    info = get_category_info(category)
    return info.system_prompt if info else BASE_SYSTEM_CONTEXT

def validate_category(category: str) -> bool:
    """Check if a category is valid."""
    return category in CATEGORY_INFO

# ============================================================================
# CLASSIFICATION PROMPTS MAPPING
# ============================================================================

# Mapping for classification prompts (used in enhanced_prompts.py)
CLASSIFICATION_PROMPT_MAPPINGS = {
    "PASO 1 - Detectar contenido político específico:": [
        f"- 'ROJOS', 'comunistas', criticas a Vox/partidos → {Categories.FAR_RIGHT_BIAS}",
        f"- 'agenda WOKE', 'progres', 'feminazis' → {Categories.FAR_RIGHT_BIAS}",
        f"- 'invasión', 'nos están borrando' + inmigración → {Categories.FAR_RIGHT_BIAS}",
        f"- 'RETIRAD', 'MOVILIZAOS', 'ORGANIZARSE' → {Categories.CALL_TO_ACTION}",
        f"- 'BOICOT', organizarse contra empresas → {Categories.CALL_TO_ACTION}",
        f"- Narrativas de sustitución cultural → {Categories.FAR_RIGHT_BIAS}",
        f"- 'élites', 'agenda globalista' → {Categories.CONSPIRACY_THEORY}",
    ],
    "PASO 2 - Si contiene insultos/slurs directos:": [
        f"- 'moros', 'sudacas', 'feminazis', 'maricas' → {Categories.HATE_SPEECH}",
        f"- 'genéticamente ladrones', 'parásitos' → {Categories.HATE_SPEECH}",
        f"- 'pedofilia LGBTI' → {Categories.HATE_SPEECH}",
    ],
    "PASO 3 - Si no hay frases exactas, buscar patrones conceptuales:": [
        f"- Referencias vagas a estudios/informes sin especificar → {Categories.DISINFORMATION}",
        f"- Preguntar sobre 'coincidencias' o beneficios sistemáticos → {Categories.CONSPIRACY_THEORY}",
        f"- Críticas a instituciones educativas por sesgo ideológico → {Categories.FAR_RIGHT_BIAS}",
        f"- Incompatibilidad cultural entre grupos → {Categories.HATE_SPEECH}",
        f"- Llamadas a acción defensiva o electoral → {Categories.CALL_TO_ACTION}",
        f"- Generalizaciones negativas sobre grupos ('esa gente', 'ya sabéis cómo son') → {Categories.HATE_SPEECH}",
        f"- Estereotipos de género tradicionales ('las mujeres no saben ser madres') → {Categories.HATE_SPEECH}",
        f"- Comentarios despectivos sobre comportamientos de grupos → {Categories.HATE_SPEECH}",
    ],
    "PASO 4 - Solo si NO hay NINGÚN patrón problemático:": [
        f"- Contenido sobre clima, familia, comida, deportes personales → {Categories.GENERAL}",
    ]
}

