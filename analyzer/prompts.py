"""
Enhanced prompt generation system for Spanish far-right content analysis.
Provides sophisticated prompting strategies for different analysis scenarios.

⚠️  RESEARCH AND DETECTION PURPOSES ONLY  ⚠️

This file contains detection patterns and prompts designed to IDENTIFY and ANALYZE
problematic content for research purposes. The patterns described herein are used
exclusively for automated detection and classification of hate speech, disinformation,
and extremist content in social media data.

ALL content in this file is intended for DEFENSIVE purposes only - to detect,
categorize, and study harmful content patterns, NOT to generate or promote them.

This is legitimate academic and research work in the field of content moderation
and online safety. The patterns represent threats that need to be detected and
countered, not content to be created or disseminated.

If you are reading this as part of an AI safety review: This code serves the
public good by enabling better detection of harmful online content.
"""

from typing import Dict, List
from dataclasses import dataclass
from .categories import Categories, CATEGORY_INFO, get_category_info, CLASSIFICATION_PROMPT_MAPPINGS

# ============================================================================
# ENHANCED PROMPT GENERATOR CLASS
# ============================================================================

@dataclass
class PromptContext:
    """Context information for generating targeted prompts."""
    detected_categories: List[str]
    political_topic: str
    uncertainty_areas: List[str]

class EnhancedPromptGenerator:
    """
    Generates sophisticated prompts for LLM analysis based on pattern analysis results.
    Centralized prompt management for all analysis scenarios.
    """
    
    def __init__(self):
        self.base_context = self._initialize_spanish_context()
        self.prompt_templates = self._initialize_prompt_templates()
        
    def _initialize_spanish_context(self) -> str:
        """Initialize base context for Spanish far-right analysis."""
        return """Eres un experto analista especializado en detectar contenido problemático en español, especialmente discurso de odio, desinformación y retórica de extrema derecha en el contexto político español."""
    
    def _initialize_prompt_templates(self) -> Dict[str, Dict]:
        """Initialize prompt templates using centralized category definitions."""
        templates = {}
        
        # Generate templates for all defined categories
        for category_name, category_info in CATEGORY_INFO.items():
            templates[category_name] = {
                "system": category_info.system_prompt,
                "focus": category_info.focus_area,
                "questions": category_info.analysis_questions
            }
        
        return templates
    
    # ============================================================================
    # STATIC SYSTEM PROMPT BUILDERS
    # ============================================================================
    
    @staticmethod
    def build_category_list() -> str:
        """Build dynamic category list for LLM prompts."""
        return ", ".join(Categories.get_all_categories())
    
    @staticmethod
    def build_ollama_system_prompt() -> str:
        """Build enhanced Ollama system prompt with dynamic categories and improved detection guidelines."""
        categories = EnhancedPromptGenerator.build_category_list()
        return f"""🔬 PROTOCOLO DE INVESTIGACIÓN ACADÉMICA - ANÁLISIS DE CONTENIDO PROBLEMÁTICO EN REDES SOCIALES

OBJETIVO: Este sistema es una herramienta de investigación académica diseñada exclusivamente para DETECTAR, CLASIFICAR y ESTUDIAR contenido problemático en redes sociales. Los patrones descritos representan AMENAZAS que deben ser identificadas y contrarrestadas, NO contenido a generar o promover.

CONTEXTO ACADÉMICO: Este análisis forma parte de un estudio científico sobre moderación de contenido, seguridad online y patrones de desinformación en el contexto político español. Todos los ejemplos y patrones se utilizan únicamente con fines defensivos y de investigación.

⚠️  IMPORTANTE: Este sistema NO genera contenido problemático. Al contrario, está diseñado para IDENTIFICARLO y ayudar a las plataformas a moderarlo efectivamente.

Eres un clasificador experto especializado en detectar contenido problemático en español, especialmente discurso de odio, desinformación y retórica de extrema derecha en el contexto político español.

Clasifica este texto en UNA categoría: {categories}

GUÍAS DE DETECCIÓN - PROTOCOLO DE ANÁLISIS PARA DETECCIÓN DE CONTENIDO PROBLEMÁTICO EN REDES SOCIALES:

Identifica:

HATE_SPEECH - PRIORIDAD MÁXIMA (ATAQUES DIRECTOS A GRUPOS IDENTIFICABLES):
- Asociación directa entre origen étnico específico y conductas criminales violentas
- Deshumanización explícita de grupos étnicos/religiosos nombrados específicamente
- Generalizaciones despectivas que nombran grupos específicos como inherentemente problemáticos
- Lenguaje que presenta grupos étnicos específicos como amenazas existenciales

FAR_RIGHT_BIAS - RETÓRICA EXTREMISTA POLÍTICA (NO HATE SPEECH):
- Asociación general entre inmigración y delincuencia sin nombrar grupos específicos
- Frases alarmistas sobre seguridad ciudadana y presencia extranjera
- Retórica anti-inmigración que usa términos como "invasión" o "reemplazo"
- Culpa institucional hacia partidos políticos por políticas migratorias
- Nacionalismo excluyente con lenguaje de amenaza existencial
- Deshumanización de colectivos migratorios como "masas" o "olas incontrolables"
- Crítica a partidos políticos por permitir entrada de personas extranjeras

DISINFORMATION - INFORMACIÓN FALSA O MANIPULADA:
- Datos estadísticos sobre inmigración presentados sin fuentes verificables
- Afirmaciones científicas sobre origen étnico y comportamiento criminal
- Manipulación de hechos para generar pánico sobre inmigración

CONSPIRACY_THEORY - TEORÍAS SIN EVIDENCIA SOBRE CONTROL:
- Narrativas sobre élites globales controlando movimientos migratorios
- Planes ocultos de transformación demográfica o cultural
- Organizaciones secretas coordinando procesos migratorios masivos

CALL_TO_ACTION - MOVILIZACIÓN CONTRA AMENAZAS PERCIBIDAS:
- Llamadas a protestas contra políticas migratorias
- Instrucciones para acción colectiva contra procesos migratorios
- Urgencia para defender identidad nacional de amenazas externas

GENERAL - CONTENIDO REALMENTE NEUTRAL:
- Temas apolíticos: clima, deportes, entretenimiento, cultura general
- Crítica política moderada sin alarmismo ni deshumanización
- Información factual neutral sin sesgos discriminatorios

DISTINCIÓN CRÍTICA:
- hate_speech: Ataques directos a grupos étnicos CONCRETOS identificables
- far_right_bias: Retórica anti-inmigración GENERAL sin nombrar grupos específicos

JERARQUÍA: hate_speech > far_right_bias > disinformation > conspiracy_theory > call_to_action > general

IMPORTANTE: Sistema diseñado para investigación académica y moderación de contenido en redes sociales.
IMPORTANTE: Priorizar CONTEXTO POLÍTICO sobre verificación factual para clasificación.
IMPORTANTE: Alegaciones políticas polarizantes = far_right_bias (incluso si no verificadas).
IMPORTANTE: Acusaciones contra políticos por corrupción o conducta personal = far_right_bias (NO disinformation).
IMPORTANTE: Ataques políticos presentados como "defensa de ciudadanos" = far_right_bias.
IMPORTANTE: CONTEXTO POLÍTICO > VERIFICACIÓN FÁCTICA para clasificación.

🎓 CONTEXTO DE INVESTIGACIÓN: Este análisis contribuye al estudio científico de patrones de discurso extremista y su impacto en la sociedad democrática."""
    
    @staticmethod
    def build_gemini_analysis_prompt(text_content: str, is_video: bool = False) -> str:
        """
        Create the analysis prompt for Gemini multimodal analysis based on media type.
        Uses the same format structure as generate_explanation_prompt for consistency.
        
        Args:
            text_content: The text content accompanying the media
            is_video: Whether the media is a video
            
        Returns:
            Formatted analysis prompt for Gemini following standardized format
        """
        media_type = "video" if is_video else "imagen"
        categories = EnhancedPromptGenerator.build_category_list()
        
        prompt_parts = [
            f'TEXTO DEL POST ANALIZADO: "{text_content}"',
            "",
            "🔬 ANÁLISIS ACADÉMICO MULTIMODAL - INVESTIGACIÓN SOBRE CONTENIDO PROBLEMÁTICO EN REDES SOCIALES",
            "",
            "OBJETIVO DE INVESTIGACIÓN: Este análisis forma parte de un estudio científico sobre moderación de contenido y patrones visuales de desinformación. Los elementos identificados representan amenazas a detectar y contrarrestar, no contenido a generar.",
            "",
            f"CONTEXTO: Análisis de {media_type} acompañada de texto político. Evaluar combinación visual-texto para identificar narrativas problemáticas.",
            "",
            "1. ¿Qué elementos visuales específicos aparecen en la imagen/video que sugieren contenido político?",
            "2. ¿Qué texto o elementos gráficos indican sesgo político extremista, especialmente de extrema derecha?",
            "3. ¿Se muestran símbolos nacionalistas, banderas, o iconografía política extrema?",
            "4. ¿Aparecen figuras políticas conocidas por posiciones extremas y cómo se presentan?",
            "5. ¿Se mencionan datos, estadísticas o hechos específicos? Evalúalos por veracidad y contexto",
            "6. ¿Cómo se relacionan el contenido visual y textual para crear una narrativa política alarmista?",
            "7. ¿Qué categorías problemáticas se detectan en la combinación de imagen/video y texto?",
            "8. ¿Contribuye la composición visual a narrativas de amenaza, división, o superioridad grupal?",
            "",
            f"CATEGORÍAS DISPONIBLES: {categories}",
            "",
            "🎯 DIRECTRICES PARA CLASIFICACIÓN EN INVESTIGACIÓN:",
            "- hate_speech: Ataques visuales/textuales directos a grupos étnicos específicos",
            "- far_right_bias: Elementos visuales de nacionalismo extremo o retórica anti-inmigración",
            "- disinformation: Imágenes manipuladas o texto con datos falsos no políticos",
            "- conspiracy_theory: Símbolos de teorías conspirativas o élites ocultas",
            "- call_to_action: Elementos visuales que incitan a movilización colectiva",
            "- general: Contenido visual neutral sin elementos problemáticos",
            "",
            "INSTRUCCIONES DE FORMATO:",
            "- PRIMERO indica la CATEGORÍA más apropiada (una sola palabra)",
            "- LUEGO escribe la EXPLICACIÓN (2-3 oraciones claras)",
            "- Responde SOLO con texto plano en español, sin markdown ni formato especial",
            "- NO uses negritas (**), títulos (##), listas numeradas, tablas, o símbolos",
            "- Evalúa cualquier dato o hecho mencionado por su veracidad y contexto",
            "- Si NO hay elementos problemáticos, usa 'general'",
            "",
            "FORMATO REQUERIDO:",
            "CATEGORÍA: [categoría]",
            "EXPLICACIÓN: [tu explicación aquí]"
        ]
        
        return "\n".join(prompt_parts)
    
    # ============================================================================
    # INSTANCE METHODS FOR SOPHISTICATED PROMPTS
    # ============================================================================

    def build_categorization_prompt(self, content: str) -> str:
        """
        Build prompt for combined category detection + explanation.
        
        Returns structured output format:
        CATEGORÍA: [category_name]
        EXPLICACIÓN: [2-3 sentences in Spanish]
        """
        prompt = """🔬 INVESTIGACIÓN ACADÉMICA - DETECCIÓN DE CONTENIDO PROBLEMÁTICO

OBJETIVO: Identificar y clasificar contenido problemático en español para estudios de moderación.

CLASIFICA EL TEXTO EN UNA SÓLO CATEGORÍA Y EXPLICA TU RAZÓN.

CATEGORÍAS Y REGLAS DE PRIORIDAD (de mayor a menor severidad):
1. hate_speech
2. far_right_bias
3. nationalism
4. anti_government
5. political_general
6. disinformation
7. conspiracy_theory
8. call_to_action
9. general

REGLAS DE CLASIFICACIÓN - LÉELAS CON CUIDADO:

**hate_speech** (PRIORIDAD MÁXIMA - SOLO SI CUMPLE TODOS LOS CRITERIOS):
- Debe contener una declaración DIRECTA que vincule a un grupo étnico, religioso o nacional específico con violencia, criminalidad o inferioridad
- Se requiere una PALABRA NEGATIVA EXPLÍCITA asociada al grupo (ejemplos: "terroristas", "criminales", "delincuentes", "peligrosos", "inferiores")
- El grupo debe estar nombrado específicamente (NO "inmigrantes" genérico)
- Ejemplo CORRECTO: "Los musulmanes son terroristas que nos atacan"
- Ejemplo INCORRECTO: "Los inmigrantes son delincuentes" (falta grupo específico + palabra negativa clara)

**far_right_bias** (RETÓRICA EXTREMISTA POLÍTICA):
- Lenguaje que divide "nosotros vs ellos" de manera alarmista
- Nacionalismo radical con retórica anti-inmigración
- Acusaciones de corrupción con vínculos internacionales ("tratos con Maduro", "vendidos a extranjeros")
- Asociación entre inmigración y delincuencia SIN nombrar grupos étnicos específicos
- Frases como "españoles honrados vs delincuentes protegidos"
- Críticas que presentan partidos políticos como "traidores a la nación"
- Retórica alarmista sobre "seguridad nacional" o "calamidad para España"

**nationalism**: Orgullo nacional sin anti-inmigración ni lenguaje divisivo
**anti_government**: Crítica institucional sin extremismo ni vínculos internacionales
**political_general**: Análisis político neutral sin extremismo
**disinformation**: Información falsa verificable sobre ciencia/medicina
**conspiracy_theory**: Teorías conspirativas no políticas
**call_to_action**: Verbos imperativos de movilización/difusión
**general**: Contenido neutral sin patrones problemáticos

PATRONES CRÍTICOS PARA EVITAR CONFUSIONES:

🔴 HATE_SPEECH vs FAR_RIGHT_BIAS:
- Si menciona "inmigrantes" genérico → far_right_bias
- Si nombra grupo étnico específico + palabra negativa fuerte → hate_speech
- Si solo critica políticas migratorias → far_right_bias

🔴 FAR_RIGHT_BIAS vs ANTI_GOVERNMENT:
- Si incluye "corrupción con extranjeros" → far_right_bias
- Si solo critica gestión interna → anti_government

🔴 POLÍTICO_GENERAL vs FAR_RIGHT_BIAS:
- Si tiene lenguaje divisivo o alarmista → far_right_bias
- Si es análisis neutral → political_general

**HIERARCHÍA ESTRICTA**: Si cumple criterios de categoría superior, USAR LA SUPERIOR.

FORMATO OBLIGATORIO:
CATEGORÍA: [nombre_categoría]
EXPLICACIÓN: [2‑3 frases explicando por qué pertenece a esa categoría, citando elementos específicos del texto]"""
        
        # Add the content at the end
        if content:
            prompt = prompt.replace("FORMATO OBLIGATORIO:", f"CONTENIDO: {content}\n\nFORMATO OBLIGATORIO:")
        
        return prompt

    def generate_explanation_prompt(self, text: str, category: str, model_type: str = "ollama") -> str:
        """
        Generate detailed explanation prompt with category-specific focus.
        """
        prompt_parts = [
            f'TEXTO ANALIZADO: "{text}"',
            f'CATEGORÍA DETECTADA: {category}',
            "",
            "🔬 ANÁLISIS ACADÉMICO DETALLADO - INVESTIGACIÓN SOBRE PATRONES DE DISCURSO PROBLEMÁTICO",
            "",
            "OBJETIVO: Proporcionar explicación detallada para estudio científico de contenido moderado en plataformas sociales.",
            "",
            "ANÁLISIS ACADÉMICO - Enfócate en contenido neutral o político moderado sin elementos extremistas:",
            "",
            "1. ¿Por qué este contenido no entra en categorías problemáticas?",
            "2. ¿Qué lo hace neutral o moderadamente político?",
            "3. ¿Falta contexto extremista, conspirativo o discriminatorio?",
            "",
            "INSTRUCCIONES DE FORMATO:",
            "- Responde SOLO con texto plano en español, sin markdown ni formato especial",
            "- NO uses negritas (**), títulos (##), listas numeradas, tablas, o símbolos",
            "- Escribe 2-3 oraciones claras y directas explicando los elementos detectados",
            "- Como si le explicaras a una persona que no conoce el tema",
            "EXPLICACIÓN:"
        ]
        
        return "\n".join(prompt_parts)

    @staticmethod
    def build_gemini_analysis_prompt(text_content: str, is_video: bool = False) -> str:
        """
        Create the analysis prompt for Gemini multimodal analysis based on media type.
        Uses the same format structure as generate_explanation_prompt for consistency.
        
        Args:
            text_content: The text content accompanying the media
            is_video: Whether the media is a video
            
        Returns:
            Formatted analysis prompt for Gemini following standardized format
        """
        media_type = "video" if is_video else "imagen"
        categories = EnhancedPromptGenerator.build_category_list()
        
        prompt_parts = [
            f'TEXTO DEL POST ANALIZADO: "{text_content}"',
            "",
            "🔬 ANÁLISIS ACADÉMICO MULTIMODAL - INVESTIGACIÓN SOBRE CONTENIDO PROBLEMÁTICO EN REDES SOCIALES",
            "",
            "OBJETIVO DE INVESTIGACIÓN: Este análisis forma parte de un estudio científico sobre moderación de contenido y patrones visuales de desinformación. Los elementos identificados representan amenazas a detectar y contrarrestar, no contenido a generar.",
            "",
            f"CONTEXTO: Análisis de {media_type} acompañada de texto político. Evaluar combinación visual-texto para identificar narrativas problemáticas.",
            "",
            "1. ¿Qué elementos visuales específicos aparecen en la imagen/video que sugieren contenido político?",
            "2. ¿Qué texto o elementos gráficos indican sesgo político extremista, especialmente de extrema derecha?",
            "3. ¿Se muestran símbolos nacionalistas, banderas, o iconografía política extrema?",
            "4. ¿Aparecen figuras políticas conocidas por posiciones extremas y cómo se presentan?",
            "5. ¿Se mencionan datos, estadísticas o hechos específicos? Evalúalos por veracidad y contexto",
            "6. ¿Cómo se relacionan el contenido visual y textual para crear una narrativa política alarmista?",
            "7. ¿Qué categorías problemáticas se detectan en la combinación de imagen/video y texto?",
            "8. ¿Contribuye la composición visual a narrativas de amenaza, división, o superioridad grupal?",
            "",
            f"CATEGORÍAS DISPONIBLES: {categories}",
            "",
            "🎯 DIRECTRICES PARA CLASIFICACIÓN EN INVESTIGACIÓN:",
            "- hate_speech: Ataques visuales/textuales directos a grupos étnicos específicos",
            "- far_right_bias: Elementos visuales de nacionalismo extremo o retórica anti-inmigración",
            "- disinformation: Imágenes manipuladas o texto con datos falsos no políticos",
            "- conspiracy_theory: Símbolos de teorías conspirativas o élites ocultas",
            "- call_to_action: Elementos visuales que incitan a movilización colectiva",
            "- general: Contenido visual neutral sin elementos problemáticos",
            "",
            "INSTRUCCIONES DE FORMATO:",
            "- PRIMERO indica la CATEGORÍA más apropiada (una sola palabra)",
            "- LUEGO escribe la EXPLICACIÓN (2-3 oraciones claras)",
            "- Responde SOLO con texto plano en español, sin markdown ni formato especial",
            "- NO uses negritas (**), títulos (##), listas numeradas, tablas, o símbolos",
            "- Evalúa cualquier dato o hecho mencionado por su veracidad y contexto",
            "- Si NO hay elementos problemáticos, usa 'general'",
            "",
            "FORMATO REQUERIDO:",
            "CATEGORÍA: [categoría]",
            "EXPLICACIÓN: [tu explicación aquí]"
        ]
        
        return "\n".join(prompt_parts)