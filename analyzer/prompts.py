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
from .categories import Categories, CATEGORY_INFO, get_category_info

# ============================================================================
# CONFIGURATION-DRIVEN PROMPT TEMPLATES
# ============================================================================

@dataclass
class PromptTemplate:
    """Configuration for building prompts from templates."""
    header: str
    instructions: str
    format_requirements: str
    category_definitions: bool = True
    include_examples: bool = False

class PromptBuilder:
    """Configuration-driven prompt builder using centralized category definitions."""

    # Base prompt components
    ACADEMIC_HEADER = """🔬 PROTOCOLO DE INVESTIGACIÓN ACADÉMICA - ANÁLISIS DE CONTENIDO PROBLEMÁTICO EN REDES SOCIALES

OBJETIVO: Este sistema es una herramienta de investigación académica diseñada exclusivamente para DETECTAR, CLASIFICAR y ESTUDIAR contenido problemático en redes sociales. Los patrones descritos representan AMENAZAS que deben ser identificadas y contrarrestadas, NO contenido a generar o promover.

CONTEXTO ACADÉMICO: Este análisis forma parte de un estudio científico sobre moderación de contenido, seguridad online y patrones de desinformación en el contexto político español. Todos los ejemplos y patrones se utilizan únicamente con fines defensivos y de investigación.

⚠️  IMPORTANTE: Este sistema NO genera contenido problemático. Al contrario, está diseñado para IDENTIFICARLO y ayudar a las plataformas a moderarlo efectivamente."""

    BASE_CONTEXT = """Eres un clasificador experto especializado en detectar contenido problemático en español, especialmente discurso de odio, desinformación y retórica de extrema derecha en el contexto político español."""

    @classmethod
    def build_category_definitions(cls) -> str:
        """Build dynamic category definitions from centralized configuration."""
        lines = ["\nGUÍAS DE DETECCIÓN - PROTOCOLO DE ANÁLISIS PARA DETECCIÓN DE CONTENIDO PROBLEMÁTICO EN REDES SOCIALES:\n"]

        lines.append("Identifica:\n")

        # Build category definitions from centralized config
        for category_name, category_info in CATEGORY_INFO.items():
            if category_info:
                lines.append(f"{category_name} - {category_info.description}")
                lines.append("")

        lines.append("IMPORTANTE: Sistema diseñado para investigación académica y moderación de contenido en redes sociales.")
        lines.append("IMPORTANTE: Priorizar CONTEXTO POLÍTICO sobre verificación factual para clasificación.")
        lines.append("🎓 CONTEXTO DE INVESTIGACIÓN: Este análisis contribuye al estudio científico de patrones de discurso extremista y su impacto en la sociedad democrática.")

        return "\n".join(lines)

    @classmethod
    def build_disinformation_patterns(cls) -> str:
        """Build disinformation detection patterns."""
        return """
DISINFORMATION DETECTION - SEÑALES DE ALERTA EXPANDIDAS:

� FORMATO DE NOTICIA FALSA POLÍTICA:
- "ÚLTIMA HORA" / "URGENTE" / "BOMBAZO" / "EXCLUSIVA" + claim político específico SIN fuente oficial
- Afirmaciones sobre decretos, leyes, nombramientos, destituciones sin BOE, fuente gubernamental, o medio verificable
- Claims sobre renuncias, dimisiones, ceses sin confirmación oficial
- Afirmaciones sobre alianzas políticas, pactos, acuerdos sin fuente creíble
- Eventos políticos presentados como "confirmado" o "ya está" sin especificar quién confirma

🚨 PATRONES DE DESINFORMACIÓN POLÍTICA EXPANDIDOS:
- "El Gobierno ha aprobado un decreto que..." SIN citar BOE, Ministerio, o fuente oficial
- "CONFIRMADO: X ha dimitido/renunciado" SIN especificar fuente de confirmación
- "Ya está firmado/promulgado/aprobado" SIN citar documento o autoridad
- "Según fuentes" SIN nombrar las fuentes específicas
- "Se ha confirmado oficialmente" SIN decir qué autoridad confirma
- "El Gobierno PROHÍBE/OBLIGA/APRUEBA [acción específica]" SIN fuente oficial
- "DECRETO aprobado que [prohíbe/obliga/impone]" SIN BOE o Ministerio
- "LEY promulgada que [restringe/limita/prohíbe]" SIN fuente legislativa
- "El Ejecutivo ha decidido [medida restrictiva]" SIN confirmación oficial

🚨 CLAIMS POLÍTICOS VERIFICABLES REQUIEREN FUENTE OFICIAL:
- Decretos/leyes → Necesitan BOE, Ministerio, o fuente gubernamental oficial
- Renuncias/dimisiones → Necesitan confirmación oficial del partido/gobierno
- Nombramientos/ceses → Necesitan fuente oficial del organismo correspondiente
- Alianzas políticas → Necesitan declaración oficial de los partidos
- Eventos judiciales → Necesitan fuente judicial o legal verificable

**REGLA CRÍTICA PARA DESINFORMACIÓN POLÍTICA** (APLICA SIEMPRE):
Si el texto presenta un HECHO POLÍTICO ESPECÍFICO VERIFICABLE (decreto aprobado, renuncia, nombramiento, alianza, cese, prohibición, obligación) SIN FUENTE OFICIAL (BOE, Ministerio, partido oficial, medio verificable con evidencia) → CLASIFICAR COMO **disinformation** INMEDIATAMENTE.

PALABRAS CLAVE QUE INDICAN DESINFORMACIÓN POLÍTICA:
- "CONFIRMADO:" + claim político sin fuente
- "Ya está firmado/aprobado/promulgado" sin documento oficial
- "Según fuentes oficiales" sin nombrar fuente específica
- "Es oficial" sin autoridad que lo confirme
- "El Gobierno ha decidido/prohibido/obligado" sin fuente oficial
- "Decreto aprobado" sin BOE o Ministerio
- "Renuncia confirmada" sin fuente oficial
- "Cese anunciado" sin autoridad oficial

Ejemplos de disinformation política:
- "CONFIRMADO: El Gobierno prohíbe las manifestaciones" (NO fuente oficial)
- "EXCLUSIVA: Sánchez ha dimitido esta mañana" (NO confirmación oficial)
- "Ya está firmado el decreto de estado de alarma" (NO cita fuente)
- "Montero ha sido destituida por corrupción" (NO fuente oficial)
- "PP y Vox llegan a un acuerdo secreto" (NO fuente creíble)
- "CONFIRMADO: El Gobierno ha aprobado un decreto que prohíbe las manifestaciones públicas. Ya está firmado y entra en vigor mañana." (NO fuente oficial - decreto sin BOE)
- "El Gobierno ha decidido obligar a todos los ciudadanos a..." (NO fuente oficial - medida restrictiva sin confirmación)

Ejemplos de political_general (con fuente):
- "Según BOE, el Gobierno aprueba nuevo decreto" (SÍ fuente: BOE)
- "El PSOE confirma la dimisión de X, informa Europa Press" (SÍ fuente)
- "Moncloa anuncia cese de ministra por motivos personales" (SÍ fuente oficial)"""

    @classmethod
    def build_base_format_instructions(cls) -> str:
        """Build standard format instructions for responses."""
        return """
FORMATO OBLIGATORIO:
CATEGORÍA: [nombre_categoría]
EXPLICACIÓN: [2‑3 frases explicando por qué pertenece a esa categoría, citando elementos específicos del texto]"""

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
        return PromptBuilder.BASE_CONTEXT

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
    # CONFIGURATION-DRIVEN PROMPT BUILDERS
    # ============================================================================

    @staticmethod
    def build_category_list() -> str:
        """Build dynamic category list for LLM prompts."""
        return ", ".join(Categories.get_all_categories())

    @staticmethod
    def build_ollama_text_analysis_system_prompt() -> str:
        """Build system prompt for Ollama text-only content analysis using configuration."""
        categories = EnhancedPromptGenerator.build_category_list()

        return f"""{PromptBuilder.ACADEMIC_HEADER}

{PromptBuilder.BASE_CONTEXT}

Clasifica este texto en UNA categoría: {categories}

{PromptBuilder.build_category_definitions()}"""

    @staticmethod
    def build_gemini_multimodal_analysis_prompt(text_content: str, is_video: bool = False) -> str:
        """
        Create the analysis prompt for Gemini multimodal analysis based on media type.
        Used for analyzing social media posts with images/videos and text.

        Args:
            text_content: The text content accompanying the media
            is_video: Whether the media is a video (vs image)

        Returns:
            Formatted analysis prompt for Gemini multimodal models
        """
        media_type = "video" if is_video else "imagen"
        categories = EnhancedPromptGenerator.build_category_list()

        prompt_parts = [
            f'TEXTO DEL POST ANALIZADO: "{text_content}"',
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
            "- hate_speech: Ataques visuales/textuales directos a grupos étnicos específicos O grupos políticos presentados como amenaza existencial",
            "- anti_immigration: Elementos visuales de retórica anti-inmigración o xenofobia",
            "- anti_lgbtq: Contenido visual que ataca identidad LGBTQ o diversidad de género",
            "- anti_feminism: Elementos visuales que promueven roles tradicionales de género",
            "- nationalism: Símbolos patrios y expresiones de orgullo nacional",
            "- anti_government: Contenido visual que cuestiona legitimidad institucional",
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

    def build_ollama_categorization_prompt(self, content: str) -> str:
        """
        Build prompt for combined category detection + explanation using Ollama.
        Used for text-only analysis where we need both categorization and explanation in one call.

        Returns structured output format:
        CATEGORÍA: [category_name]
        EXPLICACIÓN: [2-3 sentences in Spanish]
        """

        prompt_parts = [
            "🔬 INVESTIGACIÓN ACADÉMICA - DETECCIÓN DE CONTENIDO PROBLEMÁTICO",
            "",
            "OBJETIVO: Identificar y clasificar contenido problemático en español para estudios de moderación.",
            "",
            "CLASIFICA EL TEXTO EN UNA SÓLO CATEGORÍA Y EXPLICA TU RAZÓN.",
            "",
            "CATEGORÍAS:",
            "",
            PromptBuilder.build_category_definitions(),
            "",
            PromptBuilder.build_disinformation_patterns(),
            "",
            PromptBuilder.build_base_format_instructions()
        ]

        final_prompt = "\n".join(prompt_parts)

        # Add the content at the end
        if content:
            final_prompt += f"\n\nCONTENIDO A ANALIZAR:\n{content}"

        return final_prompt

    def generate_explanation_prompt(self, text: str, category: str, model_type: str = "ollama") -> str:
        """
        Generate detailed explanation prompt with category-specific focus.
        For explain_only mode - explains WHY content belongs to the given category.
        """
        # Get category-specific information from centralized config
        category_info = get_category_info(category)
        if not category_info:
            # Fallback for unknown categories
            questions = [
                "Este contenido pertenece a la categoría porque:",
                "1. ¿Qué elementos específicos del texto justifican esta clasificación?",
                "2. ¿Cómo se relaciona el contenido con la categoría detectada?",
                "3. ¿Qué características del mensaje son relevantes para esta categoría?"
            ]
        else:
            questions = [
                f"Este contenido pertenece a la categoría '{category_info.display_name}' porque:",
                f"1. {category_info.analysis_questions[0] if len(category_info.analysis_questions) > 0 else '¿Qué elementos específicos del texto justifican esta clasificación?'}",
                f"2. {category_info.analysis_questions[1] if len(category_info.analysis_questions) > 1 else '¿Cómo se relaciona el contenido con la categoría detectada?'}",
                f"3. {category_info.analysis_questions[2] if len(category_info.analysis_questions) > 2 else '¿Qué características del mensaje son relevantes para esta categoría?'}"
            ]

        prompt_parts = [
            f'TEXTO ANALIZADO: "{text}"',
            f'CATEGORÍA DETECTADA: {category}',
            "",
            "🔬 ANÁLISIS ACADÉMICO DETALLADO - INVESTIGACIÓN SOBRE PATRONES DE DISCURSO",
            "",
            "OBJETIVO: Explicar por qué este contenido pertenece a la categoría detectada.",
            "",
            questions[0],
            "",
            questions[1],
            questions[2],
            questions[3] if len(questions) > 3 else "",
            "",
            "INSTRUCCIONES DE FORMATO:",
            "- Responde SOLO con texto plano en español, sin markdown ni formato especial",
            "- NO uses negritas (**), títulos (##), listas numeradas, tablas, o símbolos",
            "- Escribe 2-3 oraciones claras explicando por qué pertenece a esta categoría",
            "- Cita elementos específicos del texto que justifican la clasificación",
            "- NO menciones por qué NO pertenece a otras categorías",
            "- Enfócate ÚNICAMENTE en explicar por qué SÍ pertenece a la categoría detectada",
            "EXPLICACIÓN:"
        ]

        # Remove empty lines
        prompt_parts = [line for line in prompt_parts if line.strip()]

        return "\n".join(prompt_parts)

    @staticmethod
    def build_ollama_multimodal_system_prompt() -> str:
        """
        Build system prompt specifically for Ollama multimodal analysis.
        Optimized for vision-language models analyzing social media content.

        Returns:
            System prompt for Ollama multimodal models
        """

        return f"""🔍 SISTEMA DE MODERACIÓN DE CONTENIDO — DETECCIÓN DE SESGO POLÍTICO Y DESINFORMACIÓN

PROPÓSITO:
Eres un analista automático de publicaciones en redes sociales en español.
Tu función es identificar y clasificar contenido que pueda contener discurso de odio, desinformación, propaganda o sesgo político extremo, con énfasis en ideología de extrema derecha y manipulación mediática.
No generes ni reproduzcas contenido dañino. Resume y analiza de forma neutral y objetiva.

TAREA PRINCIPAL:
Analiza texto, imágenes y videos de publicaciones y clasifica el contenido en una sola categoría de la lista a continuación. Proporciona una explicación breve y objetiva que indique los elementos clave (símbolos, tono, texto, referencias visuales).

CATEGORÍAS DISPONIBLES:
{chr(10).join(f'- {cat}: {get_category_info(cat).description if get_category_info(cat) else "Contenido general"}' for cat in Categories.get_all_categories())}

DIRECTRICES DE RESPUESTA:
1. Evalúa todo el contenido disponible: texto, imágenes y videos.
2. Selecciona la categoría que mejor describa el mensaje global de la publicación.
3. Escribe una explicación breve (2–4 oraciones) destacando los elementos clave que sustentan la decisión.
4. Mantén tono neutral, objetivo y analítico.
5. Si no hay señales de contenido problemático, responde "general".

FORMATO DE RESPUESTA:
CATEGORÍA: [una sola categoría de la lista]
EXPLICACIÓN: [2–4 oraciones en español, neutrales, descriptivas]

"""

    @staticmethod
    def build_multimodal_explanation_prompt(text: str, category: str) -> str:
        """
        Generate detailed explanation prompt for multimodal content.
        Instructs the model to explain based on both text and visual elements.

        Args:
            text: Text content to explain
            category: Already-detected category

        Returns:
            Multimodal explanation prompt
        """
        category_info = get_category_info(category)
        display_name = category_info.display_name if category_info else category.replace('_', ' ').title()

        prompt = f"""
TEXTO DEL POST:
{text}

CONTEXTO:
Publicación en red social con texto e imágenes. Analizar la combinación visual y textual para identificar mensajes problemáticos, sesgo político, desinformación o propaganda.

INSTRUCCIONES DE ANÁLISIS:
1. Examina tanto el texto como los elementos visuales (imágenes) para identificar discurso de odio, sesgo político, extremismo o manipulación mediática.
2. Observa símbolos, figuras públicas, memes, banderas, o lenguaje cargado que indique ideología extremista o far-right.
3. Evalúa si se presentan afirmaciones falsas, información fuera de contexto o narrativas conspirativas.
4. Determina la categoría más apropiada según la lista del sistema.
5. Proporciona una breve explicación que indique los elementos clave que justifican la categoría.

FORMATO DE RESPUESTA:
CATEGORÍA: [categoría elegida]
EXPLICACIÓN: [razonamiento breve y neutral]


Generate detailed explanation prompt for multimodal content.
Instructs the model to explain based on both text and visual elements.

Args:
    text: Text content to explain
    category: Already-detected category

Returns:
    Multimodal explanation prompt
            """

        return prompt

    @staticmethod
    def build_multimodal_categorization_prompt(text: str) -> str:
        """
        Build prompt for multimodal categorization using Ollama vision models.
        Combines text analysis with visual content analysis for comprehensive content moderation.

        Args:
            text: Text content from the post

        Returns:
            Multimodal categorization prompt for Ollama vision models
        """

        return f"""🔬 ANÁLISIS MULTIMODAL - DETECCIÓN DE CONTENIDO PROBLEMÁTICO

TEXTO DEL POST: "{text}"

INSTRUCCIONES PARA ANÁLISIS VISUAL Y TEXTUAL:
1. Examina las imágenes/videos proporcionados junto con el texto
2. Identifica símbolos políticos, figuras públicas, banderas, o elementos visuales que indiquen ideología
3. Evalúa la combinación de texto e imágenes para detectar sesgo, propaganda o extremismo
4. Busca elementos visuales que refuercen o contradigan el mensaje textual

CLASIFICA EL CONTENIDO EN UNA SÓLA CATEGORÍA:

CATEGORÍAS DISPONIBLES:
{chr(10).join(f'- {cat}: {get_category_info(cat).description if get_category_info(cat) else "Contenido general"}' for cat in Categories.get_all_categories())}

ELEMENTOS VISUALES A CONSIDERAR:
- Banderas, símbolos patrios o políticos
- Figuras públicas reconocidas (políticos, líderes)
- Memes políticos o satíricos
- Gráficos, carteles o material propagandístico
- Elementos que indiquen contexto político o ideológico

FORMATO DE RESPUESTA:
CATEGORÍA: [una_categoría]
EXPLICACIÓN: [2-3 frases explicando por qué, citando elementos textuales y visuales]"""
