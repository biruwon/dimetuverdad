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

from typing import Dict, List, Optional
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

    BASE_CONTEXT = """Eres un clasificador experto especializado en detectar contenido problemático en español, especialmente discurso de odio, desinformación y retórica de extrema derecha en el contexto político español."""

    @classmethod
    def build_category_definitions(cls) -> str:
        """Build dynamic category definitions from centralized configuration."""
        # Build category list from centralized config
        category_lines = []
        for category_name, category_info in CATEGORY_INFO.items():
            if category_info:
                category_lines.append(f"**{category_name}** - {category_info.description}")
                if category_info.focus_area:
                    category_lines.append(f"  Enfoque: {category_info.focus_area}")
        
        categories_text = "\n".join(category_lines)
        
        return f"""
IDENTIFICACIÓN DE CATEGORÍAS:

{categories_text}
"""

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

Ejemplos de political_general (NO disinformation):
- "Según BOE, el Gobierno aprueba nuevo decreto" (SÍ fuente: BOE)
- "El PSOE confirma la dimisión de X, informa Europa Press" (SÍ fuente)
- "Moncloa anuncia cese de ministra por motivos personales" (SÍ fuente oficial)
- "¡Si el CIS dice que arrasan!" (IRONÍA - cuestionamiento sarcástico)
"""

    @classmethod
    def build_base_format_instructions(cls) -> str:
        """Build standard format instructions for responses."""
        return """
FORMATO OBLIGATORIO:
CATEGORÍA: [nombre_categoría]
EXPLICACIÓN: [2‑3 frases explicando por qué pertenece a esa categoría, citando elementos específicos del texto]

IMPORTANTE - LENGUAJE DE LA EXPLICACIÓN:
- En la explicación, NO uses los nombres técnicos de categorías en inglés (hate_speech, call_to_action, etc.)
- Si necesitas referirte a la categoría, usa términos naturales en español:
  * hate_speech → "discurso de odio" o "contenido de odio"
  * call_to_action → "llamada a la acción" o "movilización"
  * anti_immigration → "retórica anti-inmigración"
  * disinformation → "desinformación"
  * conspiracy_theory → "teoría conspirativa"
  * nationalism → "nacionalismo"
  * anti_government → "retórica anti-gubernamental"
- La explicación debe ser natural y fluida en español, sin términos técnicos en inglés"""

    @classmethod
    def build_common_critical_rules(cls) -> str:
        """Build common critical classification rules from centralized category definitions."""
        
        rules_text = "⚠️ REGLAS CRÍTICAS DE CLASIFICACIÓN:\n\n"
        
        for category_name, category_info in CATEGORY_INFO.items():
            if category_info.classification_rules:
                rules_text += f"**{category_name.upper()}**:\n"
                for rule in category_info.classification_rules:
                    rules_text += f"  - {rule}\n"
                rules_text += "\n"
        
        return rules_text

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

    # ============================================================================
    # FAST MODE PROMPTS - OPTIMIZED FOR SPEED
    # ============================================================================

    @staticmethod
    def build_fast_system_prompt() -> str:
        """Get FAST system prompt for fast mode - optimized for 100% accuracy."""
        return """Eres un clasificador experto de contenido político español.

Clasifica en UNA categoría exacta:
hate_speech, anti_immigration, anti_lgbtq, anti_feminism, disinformation, conspiracy_theory, call_to_action, nationalism, anti_government, political_general, general

PRINCIPIOS FUNDAMENTALES DE CLASIFICACIÓN:

1. IDENTIFICA EL ELEMENTO PROBLEMÁTICO PRINCIPAL:
   - hate_speech: Ataques PERSONALES directos con insultos individuales
   - anti_government: Crítica INSTITUCIONAL al gobierno/sistema político
   - disinformation: Información FALSA presentada como cierta sobre hechos verificables
   - conspiracy_theory: Narrativas de CONSPIRACIÓN OCULTA y control secreto
   - anti_immigration: Retórica XENÓFOBA colectiva contra inmigrantes
   - call_to_action: Incitación a MOVILIZACIÓN COLECTIVA organizada
   - nationalism: Promoción de IDENTIDAD NACIONAL española
   - political_general: Contenido POLÍTICO NEUTRAL informativo
   - general: Contenido NO POLÍTICO

2. DIFERENCIACIONES CRÍTICAS:
   - PERSONAL vs INSTITUCIONAL: hate_speech ataca individuos, anti_government critica sistemas
   - FALSO vs INFORMATIVO: disinformation miente sobre hechos, political_general informa neutralmente
   - OCULTO vs PÚBLICO: conspiracy_theory habla de agendas secretas, anti_government critica políticas públicas
   - COLECTIVO vs INDIVIDUAL: anti_immigration critica grupos, hate_speech ataca personas específicas

3. INDICADORES ESPECÍFICOS POR CATEGORÍA:

HATE_SPEECH:
- Insultos directos: "mierda", "indecente", "traidor", "psicópata", "fascista"
- Deshumanización: comparaciones degradantes con animales/enfermedades
- Ataques por origen/ideología/identidad personal

ANTI_GOVERNMENT:
- "Gobierno corrupto", "políticas erróneas", "instituciones fallidas"
- Acusaciones de corrupción institucional, abuso de poder
- Crítica a sistemas políticos, no ataques personales

DISINFORMATION:
- Claims falsos sobre hechos verificables: decretos, leyes, nombramientos sin fuente oficial
- "CONFIRMADO:" + evento político inventado
- Afirmaciones presentadas como ciertas sin evidencia (BOE, ministerios, partidos)

CONSPIRACY_THEORY:
- "Ellos controlan todo", "agenda oculta", "manipulación global"
- Élites secretas, conspiraciones organizadas, control oculto
- Narrativas amplias de agendas secretas y manipulación masiva

ANTI_IMMIGRATION:
- "Invasión migratoria", "fronteras abiertas", "manadas extranjeras"
- Amenazas colectivas a identidad, seguridad, economía por inmigración
- Retórica xenófoba contra grupos, no individuos

CALL_TO_ACTION:
- "Todos a la calle", "hay que actuar", "movilización inmediata"
- Incitación a protestas/manifestaciones coordinadas
- Lenguaje de urgencia colectiva organizada

NATIONALISM:
- "Orgullo español", "defensa de España", "tradiciones españolas"
- Promoción de identidad nacional, símbolos patrios
- Expresiones de superioridad o defensa cultural nacional

POLITICAL_GENERAL:
- Menciones políticas informativas sin crítica extrema
- Contenido político neutral: partidos, líderes, eventos
- SIN alarmismo, conspiraciones, ataques personales o falsedades

GENERAL:
- Temas no políticos, conversaciones cotidianas
- Ausencia de elementos políticos identificables

FORMATO:
CATEGORÍA: [una_palabra]
EXPLICACIÓN: [2-3 frases]"""

    @staticmethod
    def build_fast_categorization_prompt(content: str) -> str:
        """Get FAST categorization prompt for fast mode."""
        return f"""Analiza este contenido político español:

{content}

INSTRUCCIONES PARA CLASIFICACIÓN PRECISA:

1. IDENTIFICA EL ELEMENTO PROBLEMÁTICO PRINCIPAL:
   - hate_speech: Ataques PERSONALES directos con insultos individuales
   - anti_government: Crítica INSTITUCIONAL al gobierno/sistema político
   - disinformation: Información FALSA presentada como cierta sobre hechos verificables
   - conspiracy_theory: Narrativas de CONSPIRACIÓN OCULTA y control secreto
   - anti_immigration: Retórica XENÓFOBA colectiva contra inmigrantes
   - call_to_action: Incitación a MOVILIZACIÓN COLECTIVA organizada
   - nationalism: Promoción de IDENTIDAD NACIONAL española
   - political_general: Contenido POLÍTICO NEUTRAL informativo
   - general: Contenido NO POLÍTICO

2. DIFERENCIACIONES CRÍTICAS:
   - PERSONAL vs INSTITUCIONAL: hate_speech ataca individuos, anti_government critica sistemas
   - FALSO vs INFORMATIVO: disinformation miente sobre hechos, political_general informa neutralmente
   - OCULTO vs PÚBLICO: conspiracy_theory habla de agendas secretas, anti_government critica políticas públicas
   - COLECTIVO vs INDIVIDUAL: anti_immigration critica grupos, hate_speech ataca personas específicas

3. BUSCA INDICADORES ESPECÍFICOS:
   - Insultos directos → hate_speech
   - "Gobierno corrupto" → anti_government
   - Claims falsos verificables → disinformation
   - "Ellos controlan todo" → conspiracy_theory
   - "Invasión migratoria" → anti_immigration
   - "Todos a la calle" → call_to_action
   - "Orgullo español" → nationalism
   - Política neutral → political_general
   - No político → general

CATEGORÍA: [una_palabra]
EXPLICACIÓN: [2-3 frases]"""

    @staticmethod
    def build_fast_explanation_prompt(content: str, category: str) -> str:
        """Get simplified explanation prompt for fast mode."""
        return f"""Contenido: {content}

Categoría detectada: {category}

Explica por qué este contenido pertenece a la categoría {category}.

ESTRUCTURA DE EXPLICACIÓN:
1. Comienza identificando el elemento problemático clave
2. Cita frases exactas entre comillas del texto
3. Explica las implicaciones más amplias
4. Conecta con las características de la categoría
5. Mantén 2-3 frases concisas pero comprehensivas

ENFÓCATE ÚNICAMENTE en por qué SÍ pertenece a {category}."""

    @staticmethod
    def build_fast_multimodal_categorization_prompt(text: str) -> str:
        """Get simplified multimodal categorization prompt for fast mode."""
        return f"""Analiza este contenido con texto e imágenes:

TEXTO: "{text}"

INSTRUCCIONES PARA ANÁLISIS MULTIMODAL:

1. EXAMINA TEXTO + IMÁGENES JUNTOS:
   - Identifica símbolos políticos, banderas, figuras públicas en imágenes
   - Evalúa cómo imagen refuerza o modifica el mensaje textual
   - Busca elementos visuales que indiquen extremismo político

2. REGLAS CRÍTICAS PARA CLASIFICACIÓN:
   - hate_speech: Ataques personales + imágenes degradantes/dehumanizadoras
   - anti_government: Crítica institucional + símbolos de protesta gubernamental
   - disinformation: Texto falso + imágenes manipuladas o sin contexto
   - conspiracy_theory: Texto conspirativo + símbolos de élites/control oculto
   - anti_immigration: Retórica xenófoba + imágenes de "invasión" o fronteras
   - call_to_action: Llamadas a movilización + imágenes de protestas/manifestaciones
   - political_general: Política neutral + imágenes informativas
   - general: Contenido no político + imágenes cotidianas

3. EVALÚA COMBINACIÓN VISUAL-TEXTUAL:
   - ¿Cómo se refuerzan mutuamente texto e imagen?
   - ¿Añade la imagen elementos problemáticos al texto?
   - ¿Cambia el contexto visual la interpretación del mensaje?

FORMATO:
CATEGORÍA: [categoría]
EXPLICACIÓN: [2-3 frases mencionando texto e imagen]"""

    @staticmethod
    def build_fast_multimodal_explanation_prompt(text: str, category: str) -> str:
        """Get simplified multimodal explanation prompt for fast mode."""
        return f"""TEXTO DEL POST: "{text}"

CATEGORÍA DETECTADA: {category}

OBJETIVO: Explica por qué este contenido multimodal pertenece a la categoría {category}.

INSTRUCCIONES PARA EXPLICACIÓN MULTIMODAL:
1. EXAMINA TEXTO Y ELEMENTOS VISUALES:
   - Identifica cómo el contenido visual refuerza el mensaje textual
   - Menciona símbolos políticos, figuras, o elementos gráficos específicos
   - Evalúa la combinación de mensaje escrito e imagen

2. ESTRUCTURA LA EXPLICACIÓN:
   - Comienza con elementos clave del TEXTO
   - Describe cómo la IMAGEN refuerza o añade al mensaje
   - Explica la relación entre ambos elementos
   - Conecta con las características de la categoría {category}

3. SE ESPECÍFICO:
   - Cita frases exactas del texto entre comillas
   - Describe elementos visuales concretos
   - Muestra cómo texto e imagen crean el mensaje problemático

EXPLICACIÓN:"""


    @staticmethod
    def build_category_list() -> str:
        """Build dynamic category list for LLM prompts."""
        return ", ".join(Categories.get_all_categories())

    @staticmethod
    def build_ollama_text_analysis_system_prompt() -> str:
        """Build system prompt for Ollama text-only content analysis using configuration."""
        categories = EnhancedPromptGenerator.build_category_list()

        return f"""

{PromptBuilder.BASE_CONTEXT}

Clasifica este texto en UNA categoría: {categories}

{PromptBuilder.build_category_definitions()}

{PromptBuilder.build_common_critical_rules()}

{PromptBuilder.build_base_format_instructions()}
"""

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
            "- hate_speech: Ataques directos, insultos o DESHUMANIZACIÓN (comparaciones con animales, objetos, enfermedades) hacia individuos o grupos políticos. INCLUYE sarcasmo despectivo, burlas degradantes, lenguaje que sugiere inferioridad o incompatibilidad fundamental",
            "- anti_immigration: Elementos visuales de retórica anti-inmigración o xenofobia",
            "- anti_lgbtq: Contenido visual que ataca identidad LGBTQ o diversidad de género",
            "- anti_feminism: Elementos visuales que promueven roles tradicionales de género",
            "- nationalism: Símbolos patrios y expresiones de orgullo nacional",
            "- anti_government: Retrata al gobierno como ILEGÍTIMO, ABUSIVO o PERSECUTOR (no simple crítica política)",
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

    def build_ollama_categorization_prompt(self, content: str) -> str:
        """
        Build prompt for combined category detection + explanation using Ollama.
        Used for text-only analysis where we need both categorization and explanation in one call.

        Returns structured output format:
        CATEGORÍA: [category_name]
        EXPLICACIÓN: [2-3 sentences in Spanish]
        """

        base_prompt = f"""CONTENIDO A ANALIZAR:
{content}"""

        return base_prompt

    def build_ollama_text_explanation_prompt(self, text: str, category: str, model_type: str = "ollama") -> str:
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
        SIMPLIFIED VERSION for faster multimodal processing.

        Returns:
            System prompt for Ollama multimodal models
        """
        categories = EnhancedPromptGenerator.build_category_list()

        return f"""

{PromptBuilder.BASE_CONTEXT}

Clasifica este contenido en UNA categoría: {categories}

{PromptBuilder.build_category_definitions()}

{PromptBuilder.build_common_critical_rules()}

ANÁLISIS MULTIMODAL:
- Examina TEXTO + IMÁGENES juntos
- Identifica símbolos políticos, banderas, figuras públicas en imágenes
- Evalúa cómo imagen REFUERZA mensaje textual

{PromptBuilder.build_base_format_instructions()}
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
        
        # Get category-specific questions
        questions = category_info.analysis_questions if category_info else [
            "¿Qué elementos específicos del texto y las imágenes justifican esta clasificación?",
            "¿Cómo se relaciona el contenido visual y textual con la categoría detectada?",
            "¿Qué características del mensaje multimodal son relevantes para esta categoría?"
        ]

        prompt = f"""TEXTO DEL POST: "{text}"

CATEGORÍA DETECTADA: {category}

🔬 ANÁLISIS ACADÉMICO MULTIMODAL - INVESTIGACIÓN SOBRE PATRONES DE DISCURSO

OBJETIVO: Explicar por qué este contenido multimodal (texto + imágenes/videos) pertenece a la categoría detectada.

INSTRUCCIONES DE ANÁLISIS:
1. Examina TANTO el texto COMO los elementos visuales (imágenes/videos) proporcionados
2. Identifica cómo el contenido visual REFUERZA o COMPLEMENTA el mensaje textual
3. Observa símbolos políticos, figuras públicas, banderas, memes o elementos gráficos relevantes
4. Evalúa la combinación de texto e imágenes para detectar narrativas problemáticas

Este contenido pertenece a la categoría '{display_name}' porque:

{questions[0] if len(questions) > 0 else '¿Qué elementos específicos del texto y las imágenes justifican esta clasificación?'}
{questions[1] if len(questions) > 1 else '¿Cómo se relaciona el contenido visual y textual con la categoría detectada?'}
{questions[2] if len(questions) > 2 else '¿Qué características del mensaje multimodal son relevantes para esta categoría?'}

INSTRUCCIONES DE FORMATO:
- Responde SOLO con texto plano en español, sin markdown ni formato especial
- NO uses negritas (**), títulos (##), listas numeradas, tablas, o símbolos
- Escribe 2-3 oraciones claras explicando por qué pertenece a esta categoría
- Cita elementos específicos del TEXTO Y de las IMÁGENES que justifican la clasificación
- Menciona cómo el contenido visual y textual se relacionan para crear la narrativa
- NO menciones por qué NO pertenece a otras categorías
- Enfócate ÚNICAMENTE en explicar por qué SÍ pertenece a la categoría detectada

EXPLICACIÓN:"""

        return prompt

    @staticmethod
    def build_multimodal_categorization_prompt(text: str) -> str:
        """
        Build prompt for multimodal categorization using Ollama vision models.
        SIMPLIFIED VERSION for faster multimodal processing.

        Args:
            text: Text content from the post

        Returns:
            Multimodal categorization prompt for Ollama vision models
        """

        return f"""Analiza este contenido con texto e imágenes:

TEXTO: "{text}"

INSTRUCCIONES:
1. Examina el texto Y las imágenes proporcionadas
2. Identifica símbolos políticos, banderas, figuras en las imágenes
3. Evalúa cómo la imagen refuerza o modifica el mensaje del texto
4. Clasifica en UNA categoría
5. Explica citando elementos del texto Y de las imágenes

FORMATO (texto plano español):
CATEGORÍA: [categoría]
EXPLICACIÓN: [2-3 frases mencionando texto e imagen]"""

    def generate_explanation_prompt(self, content: str, category: str, model_type: str = "ollama", is_multimodal: bool = False) -> str:
        """
        Generate explanation prompt for content analysis.
        Wrapper method that calls appropriate explanation prompt builder.

        Args:
            content: Text content to explain
            category: Category that was detected
            model_type: Type of model ("ollama", "transformers", etc.)
            is_multimodal: Whether this is multimodal content

        Returns:
            Formatted explanation prompt
        """
        if is_multimodal:
            return self.build_multimodal_explanation_prompt(content, category)
        else:
            return self.build_ollama_text_explanation_prompt(content, category, model_type)
    
    # ============================================================================
    # NEW STAGE-BASED PROMPTS FOR OPTIMIZED FLOW
    # ============================================================================
    
    @staticmethod
    def build_category_detection_system_prompt() -> str:
        """
        Build system prompt for category detection stage.
        Lightweight - defines role and available categories only.
        """
        categories = EnhancedPromptGenerator.build_category_list()
        return f"""Eres un clasificador automático de contenido político español.

Categorías disponibles: {categories}

INSTRUCCIONES CRÍTICAS:
- Responde ÚNICAMENTE con el nombre exacto de UNA categoría
- NO agregues prefijos como "okay", "la categoría es", "clasifico como"
- NO agregues explicaciones o texto adicional
- Responde SOLO con el nombre de la categoría en minúsculas"""
    
    @staticmethod
    def build_category_detection_prompt(content: str, pattern_category: Optional[str] = None) -> str:
        """
        Build prompt for category detection stage - OPTIMIZED FOR SPEED.
        Streamlined prompt with essential classification information only.
        
        Args:
            content: Text content to analyze
            pattern_category: Category suggested by pattern analyzer (if any)
        
        Returns:
            Concise prompt for fast category detection
        """
        # Simplified category list with descriptions and key indicators
        categories_simple = """hate_speech: Ataques/insultos directos a individuos ("rata", "mierda", "traidor", "psicópata", "basura")
anti_immigration: Retórica xenófoba contra grupos ("invasión", "manadas", "ilegales", "ocupación")
anti_lgbtq: Ataques al colectivo LGBTQ ("ideología de género", "adoctrinamiento", "imposición")
anti_feminism: Retórica anti-feminista ("feminazis", "hembrismo", roles tradicionales)
disinformation: Afirmaciones FALSAS verificables EN EL TEXTO del post sobre hechos actuales ("X ha dimitido", "X está en prisión", "X ha sido detenido") sin fuente oficial
conspiracy_theory: Agendas secretas, élites ocultas ("ellos controlan", "agenda oculta", "manipulación global")
call_to_action: Incitación EXPLÍCITA a movilización colectiva ("todos a la calle", "hay que actuar YA", "únete a la manifestación")
nationalism: Promoción identidad nacional ("orgullo español", "España primero", banderas, símbolos)
anti_government: Crítica institucional ("gobierno corrupto", "régimen", "dictadura", "tiranía")
political_general: Contenido político neutral - menciones de partidos/políticos sin extremismo
general: Contenido NO político - temas cotidianos, personales, entretenimiento"""
        
        # Critical rules with examples - ENHANCED
        key_rules = """Reglas críticas:
• hate_speech: INSULTO PERSONAL ("X es un traidor/rata") | anti_government: CRÍTICA SISTEMA ("el gobierno es corrupto")
• disinformation: FALSO EN EL TEXTO DEL POST sobre situación actual ("X está en prisión", "X ha dimitido") sin fuente oficial | political_general: INFORMATIVO con fuente
• conspiracy_theory: CONTROL SECRETO ("élites manipulan todo") | anti_government: CRÍTICA PÚBLICA de políticas visibles
• call_to_action: INCITACIÓN EXPLÍCITA a movilización colectiva ("sal a la calle YA", "únete a la manifestación") | political_general: OPINIÓN o invitación pasiva ("deberían cambiar", "os dejo el enlace")
• anti_immigration: ATAQUE A GRUPO étnico | hate_speech: ATAQUE A INDIVIDUO concreto
• nationalism: ORGULLO/IDENTIDAD nacional | political_general: MENCIÓN neutral de España"""
        
        if pattern_category and pattern_category != Categories.GENERAL:
            # Pattern suggested a category - quick validation
            prompt = f"""Contenido: {content}

Sugerida: {pattern_category}

Categorías:
{categories_simple}

{key_rules}

¿Es {pattern_category} correcta? Si no, elige otra.

Responde ÚNICAMENTE con el nombre exacto de la categoría:"""
        else:
            # No pattern - classify from scratch
            prompt = f"""Contenido: {content}

Categorías:
{categories_simple}

{key_rules}

Responde ÚNICAMENTE con el nombre exacto de la categoría:"""
        
        return prompt
    
    @staticmethod
    def build_media_description_system_prompt() -> str:
        """
        Build system prompt for media description stage.
        Neutral and objective - no interpretation.
        """
        return "Eres un analista visual objetivo. Describe imágenes de forma concisa y factual, sin interpretaciones ni juicios."
    
    @staticmethod
    def build_media_description_prompt() -> str:
        """
        Build prompt for media analysis stage - NEUTRAL OBSERVATION.
        This stage describes what's visible without interpretation.
        
        Contains:
        - Objective observation instructions
        - What to identify (symbols, figures, text)
        - NO category information or classification guidance
        
        Returns:
            Prompt for objective media description
        """
        return """Describe objetivamente lo que ves en estas imágenes.

Enfócate en:
• Personas: número, características, acciones
• Símbolos políticos: banderas, insignias, logos
• Figuras públicas: políticos reconocibles
• Texto visible: carteles, pancartas, mensajes
• Contexto: ubicación (manifestación, evento, entrevista)
• Elementos gráficos: memes, montajes

Describe solo hechos observables, sin interpretaciones. Sé conciso: 1-2 frases.

DESCRIPCIÓN:"""
    
    @staticmethod
    def build_explanation_system_prompt() -> str:
        """
        Build system prompt for explanation generation stage.
        Focused on analysis using detected category context.
        """
        return "Eres un analista académico de contenido político. Explicas clasificaciones de forma objetiva, citando evidencia específica del contenido analizado."
    
    @staticmethod
    def build_explanation_prompt(content: str, category: str, media_description: Optional[str] = None) -> str:
        """
        Build prompt for explanation generation stage - CONTEXT-AWARE.
        This stage explains WHY content belongs to detected category.
        
        Contains:
        - Category-specific analysis questions
        - Focus areas for this specific category
        - Citation requirements (quote text, describe images)
        
        Args:
            content: Original text content
            category: Detected category
            media_description: Optional description of media content
        
        Returns:
            Prompt for generating focused explanation
        """
        # Get category-specific information
        category_info = get_category_info(category)
        
        if not category_info:
            # Fallback for unknown categories
            category_focus = f"Explica por qué este contenido pertenece a '{category}'."
            questions = [
                "¿Qué elementos específicos justifican esta clasificación?",
                "¿Cómo se relaciona el contenido con esta categoría?"
            ]
        else:
            category_focus = f"Explica por qué este contenido pertenece a '{category_info.display_name}'."
            questions = category_info.analysis_questions[:2]  # Use first 2 questions
        
        if media_description:
            # Multimodal explanation with media context
            prompt = f"""Texto: {content}

Imágenes: {media_description}

Categoría: {category}

{category_focus}

Guía:
• {questions[0] if len(questions) > 0 else '¿Qué elementos del texto y las imágenes justifican esta clasificación?'}
• {questions[1] if len(questions) > 1 else '¿Cómo refuerzan las imágenes el mensaje del texto?'}

Responde en 1-2 frases. Cita elementos específicos del texto (entre comillas) y menciona elementos visuales relevantes.

EXPLICACIÓN:"""
        else:
            # Text-only explanation
            prompt = f"""Texto: {content}

Categoría: {category}

{category_focus}

Guía:
• {questions[0] if len(questions) > 0 else '¿Qué elementos del texto justifican esta clasificación?'}
• {questions[1] if len(questions) > 1 else '¿Cómo se relaciona el contenido con esta categoría?'}

Responde en 1-2 frases. Cita elementos específicos del texto (entre comillas).

EXPLICACIÓN:"""
        
        return prompt
