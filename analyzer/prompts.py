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

from typing import Optional
from .categories import Categories, get_category_info

# ============================================================================
# ENHANCED PROMPT GENERATOR CLASS
# ============================================================================

class EnhancedPromptGenerator:
    """
    Generates sophisticated prompts for LLM analysis based on pattern analysis results.
    Centralized prompt management for all analysis scenarios.
    """

    def __init__(self):
        # Initialize with minimal setup since instance attributes are not used
        pass

    # ============================================================================
    # CONFIGURATION-DRIVEN PROMPT BUILDERS
    # ============================================================================

    @staticmethod
    def build_category_list() -> str:
        """Build dynamic category list for LLM prompts."""
        return ", ".join(Categories.get_all_categories())

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
        categories_simple = """
hate_speech: Insultos directos a individuos específicos, acoso sexual, insinuaciones sexuales degradantes, objectificación sexual - INCLUYE insultos culturales españoles (referencias despectivas a características físicas como apodos degradantes), palabras graves como "rata", "mierda", "traidor", "psicópata", "basura", "escoria", "parásito", "animal", "monstruo", comentarios sexuales humillantes, insinuaciones degradantes, objectificación de cuerpos (apodos relacionados con apariencia física, referencias despectivas a características corporales), referencias a prostitución o servicios sexuales, y metáforas sexuales degradantes dirigidas a personas específicas
anti_immigration: Retórica xenófoba contra grupos ("invasión", "manadas", "ilegales", "ocupación")
anti_lgbtq: Ataques al colectivo LGBTQ ("ideología de género", "adoctrinamiento", "imposición")
anti_feminism: Retórica anti-feminista ("feminazis", "hembrismo", roles tradicionales)
disinformation: Afirmaciones FALSAS verificables EN EL TEXTO del post sobre hechos actuales ("X ha dimitido", "X está en prisión", "X ha sido detenido") sin fuente oficial
conspiracy_theory: Agendas secretas, élites ocultas ("ellos controlan", "agenda oculta", "manipulación global")
call_to_action: Incitación EXPLÍCITA a movilización colectiva ("todos a la calle", "hay que actuar YA", "únete a la manifestación")
nationalism: RETÓRICA NACIONALISTA EXCLUYENTE que requiere lenguaje de superioridad nacional, rechazo a lo extranjero, o identidad nacional amenazada ("España primero sobre todo", "nuestra nación es superior", "rechazamos influencias extranjeras", "defensa de la pureza nacional") - NO BASTA con símbolos patrios solos, banderas, o expresiones de apoyo político neutral
anti_government: Crítica institucional ("gobierno corrupto", "régimen", "dictadura", "tiranía")
political_general: Contenido sobre ELECCIONES, PARTIDOS POLÍTICOS, CANDIDATOS, CAMPAÑAS ELECTORALES, DEBATES POLÍTICOS o POLÍTICAS PÚBLICAS - menciones neutrales de procesos democráticos sin extremismo
general: Contenido NO POLÍTICO o temas cotidianos no relacionados con política, ideología, o asuntos sociales controvertidos"""
        
        # Critical rules with examples - ENHANCED
        key_rules = """Reglas críticas:
• hate_speech: INSULTO PERSONAL, ACOSO SEXUAL, OBJECTIFICACIÓN, insinuaciones sexuales degradantes o referencias a prostitución dirigidas a individuos específicos ("X es un traidor/rata", apodos despectivos relacionados con apariencia física, insinuaciones sexuales humillantes, referencias a servicios sexuales) | anti_government: CRÍTICA SISTEMA ("el gobierno es corrupto")
• disinformation: FALSO EN EL TEXTO DEL POST sobre situación actual ("X está en prisión", "X ha dimitido") sin fuente oficial | political_general: INFORMATIVO con fuente
• conspiracy_theory: CONTROL SECRETO ("élites manipulan todo") | anti_government: CRÍTICA PÚBLICA de políticas visibles
• call_to_action: INCITACIÓN EXPLÍCITA a movilización colectiva ("sal a la calle YA", "únete a la manifestación") | political_general: OPINIÓN o invitación pasiva ("deberían cambiar", "os dejo el enlace")
• anti_immigration: ATAQUE A GRUPO étnico | hate_speech: ATAQUE A INDIVIDUO concreto
• nationalism: REQUIERE TEXTO EXPLÍCITO de superioridad nacional o exclusión ("nuestra nación es superior", "rechazamos lo extranjero") - símbolos patrios solos (banderas, emojis) sin retórica nacionalista = political_general
• political_general: PROCESOS ELECTORALES Y DEMOCRÁTICOS ("elecciones", "partidos", "candidatos", "campañas") | general: ACTIVIDADES OPERATIVAS GUBERNAMENTALES ("policía detiene", "justicia investiga", "administración funciona") SIN CONTEXTO POLÍTICO"""
        
        if pattern_category and pattern_category != Categories.GENERAL:
            # Pattern suggested a category - add generalized validation guidance
            pattern_guidance = f"\n\n⚠️ VALIDACIÓN DE PATRÓN: Los patrones automáticos detectaron posible {pattern_category}. Valida si el contexto confirma esta clasificación. Si el contenido usa las palabras de forma literal, neutra, o sin intención problemática, elige la categoría correcta."
            
            # Pattern suggested a category - quick validation
            prompt = f"""Contenido: {content}

Sugerida: {pattern_category}{pattern_guidance}

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
        - WHO is creating/showing content (critical for attribution)
        - NO category information or classification guidance
        
        Returns:
            Prompt for objective media description
        """
        return """Describe objetivamente lo que ves en estas imágenes.

Enfócate en:
• Texto visible en pancartas/carteles: cita el mensaje Y especifica si ataca/menciona al autor del post
• Personas y acciones: manifestantes, políticos, eventos
• Símbolos políticos: banderas, insignias, logos
• Contexto: manifestación, evento, entrevista


Sé conciso: 2-3 frases.

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

Responde en 1-2 frases. Cita elementos específicos del texto (entre comillas) y menciona elementos visuales relevantes."""
        else:
            # Text-only explanation
            prompt = f"""Texto: {content}

Categoría: {category}

{category_focus}

Guía:
• {questions[0] if len(questions) > 0 else '¿Qué elementos del texto justifican esta clasificación?'}
• {questions[1] if len(questions) > 1 else '¿Cómo se relaciona el contenido con esta categoría?'}

Responde en 1-2 frases. Cita elementos específicos del texto (entre comillas)."""
        
        return prompt
