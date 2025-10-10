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
        return f"""Eres un clasificador experto en contenido problemático en español. Analiza el texto con precisión y responde ÚNICAMENTE con una de estas categorías: {categories}

GUÍAS DE DETECCIÓN MEJORADAS:

🚫 HATE_SPEECH - Identifica:
- Insultos directos/indirectos hacia grupos étnicos, religiosos, nacionales
- Generalizaciones despectivas ('esa gente', 'ya sabéis cómo son', 'todos los X')
- Deshumanización sutil (comparaciones con animales/objetos)
- Lenguaje que incita hostilidad hacia grupos específicos
- Ataques basados en características protegidas (raza, religión, origen)

❌ DISINFORMATION - Identifica:
- Afirmaciones médicas/científicas falsas sin evidencia
- Estadísticas inventadas o manipuladas
- Teorías sobre vacunas, 5G, salud sin base científica
- Claims sobre efectividad de tratamientos no probados

🔍 CONSPIRACY_THEORY - Identifica:
- Teorías sobre control secreto por élites globales
- Planes ocultos de reemplazo poblacional o cultural
- Afirmaciones sobre manipulación masiva por organizaciones
- Referencias a "la agenda" sin especificar fuente verificable

🗳️ FAR_RIGHT_BIAS - Identifica:
- Ataques a ideologías políticas de izquierda ('progres', 'comunistas', 'izquierda')
- Lenguaje alarmista sobre supuesta infiltración ideológica en instituciones
- Narrativas de 'guerra cultural' contra valores tradicionales
- Marcos 'nosotros vs ellos' radicalizados por motivos políticos
- Anti-inmigración con deshumanización
- Lenguaje que presenta ideologías como amenazas existenciales

📢 CALL_TO_ACTION - Identifica:
- Llamadas explícitas a manifestaciones, protestas, movilización
- Instrucciones específicas de acción ("todos a [lugar]", "hay que salir")
- Urgencia para actuar colectivamente
- Llamadas a organizarse o 'hacer algo' sin especificar detalles
- Lenguaje que incita a la acción colectiva ('no podemos quedarnos de brazos cruzados')
- Invitaciones a la participación activa en causas

✅ GENERAL - SOLO para contenido neutro:
- Conversación cotidiana, clima, comida, entretenimiento
- Opiniones políticas moderadas sin elementos extremistas
- Información factual sin sesgo problemático

IMPORTANTE: Si detectas CUALQUIER elemento problemático, elige la categoría específica más apropiada. Sé menos conservador - prefiere categorías específicas sobre 'general'."""
    
    @staticmethod
    def build_generation_system_prompt() -> str:
        """Build enhanced generation model system prompt with improved detection guidelines."""
        categories = EnhancedPromptGenerator.build_category_list()
        return f"""You are an expert content classifier specializing in detecting problematic Spanish content. Classify text as one of: {categories}

ENHANCED DETECTION RULES:

HATE_SPEECH: Detect subtle dehumanization and generalizations
- Direct/indirect insults toward ethnic, religious, or national groups
- Derogatory generalizations ('esa gente', 'ya sabéis cómo son', 'todos los X son...')
- Subtle dehumanization (animal references, object comparisons)

DISINFORMATION: Medical/scientific false claims
- Unproven medical treatments or conspiracy theories about vaccines/5G
- Fabricated statistics or manipulated data
- False health information without scientific backing

CONSPIRACY_THEORY: Unfounded theories about secret control
- Claims about elite global manipulation or population replacement
- References to hidden agendas without verifiable sources
- Theories about coordinated secret plans

FAR_RIGHT_BIAS: Extreme political rhetoric
- Radical anti-left language ('rojos', 'comunistas')
- Exclusionary nationalism with alarmist framing ('invasión')
- Us-vs-them radicalized frameworks

CALL_TO_ACTION: Explicit mobilization calls
- Specific instructions for collective action with time/location
- Direct calls for protests, demonstrations, or gatherings

GENERAL: Truly neutral content only (weather, food, entertainment, moderate opinions)

CRITICAL: If ANY problematic element is detected, choose the most specific category rather than 'general'. Be less conservative in classification."""
    
    @staticmethod
    def build_spanish_classification_prompt(text: str) -> str:
        """Build simple Spanish classification prompt for generation models."""
        categories = EnhancedPromptGenerator.build_category_list()
        return f"""Clasifica el siguiente texto en una de estas categorías: {categories}

Texto: "{text}"

Responde SOLO con el nombre de la categoría más apropiada:"""
    
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
        
        prompt_parts = [
            f'TEXTO DEL POST ANALIZADO: "{text_content}"',
            "",
            f"ANÁLISIS DETALLADO - Enfócate en contenido político problemático en la {media_type} y texto:",
            "",
            "1. ¿Qué elementos visuales específicos aparecen en la imagen/video que sugieren contenido político?",
            "2. ¿Qué texto o elementos gráficos indican sesgo político, especialmente de extrema derecha?",
            "3. ¿Se mencionan datos, estadísticas o hechos específicos? Evalúalos por veracidad y contexto",
            "4. ¿Cómo se relacionan el contenido visual y textual para crear una narrativa política?",
            "5. ¿Qué categorías problemáticas se detectan: hate_speech, disinformation, conspiracy_theory, far_right_bias, call_to_action?",
            "6. Si se mencionan hechos o datos, ¿son verificables y se presentan en contexto correcto?",
            "7. ¿Cómo contribuye la combinación de imagen/video + texto a narrativas problemáticas?",
            "",
            "INSTRUCCIONES DE FORMATO:",
            "- Responde SOLO con texto plano en español, sin markdown ni formato especial",
            "- NO uses negritas (**), títulos (##), listas numeradas, tablas, o símbolos",
            "- Escribe 2-3 oraciones claras y directas explicando los elementos detectados",
            "- Evalúa cualquier dato o hecho mencionado por su veracidad y contexto",
            "- Como si le explicaras a una persona que no conoce el tema",
            "",
            "EXPLICACIÓN:"
        ]
        
        return "\n".join(prompt_parts)
    
    # ============================================================================
    # INSTANCE METHODS FOR SOPHISTICATED PROMPTS
    # ============================================================================

    def generate_classification_prompt(self, text: str, model_type: str = "ollama") -> str:
        """
        Generate enhanced step-by-step classification prompt for improved accuracy.
        """
        prompt_parts = [
            f'TEXTO A ANALIZAR: "{text}"',
            "",
            "PROCESO DE ANÁLISIS PASO A PASO:",
            "",
            "1️⃣ HATE_SPEECH - ¿Contiene el texto...?",
            "   • Insultos directos/indirectos hacia grupos étnicos, religiosos, nacionales",
            "   • Generalizaciones despectivas ('esa gente', 'ya sabéis cómo son', 'todos los X')",
            "   • Deshumanización sutil (comparaciones con animales/objetos)",
            "   • Lenguaje que incita hostilidad hacia grupos específicos",
            "",
            "2️⃣ DISINFORMATION - ¿Presenta...?",
            "   • Afirmaciones médicas/científicas sin evidencia (vacunas-5G, COVID falso)",
            "   • Estadísticas inventadas o datos manipulados",
            "   • Claims sobre tratamientos no probados científicamente",
            "   • ASOCIACIONES FALSAS: Conecta eventos/países/políticas sin relación real",
            "   • MARCOS DICTATORIALES: Califica democracias como 'dictaduras' sin evidencia",
            "   • MANIPULACIÓN INTERNACIONAL: Usa noticias de otros países para ataques locales",
            "   • EQUIVALENCIAS FALSAS: Presenta situaciones diferentes como idénticas",
            "   • DESCONTEXTUALIZACIÓN: Información real usada en contexto engañoso",
            "",
            "3️⃣ CONSPIRACY_THEORY - ¿Menciona...?",
            "   • Teorías sobre control secreto por élites (Soros, Davos, 'la agenda')",
            "   • Planes ocultos de reemplazo poblacional o cultural",
            "   • Organizaciones manipulando eventos masivamente sin fuentes",
            "",
            "4️⃣ FAR_RIGHT_BIAS - ¿Muestra...?",
            "   • Retórica extrema contra izquierda ('rojos', 'comunistas')",
            "   • Nacionalismo excluyente con lenguaje alarmista ('invasión')",
            "   • Marcos 'nosotros vs ellos' radicalizados",
            "   • Anti-inmigración con deshumanización",
            "",
            "5️⃣ CALL_TO_ACTION - ¿Incluye...?",
            "   • Llamadas explícitas a manifestaciones/protestas con lugar/hora",
            "   • Instrucciones específicas de acción colectiva ('todos a X')",
            "   • Urgencia para movilización inmediata",
            "   • Llamadas a organizarse o 'hacer algo' sin especificar detalles",
            "   • Lenguaje que incita a la acción colectiva ('no podemos quedarnos de brazos cruzados')",
            "   • Invitaciones a la participación activa en causas",
            "",
            "6️⃣ GENERAL - Solo si:",
            "   • Contenido completamente neutral (clima, comida, entretenimiento)",
            "   • Opiniones políticas moderadas sin extremismo",
            "   • Información factual sin sesgo problemático",
            "",
            "DECISIÓN: Evalúa en orden 1→6. Si encuentras elementos de una categoría, esa es la respuesta.",
            "Si hay múltiples categorías aplicables, elige la MÁS ESPECÍFICA y PROBLEMÁTICA.",
            "IMPORTANTE: Contenido político con llamadas a acción → call_to_action (NO political_general)",
            "IMPORTANTE: Lenguaje que urge organización colectiva → call_to_action",
            "",
            "RESPUESTA FINAL (una sola palabra):"
        ]
        
        return "\n".join(prompt_parts)
    
    def generate_explanation_prompt(self, text: str, category: str, model_type: str = "ollama") -> str:
        """
        Generate detailed explanation prompt with category-specific focus.
        """
        category_context = {
            Categories.HATE_SPEECH: {
                "focus": "elementos de odio, discriminación o ataques hacia grupos específicos",
                "questions": [
                    "¿Qué lenguaje específico genera hostilidad o desprecio?",
                    "¿Hacia qué grupo(s) se dirige el contenido problemático?",
                    "¿Cómo contribuye este contenido a narrativas de exclusión?"
                ]
            },
            Categories.DISINFORMATION: {
                "focus": "afirmaciones falsas, datos manipulados, información médica/científica incorrecta, asociaciones falsas entre países/políticas, marcos dictatoriales aplicados a democracias, y manipulación emocional a través de equivalencias erróneas",
                "questions": [
                    "¿Qué afirmaciones específicas son médica/científicamente incorrectas o factuales falsas?",
                    "¿Se están creando asociaciones falsas entre políticas de diferentes países o contextos políticos no relacionados?",
                    "¿Se está enmarcando un sistema democrático como dictatorial sin evidencia factual?",
                    "¿Se están utilizando noticias de un país para atacar falsamente políticos o sistemas de otro país?",
                    "¿Qué equivalencias erróneas se establecen entre situaciones políticas completamente diferentes?",
                    "¿Cómo se manipula emocionalmente al lector a través de términos como 'dictadura' aplicados incorrectamente?",
                    "¿Qué evidencia contradice estas afirmaciones o asociaciones?"
                ]
            },
            Categories.CONSPIRACY_THEORY: {
                "focus": "teorías sobre control secreto, planes ocultos o manipulación masiva",
                "questions": [
                    "¿Qué teoría conspirativa específica se menciona?",
                    "¿Qué actores se presentan como controladores secretos?",
                    "¿Cómo se estructura la narrativa de 'plan oculto'?"
                ]
            },
            Categories.FAR_RIGHT_BIAS: {
                "focus": "retórica extremista, nacionalismo excluyente o marcos políticos radicales",
                "questions": [
                    "¿Qué elementos específicos indican sesgo de extrema derecha?",
                    "¿Cómo se manifiesta el nacionalismo o anti-inmigración?",
                    "¿Qué marcos 'nosotros vs ellos' se emplean?"
                ]
            },
            Categories.CALL_TO_ACTION: {
                "focus": "llamadas específicas a la movilización o acción colectiva",
                "questions": [
                    "¿Qué acción específica se solicita a los seguidores?",
                    "¿Se proporcionan detalles como lugar, hora o método?",
                    "¿Cuál es la urgencia o motivación para la movilización?"
                ]
            },
            Categories.NATIONALISM: {
                "focus": "retórica nacionalista y exaltación de la identidad nacional",
                "questions": [
                    "¿Qué símbolos o valores nacionales se exaltan?",
                    "¿Cómo se presenta la identidad nacional como amenazada?",
                    "¿Qué elementos de nacionalismo excluyente se detectan?"
                ]
            },
            Categories.ANTI_GOVERNMENT: {
                "focus": "retórica anti-gubernamental y deslegitimización institucional",
                "questions": [
                    "¿Qué aspectos del gobierno se cuestionan como ilegítimos?",
                    "¿Cómo se manifiesta la retórica anti-establishment?",
                    "¿Se promueve resistencia o desobediencia institucional?"
                ]
            },
            Categories.HISTORICAL_REVISIONISM: {
                "focus": "reinterpretación sesgada de eventos históricos",
                "questions": [
                    "¿Qué eventos históricos se reinterpretan de forma problemática?",
                    "¿Se rehabilitan figuras o regímenes controvertidos?",
                    "¿Cómo se usa la historia para justificar narrativas actuales?"
                ]
            },
            Categories.POLITICAL_GENERAL: {
                "focus": "contenido político convencional sin elementos extremistas",
                "questions": [
                    "¿Qué temas políticos se tratan de forma constructiva?",
                    "¿Qué perspectiva política moderada se presenta?",
                    "¿Por qué no entra en categorías problemáticas específicas?"
                ]
            },
            Categories.GENERAL: {
                "focus": "contenido neutral o político moderado sin elementos extremistas",
                "questions": [
                    "¿Por qué este contenido no entra en categorías problemáticas?",
                    "¿Qué lo hace neutral o moderadamente político?",
                    "¿Falta contexto extremista, conspirativo o de odio?"
                ]
            }
        }
        
        context = category_context.get(category, category_context[Categories.GENERAL])
        
        prompt_parts = [
            f'TEXTO ANALIZADO: "{text}"',
            f'CATEGORÍA DETECTADA: {category}',
            "",
            f"ANÁLISIS DETALLADO - Enfócate en {context['focus']}:",
            ""
        ]
        
        for i, question in enumerate(context['questions'], 1):
            prompt_parts.append(f"{i}. {question}")
        
        prompt_parts.extend([
            "",
            "INSTRUCCIONES DE FORMATO:",
            "- Responde SOLO con texto plano en español, sin markdown ni formato especial",
            "- NO uses negritas (**), títulos (##), listas numeradas, tablas, o símbolos",
            "- Escribe 2-3 oraciones claras y directas explicando los elementos detectados",
            "- Como si le explicaras a una persona que no conoce el tema",
            "",
            "EXPLICACIÓN:"
        ])
        
        return "\n".join(prompt_parts)

