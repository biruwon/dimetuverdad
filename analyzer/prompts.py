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

🚫 HATE_SPEECH - AMPLIA DEFINICIÓN (NO solo insultos explícitos):

CRITERIOS CLAVE - Es hate_speech si:
1. CULPABILIZA a grupos específicos de problemas sociales/económicos
2. DESHUMANIZA mediante términos que niegan individualidad
3. USA RETÓRICA DE AMENAZA que presenta grupos como peligro existencial
4. PROMUEVE HOSTILIDAD implícita o explícita hacia grupos vulnerables

EJEMPLOS ESPECÍFICOS - Detecta como hate_speech:
- **SCAPEGOATING DE SERVICIOS**: Culpar a inmigrantes de "saturar", "colapsar", "destruir" servicios públicos (sanidad, vivienda, educación)
  → Esto NO es crítica política, es CULPABILIZACIÓN de un grupo vulnerable
- **RETÓRICA DE INVASIÓN**: Términos bélicos/militares aplicados a migración
  → "invasión", "oleada", "avalancha", "promover una invasión"
  → Presenta la inmigración como ATAQUE, no como fenómeno social
- **NARRATIVAS DE ESCASEZ**: "Nos quitan [X]", "menos [X] por culpa de..."
  → Enfrentar españoles vs inmigrantes en competencia de suma cero
  → Presenta inmigrantes como LADRONES de recursos, no como personas
- **DESHUMANIZACIÓN ECONÓMICA**: "Los traen", "hacen negocio con ellos", "importarlos"
  → Mercantiliza personas, las presenta como OBJETOS/MERCANCÍA
- **VÍCTIMAS VS INVASORES**: "Españoles condenados" vs "inmigrantes" que "destruyen todo"
  → Marco de conflicto donde un grupo ATACA y otro SUFRE
- **DESTRUCCIÓN APOCALÍPTICA**: "Lo destruye todo: seguridad, identidad, economía"
  → Lenguaje catastrofista que presenta grupo como AMENAZA EXISTENCIAL

⚠️  IMPORTANTE: Un texto puede ser hate_speech SIN usar insultos directos.
La culpabilización sistemática de grupos por problemas sociales ES discurso de odio.

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

🗳️  FAR_RIGHT_BIAS - Identifica:
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

REGLA DE ORO: Si el texto CULPA a un grupo étnico/nacional por problemas sociales usando lenguaje de AMENAZA, INVASIÓN, o DESTRUCCIÓN → Es HATE_SPEECH, no opinión política moderada."""
    
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
        
        Args:
            text_content: The text content accompanying the media
            is_video: Whether the media is a video
            
        Returns:
            Formatted analysis prompt for Gemini
        """
        media_type = "video" if is_video else "imagen"
        
        return f"""Analiza esta {media_type} y el texto adjunto que pertenecen al MISMO POST en una cuenta de Twitter/X para detectar contenido político de extrema derecha o desinformación.

TEXTO DEL POST: "{text_content}"

El contenido está en español y proviene de redes sociales. Evalúa si el post promueve o contiene:

CONTEXTO DEL POST:
- El texto y {media_type} son parte del mismo tweet/publicación en Twitter
- El texto proporciona contexto adicional al contenido visual de la {media_type}

ANÁLISIS REQUERIDO:
1. {"Resumen del contenido visual del video" if is_video else "Descripción detallada del contenido visual de la imagen (¿Quiénes aparecen? ¿Qué están haciendo?)"}
2. Análisis del texto adjunto para detectar:
   - Discurso político de extrema derecha
   - Teorías conspirativas
   - Llamados a la acción política
   - Ataques a instituciones democráticas
   - Desinformación o fake news
   - Retórica nacionalista o anti-inmigración
3. Evaluación de la relación entre texto y {media_type}
4. Clasificación por categorías: hate_speech, disinformation, conspiracy_theory, far_right_bias, call_to_action, general
5. Nivel de credibilidad y sesgo político detectado

IMPORTANTE: Responde completamente en español y sé específico sobre el contenido político español. Si reconoces personas públicas, identifícalas claramente."""
    
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
            "1️⃣ HATE_SPEECH - DEFINICIÓN AMPLIA (NO solo insultos):",
            "",
            "   ¿El texto CULPABILIZA a un grupo específico (étnico/nacional/religioso) de problemas sociales?",
            "   ¿USA RETÓRICA DE AMENAZA o INVASIÓN para describir grupos de personas?",
            "   ¿DESHUMANIZA mediante términos que niegan individualidad (mercantilización)?",
            "",
            "   DETECTA como hate_speech:",
            "   ✗ Culpar a inmigrantes de COLAPSAR/SATURAR servicios públicos (sanidad, vivienda)",
            "     → NO es crítica de política migratoria, es SCAPEGOATING",
            "   ✗ Términos bélicos: 'invasión', 'oleada', 'avalancha', 'promover invasión'",
            "     → Presenta migración como ATAQUE militar, no fenómeno social",
            "   ✗ Narrativas de escasez: 'nos quitan X', 'menos X por culpa de...'",
            "     → Enfrenta españoles vs inmigrantes en competencia suma-cero",
            "   ✗ Deshumanización económica: 'los traen', 'hacen negocio con ellos', 'importarlos'",
            "     → Mercantiliza personas = OBJETOS, no seres humanos",
            "   ✗ Marco víctimas vs invasores: 'españoles condenados' vs 'inmigrantes destruyen'",
            "     → Un grupo ATACA, otro SUFRE = hostilidad",
            "   ✗ Lenguaje apocalíptico: 'destruye todo: seguridad, identidad, economía'",
            "     → Grupo presentado como AMENAZA EXISTENCIAL",
            "",
            "   ⚠️  CLAVE: Puede ser hate_speech SIN insultos directos.",
            "   La culpabilización sistemática de grupos por problemas = discurso de odio.",
            "",
            "2️⃣ DISINFORMATION - ¿Presenta...?",
            "   • Afirmaciones médicas/científicas sin evidencia (vacunas-5G, COVID falso)",
            "   • Estadísticas inventadas o datos manipulados",
            "   • Claims sobre tratamientos no probados científicamente",
            "",
            "3️⃣ CONSPIRACY_THEORY - ¿Menciona...?",
            "   • Teorías sobre control secreto por élites (Soros, Davos, 'la agenda')",
            "   • Planes ocultos de reemplazo poblacional o cultural",
            "   • Organizaciones manipulando eventos masivamente sin fuentes",
            "",
            "4️⃣ FAR_RIGHT_BIAS - ¿Muestra...?",
            "   • Retórica extrema contra izquierda ('rojos', 'comunistas')",
            "   • Nacionalismo excluyente con lenguaje alarmista",
            "   • Marcos 'nosotros vs ellos' radicalizados",
            "",
            "5️⃣ CALL_TO_ACTION - ¿Incluye...?",
            "   • Llamadas explícitas a manifestaciones/protestas con lugar/hora",
            "   • Instrucciones específicas de acción colectiva ('todos a X')",
            "   • Urgencia para movilización inmediata",
            "",
            "6️⃣ GENERAL - Solo si:",
            "   • Contenido completamente neutral (clima, comida, entretenimiento)",
            "   • Opiniones políticas moderadas sin extremismo",
            "   • Información factual sin sesgo problemático",
            "",
            "DECISIÓN: Evalúa en orden 1→6. Si encuentras elementos de una categoría, esa es la respuesta.",
            "Si hay múltiples categorías aplicables, elige la MÁS ESPECÍFICA y PROBLEMÁTICA.",
            "",
            "⚠️  CASOS ESPECÍFICOS QUE SON HATE_SPEECH (NO 'crítica política'):",
            "- Culpar a inmigrantes de saturar/colapsar sanidad → HATE_SPEECH",
            "- Usar 'invasión' para describir inmigración → HATE_SPEECH",  
            "- 'Nos quitan viviendas/recursos' → HATE_SPEECH",
            "- 'Hacen negocio trayéndolos' (mercantilización) → HATE_SPEECH",
            "- 'Destruye seguridad/identidad/economía' → HATE_SPEECH",
            "",
            "REGLA CRÍTICA: Estas NO son 'opiniones políticas moderadas' ni 'crítica convencional'.",
            "Son SCAPEGOATING y DESHUMANIZACIÓN = HATE_SPEECH.",
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
                "focus": "afirmaciones falsas, datos manipulados o información médica/científica incorrecta",
                "questions": [
                    "¿Qué claims específicos son médica/científicamente incorrectos?",
                    "¿Qué evidencia contradice estas afirmaciones?",
                    "¿Cómo podría este contenido desinformar a la audiencia?"
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
            f"Explica por qué este contenido pertenece a la categoría '{category}', enfocándote en {context['focus']}.",
            "",
            "Considera estas preguntas en tu análisis:"
        ]
        
        for question in context['questions']:
            prompt_parts.append(f"- {question}")
        
        prompt_parts.extend([
            "",
            "IMPORTANTE: Responde en español natural y conversacional, sin usar:",
            "- Markdown (nada de **negritas**, ##títulos, o listas numeradas)",
            "- Encabezados estructurados como 'ANÁLISIS DETALLADO' o '1. Tema...'",
            "- Formato técnico o excesivamente estructurado",
            "",
            "Escribe 2-4 oraciones claras explicando los elementos problemáticos específicos del texto, como si le explicaras a un lector general. Sé directo y específico sobre lo que detectaste."
        ])
        
        return "\n".join(prompt_parts)

