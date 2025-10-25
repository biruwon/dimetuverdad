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
from .categories import Categories, CATEGORY_INFO

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

ANTI_IMMIGRATION - RETÓRICA ANTI-INMIGRACIÓN Y XENOFOBIA:
- Asociación general entre inmigración y delincuencia sin nombrar grupos específicos
- Frases alarmistas sobre seguridad ciudadana y presencia extranjera
- Retórica anti-inmigración que usa términos como "invasión" o "reemplazo"
- Culpa institucional hacia partidos políticos por políticas migratorias
- Nacionalismo excluyente con lenguaje de amenaza existencial
- Deshumanización de colectivos migratorios como "masas" o "olas incontrolables"
- Crítica a partidos políticos por permitir entrada de personas extranjeras

ANTI_LGBTQ - ATAQUES A LA COMUNIDAD LGBTQ Y GÉNERO:
- Ataques a la "ideología de género" o "agenda LGBT"
- Retórica sobre "adoctrinamiento infantil" o "van por los niños"
- Defensa de la "familia tradicional" contra amenazas percibidas
- Anti-trans rhetoric sobre biología o deportes
- Lenguaje que presenta identidad de género como amenaza cultural

ANTI_FEMINISM - RETÓRICA ANTI-FEMINISTA Y ROLES DE GÉNERO TRADICIONALES:
- Ataques a "feminazis" o feminismo radical
- Promoción de roles tradicionales de género
- Acusaciones de "machismo inverso" o "matriarcado opresivo"
- Lenguaje sobre mujeres en casa vs hombres proveedores

DISINFORMATION - INFORMACIÓN FALSA O MANIPULADA:
- Datos estadísticos sobre inmigración presentados sin fuentes verificables
- Afirmaciones científicas sobre origen étnico y comportamiento criminal
- Manipulación de hechos para generar pánico sobre inmigración

CONSPIRACY_THEORY - TEORÍAS SIN EVIDENCIA SOBRE CONTROL:
- Narrativas sobre élites globales controlando movimientos migratorios
- Planes ocultos de transformación demográfica o cultural
- Organizaciones secretas coordinando procesos migratorios masivos

**call_to_action** (MOVILIZACIÓN - ÚLTIMA PRIORIDAD PARA CLAIMS POLÍTICOS):
- Llamadas directas a protestar, manifestarse, o actuar colectivamente
- NO usar para claims políticos presentados como "confirmados" sin fuente
- Si un texto dice "CONFIRMADO: El Gobierno prohíbe..." → ES disinformation, NO call_to_action
- Si un texto dice "Ya está firmado el decreto..." → ES disinformation, NO call_to_action
- Call_to_action requiere LLAMADA DIRECTA A ACCIÓN, no solo información sobre decisiones políticas

GENERAL - CONTENIDO REALMENTE NEUTRAL:
- Temas apolíticos: clima, deportes, entretenimiento, cultura general
- Crítica política moderada sin alarmismo ni deshumanización
- Información factual neutral sin sesgos discriminatorios

DISTINCIÓN CRÍTICA:
- hate_speech: Ataques directos a grupos étnicos CONCRETOS identificables
- anti_immigration: Retórica anti-inmigración GENERAL sin nombrar grupos específicos

JERARQUÍA DE CLASIFICACIÓN (de mayor a menor prioridad):
1. hate_speech (ataques directos a grupos específicos)
2. anti_immigration (retórica anti-inmigración y xenofobia)
3. anti_lgbtq (ataques a comunidad LGBTQ)
4. anti_feminism (retórica anti-feminista)
5. disinformation (información falsa verificable, ESPECIALMENTE claims políticos sin fuente oficial)
6. conspiracy_theory (teorías conspirativas)
7. call_to_action (llamadas a movilización)
8. political_general (análisis político neutral)
9. general (contenido neutral)

IMPORTANTE: Sistema diseñado para investigación académica y moderación de contenido en redes sociales.
IMPORTANTE: Priorizar CONTEXTO POLÍTICO sobre verificación factual para clasificación.
IMPORTANTE: Alegaciones políticas polarizantes pueden clasificarse como anti_government o nationalism según el contexto.
IMPORTANTE: Acusaciones contra políticos por corrupción o conducta personal pueden clasificarse como anti_government.
IMPORTANTE: Ataques políticos presentados como "defensa de ciudadanos" pueden clasificarse como nationalism o anti_government.
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
2. anti_immigration
3. anti_lgbtq
4. anti_feminism
5. nationalism
6. anti_government
7. political_general
8. disinformation
9. conspiracy_theory
10. call_to_action
11. general

CATEGORÍAS:

REGLAS DE CLASIFICACIÓN - LÉELAS CON CUIDADO:

**hate_speech** (PRIORIDAD MÁXIMA - SOLO SI CUMPLE TODOS LOS CRITERIOS):
- Debe contener una declaración DIRECTA que vincule a un grupo étnico, religioso o nacional específico con violencia, criminalidad o inferioridad
- Se requiere una PALABRA NEGATIVA EXPLÍCITA asociada al grupo (ejemplos: "terroristas", "criminales", "delincuentes", "peligrosos", "inferiores")
- El grupo debe estar nombrado específicamente (NO "inmigrantes" genérico)
- Ejemplo CORRECTO: "Los musulmanes son terroristas que nos atacan"
- Ejemplo INCORRECTO: "Los inmigrantes son delincuentes" (falta grupo específico + palabra negativa clara)

**anti_immigration** (RETÓRICA ANTI-INMIGRACIÓN Y XENOFOBIA):
- Asociación general entre inmigración y delincuencia sin nombrar grupos étnicos específicos
- Frases alarmistas sobre seguridad ciudadana y presencia extranjera
- Retórica anti-inmigración que usa términos como "invasión" o "reemplazo"
- Culpa institucional hacia partidos políticos por políticas migratorias
- Nacionalismo excluyente con lenguaje de amenaza existencial
- Deshumanización de colectivos migratorios como "masas" o "olas incontrolables"
- Crítica a partidos políticos por permitir entrada de personas extranjeras

**anti_lgbtq** (ATAQUES A LA COMUNIDAD LGBTQ Y GÉNERO):
- Ataques a la "ideología de género" o "agenda LGBT"
- Retórica sobre "adoctrinamiento infantil" o "van por los niños"
- Defensa de la "familia tradicional" contra amenazas percibidas
- Anti-trans rhetoric sobre biología o deportes
- Lenguaje que presenta identidad de género como amenaza cultural

**anti_feminism** (RETÓRICA ANTI-FEMINISTA Y ROLES DE GÉNERO TRADICIONALES):
- Ataques a "feminazis" o feminismo radical
- Promoción de roles tradicionales de género
- Acusaciones de "machismo inverso" o "matriarcado opresivo"
- Lenguaje sobre mujeres en casa vs hombres proveedores

**nationalism**: Orgullo nacional sin anti-inmigración ni lenguaje divisivo
**anti_government**: Crítica institucional sin extremismo ni vínculos internacionales
**political_general**: Análisis político neutral sin extremismo
**disinformation**: Información falsa verificable sobre ciencia/medicina O información política falsa sobre eventos, decretos, nombramientos, destituciones, renuncias, alianzas políticas, o hechos verificables sin fuente creíble.

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
- "Moncloa anuncia cese de ministra por motivos personales" (SÍ fuente oficial)

FORMATO OBLIGATORIO:
CATEGORÍA: [nombre_categoría]
EXPLICACIÓN: [2‑3 frases explicando por qué pertenece a esa categoría, citando elementos específicos del texto]"""
        
        # Add the content at the end
        if content:
            prompt = prompt.replace("FORMATO OBLIGATORIO:", f"CONTENIDO A ANALIZAR:\n{content}\n\nFORMATO OBLIGATORIO:")
        
        return prompt

    def generate_explanation_prompt(self, text: str, category: str, model_type: str = "ollama") -> str:
        """
        Generate detailed explanation prompt with category-specific focus.
        For explain_only mode - explains WHY content belongs to the given category.
        """
        # Category-specific explanation prompts
        category_explanations = {
            "hate_speech": [
                "Este contenido contiene lenguaje de odio porque:",
                "1. ¿Qué grupos específicos son atacados o estereotipados negativamente?",
                "2. ¿Qué palabras o frases expresan desprecio, inferioridad o amenaza?",
                "3. ¿Cómo se vincula al grupo con violencia, criminalidad o características negativas?"
            ],
            "anti_immigration": [
                "Este contenido muestra retórica anti-inmigración porque:",
                "1. ¿Qué asociaciones se hacen entre inmigración y delincuencia o amenaza social?",
                "2. ¿Cómo se presenta la inmigración como amenaza existencial?",
                "3. ¿Qué lenguaje alarmista se usa sobre presencia extranjera?"
            ],
            "anti_lgbtq": [
                "Este contenido ataca a la comunidad LGBTQ porque:",
                "1. ¿Qué críticas se hacen a la identidad o derechos LGBTQ?",
                "2. ¿Cómo se presenta la diversidad de género como amenaza?",
                "3. ¿Qué lenguaje se usa sobre 'ideología de género' o 'adoctrinamiento infantil'?"
            ],
            "anti_feminism": [
                "Este contenido muestra retórica anti-feminista porque:",
                "1. ¿Qué críticas se hacen al movimiento feminista o igualdad de género?",
                "2. ¿Cómo se promueven roles tradicionales de género?",
                "3. ¿Qué lenguaje se usa sobre 'feminazis' o 'matriarcado opresivo'?"
            ],
            "nationalism": [
                "Este contenido muestra nacionalismo porque:",
                "1. ¿Qué expresiones de orgullo nacional se hacen?",
                "2. ¿Cómo se enfatiza la identidad nacional como valor primordial?",
                "3. ¿Qué símbolos patrios o narrativas de identidad nacional se usan?"
            ],
            "anti_government": [
                "Este contenido muestra anti-gubernamentalismo porque:",
                "1. ¿Qué críticas se hacen a la legitimidad del gobierno?",
                "2. ¿Cómo se cuestiona la autoridad institucional?",
                "3. ¿Qué retórica de deslegitimación política se usa?"
            ],
            "historical_revisionism": [
                "Este contenido muestra revisionismo histórico porque:",
                "1. ¿Qué eventos históricos se reinterpretan de forma sesgada?",
                "2. ¿Cómo se rehabilitan figuras o regímenes autoritarios?",
                "3. ¿Qué narrativas nostálgicas del pasado autoritario se promueven?"
            ],
            "political_general": [
                "Este contenido es político general porque:",
                "1. ¿Qué temas políticos convencionales se tratan?",
                "2. ¿Qué perspectivas políticas moderadas se presentan?",
                "3. ¿Cómo se debate de forma constructiva sin extremismo?"
            ],
            "disinformation": [
                "Este contenido es desinformación porque:",
                "1. ¿Qué afirmación específica se hace sobre hechos verificables?",
                "2. ¿Por qué carece de fuente oficial o creíble?",
                "3. ¿Cómo se presenta como cierto sin evidencia verificable?"
            ],
            "conspiracy_theory": [
                "Este contenido promueve una teoría conspirativa porque:",
                "1. ¿Qué narrativa oculta o agenda secreta se sugiere?",
                "2. ¿Qué grupos o instituciones son acusados de conspirar?",
                "3. ¿Cómo se presenta evidencia circunstancial como prueba definitiva?"
            ],
            "call_to_action": [
                "Este contenido incita a la acción porque:",
                "1. ¿Qué acción específica se pide realizar?",
                "2. ¿Cómo se usa lenguaje urgente o temporal para presionar?",
                "3. ¿Qué movilización colectiva se promueve?"
            ],
            "general": [
                "Este contenido neutral es porque:",
                "1. ¿Por qué no contiene elementos extremistas o problemáticos?",
                "2. ¿Qué lo hace informativo o moderadamente político?",
                "3. ¿Falta contexto discriminatorio, conspirativo o alarmista?"
            ]
        }
        
        # Get category-specific questions, fallback to general
        questions = category_explanations.get(category.lower(), category_explanations["general"])
        
        prompt_parts = [
            f'TEXTO ANALIZADO: "{text}"',
            f'CATEGORÍA DETECTADA: {category}',
            "",
            "🔬 ANÁLISIS ACADÉMICO DETALLADO - INVESTIGACIÓN SOBRE PATRONES DE DISCURSO",
            "",
            "OBJETIVO: Explicar por qué este contenido pertenece a la categoría detectada.",
            "",
            f"{questions[0]}",
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