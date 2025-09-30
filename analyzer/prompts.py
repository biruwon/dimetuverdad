"""
Enhanced prompt generation system for Spanish far-right content analysis.
Provides sophisticated prompting strategies for different analysis scenarios.
"""

from typing import Dict, List
from dataclasses import dataclass
from .categories import Categories, CATEGORY_INFO, get_category_info, CLASSIFICATION_PROMPT_MAPPINGS

# ============================================================================
# DYNAMIC PROMPT BUILDERS
# ============================================================================

def build_category_list_prompt() -> str:
    """Build dynamic category list for LLM prompts."""
    return ", ".join(Categories.get_all_categories())

def build_spanish_classification_prompt(text: str) -> str:
    """Build enhanced Spanish classification prompt with improved category descriptions."""
    categories = Categories.get_all_categories()
    
    # Enhanced category descriptions with specific examples
    category_descriptions = {
        Categories.HATE_SPEECH: "Insultos, generalizaciones despectivas, deshumanización de grupos ('moros', 'menas', 'esa gente')",
        Categories.DISINFORMATION: "Información médica/científica falsa (vacunas-5G, COVID inventado, curas milagrosas)",
        Categories.CONSPIRACY_THEORY: "Teorías conspirativas (Soros, élites secretas, planes ocultos, 'la agenda')", 
        Categories.FAR_RIGHT_BIAS: "Retórica extrema derecha ('rojos', anti-woke, nacionalismo excluyente)",
        Categories.CALL_TO_ACTION: "Llamadas a movilización con lugar/tiempo específicos ('Todos a Cibeles')",
        Categories.GENERAL: "Contenido neutro sin elementos problemáticos"
    }
    
    category_lines = []
    for category in categories:
        desc = category_descriptions.get(category, "Categoría de análisis")
        category_lines.append(f"• {category}: {desc}")
    
    return f"""ANÁLISIS DE CONTENIDO PROBLEMÁTICO

TEXTO A CLASIFICAR: "{text}"

CATEGORÍAS DISPONIBLES:
{chr(10).join(category_lines)}

INSTRUCCIONES:
1. Lee el texto completo cuidadosamente
2. Identifica elementos problemáticos sutiles o explícitos  
3. Si hay múltiples categorías aplicables, elige la MÁS ESPECÍFICA
4. Usa 'general' SOLO si el contenido es completamente neutral
5. Prefiere categorías específicas sobre 'general'

RESPUESTA (una sola palabra):"""

def build_ollama_system_prompt() -> str:
    """Build enhanced Ollama system prompt with dynamic categories and improved detection guidelines."""
    categories = build_category_list_prompt()
    return f"""Eres un clasificador experto en contenido problemático en español. Analiza el texto con precisión y responde ÚNICAMENTE con una de estas categorías: {categories}

GUÍAS DE DETECCIÓN MEJORADAS:

🚫 HATE_SPEECH - Identifica:
- Insultos directos o indirectos hacia grupos (étnicos, religiosos, nacionales)
- Generalizaciones despectivas ("esa gente", "ya sabéis cómo son", "todos los [grupo] son...")
- Deshumanización sutil (referencias a animales, objetos, amenazas)
- Lenguaje que incita hostilidad o violencia hacia grupos específicos
- Términos despectivos aunque no sean insultos obvios

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

⚡ FAR_RIGHT_BIAS - Identifica:
- Retórica extrema contra "rojos", "comunistas", izquierda
- Nacionalismo extremo excluyente
- Marcos interpretativos de "nosotros vs ellos" radicalizados
- Anti-inmigración con lenguaje alarmista ("invasión")

📢 CALL_TO_ACTION - Identifica:
- Llamadas explícitas a manifestaciones, protestas, movilización
- Instrucciones específicas de acción ("todos a [lugar]", "hay que salir")
- Urgencia para actuar colectivamente

✅ GENERAL - SOLO para contenido neutro:
- Conversación cotidiana, clima, comida, entretenimiento
- Opiniones políticas moderadas sin elementos extremistas
- Información factual sin sesgo problemático

IMPORTANTE: Si detectas CUALQUIER elemento problemático, elige la categoría específica más apropiada. Sé menos conservador - prefiere categorías específicas sobre 'general'."""

def build_generation_system_prompt() -> str:
    """Build enhanced generation model system prompt with improved detection guidelines."""
    categories = build_category_list_prompt()
    return f"""You are an expert content classifier specializing in detecting problematic Spanish content. Classify text as one of: {categories}

ENHANCED DETECTION RULES:

HATE_SPEECH: Detect subtle dehumanization and generalizations
- Direct/indirect insults toward ethnic, religious, or national groups
- Derogatory generalizations ('esa gente', 'ya sabéis cómo son', 'todos los X son...')
- Subtle dehumanization (animal references, object comparisons)
- Language inciting hostility toward specific groups

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

@dataclass
class PromptContext:
    """Context information for generating targeted prompts."""
    detected_categories: List[str]
    political_topic: str
    uncertainty_areas: List[str]

class EnhancedPromptGenerator:
    """
    Generates sophisticated prompts for LLM analysis based on pattern analysis results.
    Focuses on areas where pattern matching shows uncertainty.
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
    
    def generate_prompt(self, 
                       text: str, 
                       category: str = Categories.GENERAL,
                       context: PromptContext = None,
                       complexity_level: str = "medium",
                       model_type: str = "transformers") -> str:
        """
        Generate a sophisticated prompt based on content category and context.
        
        Args:
            text: Content to analyze
            category: Content category from Categories class
            context: Analysis context with detected patterns
            complexity_level: Prompt complexity (simple/medium/complex)
            model_type: Target model type (transformers/ollama)
        """
        # Use general template if category not found
        template = self.prompt_templates.get(category, self.prompt_templates[Categories.GENERAL])
        
        # Build context-aware prompt
        prompt_parts = [
            template["system"],
            "",
            f"CONTENIDO A ANALIZAR:",
            f'"{text}"',
            ""
        ]
        
        # Add context information to guide analysis
        if context and context.detected_categories:
            prompt_parts.extend([
                f"PATRONES DETECTADOS: {', '.join(context.detected_categories)}",
                ""
            ])
        
        if context and context.political_topic and context.political_topic != "no_político":
            prompt_parts.extend([
                f"CONTEXTO POLÍTICO: {context.political_topic}",
                ""
            ])
        
        # Add uncertainty-focused questions for areas needing LLM insight
        if context and context.uncertainty_areas:
            prompt_parts.extend([
                "ÁREAS DE INCERTIDUMBRE A CLARIFICAR:",
                *[f"- {area}" for area in context.uncertainty_areas],
                ""
            ])
        
        # Add analysis instructions based on complexity
        if complexity_level == "simple":
            prompt_parts.extend([
                f"Proporciona un {template['focus']} breve y directo.",
                "Responde en 2-3 frases concisas."
            ])
        elif complexity_level == "complex":
            prompt_parts.extend([
                f"Realiza un {template['focus']} detallado respondiendo:",
                *[f"- {q}" for q in template.get("questions", [])],
                "",
                "Proporciona un análisis extenso con ejemplos específicos del texto."
            ])
        else:  # medium
            prompt_parts.extend([
                f"Analiza el contenido enfocándote en {template['focus']}.",
                "Responde de forma clara y estructurada en 4-6 frases."
            ])
        
        # Add model-specific formatting
        if model_type == "ollama":
            prompt_parts.append("\nRespuesta:")
        
        return "\n".join(prompt_parts)
    
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
            "",
            "6️⃣ GENERAL - Solo si:",
            "   • Contenido completamente neutral (clima, comida, entretenimiento)",
            "   • Opiniones políticas moderadas sin extremismo",
            "   • Información factual sin sesgo problemático",
            "",
            "DECISIÓN: Evalúa en orden 1→6. Si encuentras elementos de una categoría, esa es la respuesta.",
            "Si hay múltiples categorías aplicables, elige la MÁS ESPECÍFICA y PROBLEMÁTICA.",
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
            f"ANÁLISIS DETALLADO - Enfócate en {context['focus']}:",
            ""
        ]
        
        for i, question in enumerate(context['questions'], 1):
            prompt_parts.append(f"{i}. {question}")
        
        prompt_parts.extend([
            "",
            "EXPLICACIÓN (2-3 oraciones claras y específicas sobre los elementos detectados):"
        ])
        
        return "\n".join(prompt_parts)

def create_context_from_analysis(analysis_results: Dict) -> PromptContext:
    """
    Create prompt context from analysis results.
    Used by LLM pipeline for backwards compatibility.
    """
    # Handle different context formats
    detected_categories = []
    
    # Check for detected_categories from enhanced analyzer
    if 'detected_categories' in analysis_results:
        detected_categories = analysis_results['detected_categories']
    # Check for categories from pattern analysis
    elif 'categories' in analysis_results:
        detected_categories = analysis_results['categories']
    # Fall back to single category
    elif 'detected_category' in analysis_results:
        detected_categories = [analysis_results['detected_category']]
    
    # Get political topic
    political_topic = analysis_results.get('category', analysis_results.get('detected_category', 'general'))
    
    # Get uncertainty areas
    uncertainty_areas = analysis_results.get('uncertainty_areas', [])
    
    return PromptContext(
        detected_categories=detected_categories,
        political_topic=political_topic,
        uncertainty_areas=uncertainty_areas
    )

# Test function
if __name__ == "__main__":
    generator = EnhancedPromptGenerator()
    
    # Test context using centralized categories
    test_context = PromptContext(
        detected_categories=[Categories.HATE_SPEECH, Categories.FAR_RIGHT_BIAS],
        political_topic='inmigración',
        uncertainty_areas=['Nivel de amenaza real', 'Intención del autor']
    )
    
    test_text = "Los musulmanes están invadiendo España y hay que pararlos ya"
    
    # Test all content categories using centralized definitions
    test_categories = Categories.get_all_categories()
    
    for category in test_categories:
        print(f"\n=== {category.upper()} ===")
        prompt = generator.generate_prompt(test_text, category, test_context)
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
