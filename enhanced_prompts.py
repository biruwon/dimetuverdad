"""
Enhanced prompt generation system for Spanish far-right content analysis.
Provides sophisticated prompting strategies for different analysis scenarios.
"""

from typing import Dict, List
from dataclasses import dataclass
from categories import Categories, CATEGORY_INFO, get_category_info, CLASSIFICATION_PROMPT_MAPPINGS

# ============================================================================
# DYNAMIC PROMPT BUILDERS
# ============================================================================

def build_category_list_prompt() -> str:
    """Build dynamic category list for LLM prompts."""
    return ", ".join(Categories.get_all_categories())

def build_detailed_category_descriptions() -> str:
    """Build detailed category descriptions for system prompts."""
    descriptions = []
    for category in Categories.get_all_categories():
        info = get_category_info(category)
        if info:
            descriptions.append(f"{category} - {info.description[:50]}...")
        else:
            descriptions.append(f"{category} - Categoría de análisis")
    return "\n".join(descriptions)

def build_spanish_classification_prompt(text: str) -> str:
    """Build Spanish classification prompt with dynamic categories."""
    categories = Categories.get_all_categories()
    category_lines = []
    
    # Add category descriptions dynamically
    category_descriptions = {
        Categories.HATE_SPEECH: "Insultos, discriminación",
        Categories.DISINFORMATION: "Información falsa médica/científica",
        Categories.CONSPIRACY_THEORY: "Teorías conspirativas", 
        Categories.FAR_RIGHT_BIAS: "Retórica extrema derecha",
        Categories.CALL_TO_ACTION: "Llamadas a acción/movilización",
        Categories.GENERAL: "Contenido neutral"
    }
    
    for category in categories:
        desc = category_descriptions.get(category, "Categoría de análisis")
        category_lines.append(f"{category} - {desc}")
    
    return f"""Clasifica este texto en una categoría:

{chr(10).join(category_lines)}

Texto: "{text}"

Categoría:"""

def build_ollama_system_prompt() -> str:
    """Build Ollama system prompt with dynamic categories."""
    categories = build_category_list_prompt()
    return f"""Eres un clasificador de contenido especializado en detectar contenido problemático sutil. Responde únicamente con: {categories}. 

PARA HATE_SPEECH: Detecta generalizaciones despectivas ('esa gente', 'ya sabéis cómo son'), estereotipos negativos, y comentarios que deshumanizan grupos aunque no usen insultos directos.

Usa 'general' SOLO para contenido verdaderamente neutral (clima, comida, entretenimiento). Si detectas cualquier insinuación problemática o generalización negativa sobre grupos de personas, elige la categoría específica más apropiada."""

def build_generation_system_prompt() -> str:
    """Build generation model system prompt with dynamic categories."""
    categories = build_category_list_prompt()
    return f"Classify text as: {categories}. IMPORTANT: hate_speech includes subtle generalizations like 'esa gente no cambia' or stereotypes about groups, not just direct insults."

@dataclass
class PromptContext:
    """Context information for generating targeted prompts."""
    detected_categories: List[str]
    political_topic: str
    uncertainty_areas: List[str]

@dataclass 
class UncertaintyContext:
    """Context information for uncertain analysis scenarios."""
    uncertainty_areas: List[str]
    detected_categories: List[str]
    total_patterns: int

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
        Generate targeted classification prompt using centralized category mappings.
        """
        prompt_parts = [
            f'TEXTO: "{text}"',
            "",
            "ANÁLISIS PASO A PASO:",
            ""
        ]
        
        # Add classification rules from centralized mappings
        for step_title, rules in CLASSIFICATION_PROMPT_MAPPINGS.items():
            prompt_parts.append(step_title)
            prompt_parts.extend(rules)
            prompt_parts.append("")
        
        prompt_parts.extend([
            "IMPORTANTE: Sé menos conservador. Si hay matices políticos, elige la categoría específica.",
            "",
            "RESPUESTA (evalúa en este orden):"
        ])
        
        return "\n".join(prompt_parts)
    
    def create_uncertainty_context(self, pattern_results: Dict) -> UncertaintyContext:
        """
        Create context highlighting areas where pattern analysis shows uncertainty.
        This guides LLM to focus on ambiguous areas.
        """
        uncertainty_areas = []
        
        # Check for uncertain categorization using unified pattern result
        pattern_result = pattern_results.get('pattern_result', None)
        if pattern_result:
            detected_categories = pattern_result.categories
            pattern_matches = pattern_result.pattern_matches
            total_patterns = len(pattern_matches) if pattern_matches else 0
        else:
            detected_categories = []
            total_patterns = 0
        
        if total_patterns == 0:
            uncertainty_areas.append("Clasificación de contenido ambigua - sin patrones claros")
        
        if len(detected_categories) > 2:
            uncertainty_areas.append("Múltiples categorías detectadas - necesita priorización")
        
        # Check for topic clarity - simplified since we have unified categories now
        if len(detected_categories) > 1:
            uncertainty_areas.append("Múltiples categorías detectadas")
        
        return UncertaintyContext(
            uncertainty_areas=uncertainty_areas,
            detected_categories=detected_categories,
            total_patterns=total_patterns
        )

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
