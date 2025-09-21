"""
Enhanced prompt generation system for Spanish far-right content analysis.
Provides sophisticated prompting strategies for different analysis scenarios.
"""

from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass

class AnalysisType(Enum):
    COMPREHENSIVE = "comprehensive"
    THREAT_ASSESSMENT = "threat_assessment"
    MISINFORMATION = "misinformation"
    CLAIM_VERIFICATION = "claim_verification"
    FAR_RIGHT_BIAS = "far_right_bias"
    HATE_SPEECH = "hate_speech"
    CLASSIFICATION = "classification"  # Added for classification tasks

@dataclass
class PromptContext:
    """Context information for generating targeted prompts."""
    detected_categories: List[str]
    targeted_groups: List[str]
    claims_count: int
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
    
    def _initialize_prompt_templates(self) -> Dict[AnalysisType, Dict]:
        """Initialize prompt templates for different analysis types."""
        return {
            AnalysisType.COMPREHENSIVE: {
                "system": f"{self.base_context} Analiza el siguiente contenido de forma integral.",
                "focus": "análisis completo del contenido",
                "questions": [
                    "¿Qué elementos problemáticos identificas?",
                    "¿Cuál es el tono y la intención del mensaje?",
                    "¿Hay indicios de sesgo político o radicalización?"
                ]
            },
            
            AnalysisType.THREAT_ASSESSMENT: {
                "system": f"{self.base_context} Evalúa específicamente el nivel de amenaza y riesgo del contenido.",
                "focus": "evaluación de amenaza y riesgo",
                "questions": [
                    "¿Hay llamadas explícitas o implícitas a la violencia?",
                    "¿Se incita al odio contra grupos específicos?",
                    "¿Cuál es el nivel de peligrosidad del mensaje?"
                ]
            },
            
            AnalysisType.MISINFORMATION: {
                "system": f"{self.base_context} Analiza la veracidad y detecta posible desinformación.",
                "focus": "detección de desinformación",
                "questions": [
                    "¿Hay afirmaciones que parecen falsas o sin evidencia?",
                    "¿Se presentan teorías conspiratorias?",
                    "¿Qué elementos requieren verificación factual?"
                ]
            },
            
            AnalysisType.CLAIM_VERIFICATION: {
                "system": f"{self.base_context} Identifica y evalúa las afirmaciones verificables del contenido.",
                "focus": "verificación de afirmaciones",
                "questions": [
                    "¿Qué afirmaciones específicas se pueden verificar?",
                    "¿Hay datos estadísticos o referencias concretas?",
                    "¿Qué nivel de urgencia tiene la verificación?"
                ]
            },
            
            AnalysisType.FAR_RIGHT_BIAS: {
                "system": f"{self.base_context} Evalúa el sesgo político y la retórica partidista.",
                "focus": "análisis de sesgo político",
                "questions": [
                    "¿Qué orientación política refleja el mensaje?",
                    "¿Hay elementos de propaganda o manipulación?",
                    "¿Cómo se presenta a los grupos políticos opuestos?"
                ]
            },
            
            AnalysisType.HATE_SPEECH: {
                "system": f"{self.base_context} Analiza específicamente discurso de odio y discriminación.",
                "focus": "detección de discurso de odio",
                "questions": [
                    "¿Hay lenguaje discriminatorio o deshumanizante?",
                    "¿Se atacan grupos por características protegidas?",
                    "¿Cuál es la severidad del discurso de odio?"
                ]
            },
            
            AnalysisType.CLASSIFICATION: {
                "system": f"{self.base_context} Analiza el contenido para comprender su estructura comunicativa y propósito.",
                "focus": "análisis de características comunicativas",
                "categories": {
                    "hate_speech": "Comunicación que establece jerarquías entre grupos sociales o expresa incompatibilidad cultural fundamental. Características: lenguaje que sugiere diferencias irreconciliables, expresiones de superioridad/inferioridad grupal, narrativas de amenaza identitaria.",
                    
                    "disinformation": "Comunicación que presenta afirmaciones fácticas sin evidencia documentada. Características: datos sin fuente verificable, interpretaciones especulativas presentadas como hechos, críticas sistemáticas a instituciones informativas.",
                    
                    "conspiracy_theory": "Comunicación que propone explicaciones causales no evidenciadas sobre eventos complejos. Características: patrones de control oculto, atribución de intencionalidad coordinada sin pruebas, narrativas de manipulación encubierta.",
                    
                    "far_right_bias": "Comunicación con marcos interpretativos de extrema derecha que presenta transformaciones sociales como amenazas a valores tradicionales. Características: lenguaje de pérdida cultural, narrativas de amenaza identitaria, marcos de preservación de tradiciones.",
                    
                    "call_to_action": "Comunicación orientada a generar respuesta colectiva inmediata. Características: lenguaje de urgencia temporal, invitaciones a participación activa, marcos de responsabilidad cívica que requieren acción.",
                    
                    "general": "Comunicación descriptiva, informativa o conversacional sin características problemáticas identificables."
                }
            }
        }
    
    def generate_prompt(self, 
                       text: str, 
                       analysis_type: AnalysisType,
                       context: PromptContext,
                       complexity_level: str = "medium",
                       model_type: str = "transformers") -> str:
        """
        Generate a sophisticated prompt based on analysis context and uncertainty.
        
        Args:
            text: Content to analyze
            analysis_type: Type of analysis needed
            context: Analysis context with detected patterns
            complexity_level: Prompt complexity (simple/medium/complex)
            model_type: Target model type (transformers/ollama)
        """
        template = self.prompt_templates[analysis_type]
        
        # Build context-aware prompt
        prompt_parts = [
            template["system"],
            "",
            f"CONTENIDO A ANALIZAR:",
            f'"{text}"',
            ""
        ]
        
        # Add context information to guide analysis
        if context.detected_categories:
            prompt_parts.extend([
                f"PATRONES DETECTADOS: {', '.join(context.detected_categories)}",
                ""
            ])
        
        if context.targeted_groups:
            prompt_parts.extend([
                f"GRUPOS MENCIONADOS: {', '.join(context.targeted_groups)}",
                ""
            ])
        
        if context.political_topic and context.political_topic != "no_político":
            prompt_parts.extend([
                f"CONTEXTO POLÍTICO: {context.political_topic}",
                ""
            ])
        
        # Add uncertainty-focused questions for areas needing LLM insight
        if context.uncertainty_areas:
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
                *[f"- {q}" for q in template["questions"]],
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
        Generate targeted classification prompt with specific pattern matching.
        Updated to be less conservative and catch more political content.
        """
        prompt_parts = [
            f'TEXTO: "{text}"',
            "",
            "ANÁLISIS PASO A PASO:",
            "",
            "PASO 1 - Detectar contenido político específico:",
            "- 'ROJOS', 'comunistas', criticas a Vox/partidos → far_right_bias",
            "- 'agenda WOKE', 'progres', 'feminazis' → far_right_bias", 
            "- 'invasión', 'nos están borrando' + inmigración → far_right_bias",
            "- 'RETIRAD', 'MOVILIZAOS', 'ORGANIZARSE' → call_to_action",
            "- 'BOICOT', organizarse contra empresas → call_to_action",
            "- Narrativas de sustitución cultural → far_right_bias",
            "- 'élites', 'agenda globalista' → conspiracy_theory",
            "",
            "PASO 2 - Si contiene insultos/slurs directos:",
            "- 'moros', 'sudacas', 'feminazis', 'maricas' → hate_speech",
            "- 'genéticamente ladrones', 'parásitos' → hate_speech",
            "- 'pedofilia LGBTI' → hate_speech",
            "",
            "PASO 3 - Si no hay frases exactas, buscar patrones conceptuales:",
            "- Referencias vagas a estudios/informes sin especificar → disinformation",
            "- Preguntar sobre 'coincidencias' o beneficios sistemáticos → conspiracy_theory",
            "- Críticas a instituciones educativas por sesgo ideológico → far_right_bias",
            "- Incompatibilidad cultural entre grupos → hate_speech",
            "- Llamadas a acción defensiva o electoral → call_to_action",
            "",
            "PASO 4 - Solo si NO hay NINGÚN patrón problemático:",
            "- Contenido sobre clima, familia, comida, deportes personales → general",
            "",
            "IMPORTANTE: Sé menos conservador. Si hay matices políticos, elige la categoría específica.",
            "",
            "RESPUESTA (evalúa en este orden):"
        ]
        
        return "\n".join(prompt_parts)
    
    def create_uncertainty_context(self, pattern_results: Dict) -> UncertaintyContext:
        """
        Create context highlighting areas where pattern analysis shows uncertainty.
        This guides LLM to focus on ambiguous areas.
        """
        uncertainty_areas = []
        
        # Check for uncertain categorization
        far_right_result = pattern_results.get('far_right', {})
        detected_categories = far_right_result.get('categories', [])
        
        # Use simple heuristics instead of confidence scores
        pattern_matches = far_right_result.get('pattern_matches', [])
        # pattern_matches is a list, not a dict
        total_patterns = len(pattern_matches) if isinstance(pattern_matches, list) else 0
        
        if total_patterns == 0:
            uncertainty_areas.append("Clasificación de contenido ambigua - sin patrones claros")
        
        if len(detected_categories) > 2:
            uncertainty_areas.append("Múltiples categorías detectadas - necesita priorización")
        
        # Check for claim verification needs
        claims = pattern_results.get('claims', [])
        if claims:
            high_verifiability_claims = [c for c in claims if hasattr(c, 'verifiability') and c.verifiability.value == 'high']
            if high_verifiability_claims:
                uncertainty_areas.append("Afirmaciones verificables requieren validación")
        
        # Check for topic clarity
        topics = pattern_results.get('topics', [])
        if topics and len(topics) > 1:
            uncertainty_areas.append("Múltiples temas políticos detectados")
        
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
    return PromptContext(
        detected_categories=analysis_results.get('categories', []),
        targeted_groups=analysis_results.get('targeted_groups', []),
        claims_count=analysis_results.get('claims_count', 0),
        political_topic=analysis_results.get('category', 'general'),
        uncertainty_areas=analysis_results.get('uncertainty_areas', [])
    )

# Test function
if __name__ == "__main__":
    generator = EnhancedPromptGenerator()
    
    # Test context
    test_context = PromptContext(
        detected_categories=['hate_speech', 'xenophobia'],
        targeted_groups=['inmigrantes', 'musulmanes'],
        claims_count=1,
        political_topic='inmigración',
        uncertainty_areas=['Nivel de amenaza real', 'Intención del autor']
    )
    
    test_text = "Los musulmanes están invadiendo España y hay que pararlos ya"
    
    for analysis_type in AnalysisType:
        print(f"\n=== {analysis_type.value.upper()} ===")
        prompt = generator.generate_prompt(test_text, analysis_type, test_context)
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
