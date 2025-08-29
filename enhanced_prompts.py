"""
Enhanced prompting system for LLM analysis of far-right activism content.
Specialized prompts for different types of analysis and content.
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class AnalysisType(Enum):
    BIAS_DETECTION = "bias_detection"
    MISINFORMATION = "misinformation"
    THREAT_ASSESSMENT = "threat_assessment"
    CLAIM_VERIFICATION = "claim_verification"
    MOBILIZATION = "mobilization"
    COMPREHENSIVE = "comprehensive"

@dataclass
class PromptContext:
    """Context information for generating appropriate prompts."""
    content_type: str
    far_right_score: float
    threat_level: str
    detected_topics: List[str]
    claims_detected: List[Dict]
    pattern_matches: List[Dict]
    targeted_groups: List[str]

class EnhancedPromptGenerator:
    """
    Generates sophisticated prompts for LLM analysis of Spanish far-right content.
    Adapts prompts based on detected patterns and analysis context.
    """
    
    def __init__(self):
        self.base_instructions = self._get_base_instructions()
        self.context_templates = self._get_context_templates()
        self.output_schemas = self._get_output_schemas()
        
    def _get_base_instructions(self) -> str:
        """Core instructions for all LLM analysis."""
        return """
CONTEXTO: Eres un experto analista especializado en detectar activismo de extrema derecha en redes sociales españolas. Tu función es analizar contenido en busca de:
- Desinformación y teorías conspiratorias (especialmente anti-inmigración, anti-globalización)
- Incitación al odio y amenazas contra minorías, inmigrantes, políticos
- Retórica nacionalista extrema, nostalgia franquista o fascista
- Llamadas a la movilización o acción (manifestaciones, boicots, "resistencia")
- Sesgos políticos y manipulación emocional (miedo, ira, victimización)

PATRONES ESPECÍFICOS ESPAÑOLES A DETECTAR:
- Referencias a "invasión migratoria" o "gran reemplazo"
- Ataques a "élite globalista", "agenda 2030", "nuevo orden mundial"
- Nostalgia del franquismo o referencias positivas a la dictadura
- Ataques sistemáticos contra feminismo, LGTBI+, "ideología de género"
- Teorías conspirativas sobre COVID-19, vacunas, "dictadura sanitaria"
- Lenguaje sobre "España primero", "reconquista", "despertar nacional"
- Deslegitimación de instituciones democráticas y medios "mainstream"

DIRECTRICES CRÍTICAS:
1. Responde EXCLUSIVAMENTE en español
2. Mantén objetividad y base el análisis en evidencia textual específica
3. Distingue entre opinión política legítima y extremismo peligroso
4. Identifica técnicas de manipulación y propaganda específicas
5. Evalúa el potencial de daño social, radicalización y violencia real
6. Considera el contexto político y social actual de España

PROHIBIDO:
- Inventar enlaces o citar fuentes externas
- Traducir el texto original
- Hacer juicios morales subjetivos sin base empírica
- Ignorar el contexto español específico y actual
- Confundir conservadurismo legítimo con extremismo
"""
    
    def _get_context_templates(self) -> Dict[str, str]:
        """Context templates for different scenarios."""
        return {
            'high_threat': """
🚨 ALERTA DE ALTO RIESGO: Este contenido ha sido identificado con nivel de amenaza {threat_level} y puntuación de extrema derecha {score:.3f}.
Patrones detectados: {patterns}
Grupos objetivo: {groups}

Presta especial atención a:
- Amenazas explícitas o implícitas de violencia física
- Llamadas a la acción inmediata con urgencia temporal
- Deshumanización de grupos específicos (inmigrantes, políticos, minorías)
- Referencias históricas fascistas, nazis o franquistas
- Lenguaje de "guerra" o "invasión" que justifique violencia
- Identificación de objetivos específicos (personas, lugares, instituciones)
""",
            
            'conspiracy_focused': """
🔍 ANÁLISIS DE CONSPIRACIONES: Se han detectado teorías conspiratorias en el contenido.
Temas identificados: {topics}
Afirmaciones detectadas: {claims}

Evalúa específicamente:
- Veracidad factual de las afirmaciones conspiratorias
- Técnicas de desinformación empleadas (datos manipulados, fuentes falsas)
- Potencial de radicalización progresiva hacia extremismo
- Conexiones con narrativas extremistas conocidas (gran reemplazo, QAnon, etc.)
- Grado de paranoia y teorías sin evidencia empírica
- Referencias a "élites ocultas" o "planes secretos"
""",
            
            'mobilization_detected': """
📢 POTENCIAL MOVILIZACIÓN: Se han detectado posibles llamadas a la acción.
Contexto: {context}

Analiza cuidadosamente:
- Urgencia y especificidad temporal de las llamadas ("este domingo", "ya", "ahora")
- Canales de movilización sugeridos (redes, grupos, ubicaciones físicas)
- Objetivos específicos de la acción propuesta (manifestaciones, boicots, "resistencia")
- Riesgo de escalada a violencia o confrontación
- Tono emocional y técnicas de manipulación para motivar acción
- Referencias a "legítima defensa" o justificaciones de violencia
""",
            
            'claims_verification': """
✅ VERIFICACIÓN DE AFIRMACIONES: Se han detectado {claim_count} afirmaciones verificables.
Afirmaciones principales: {main_claims}

Evalúa para cada afirmación:
- Verificabilidad con fuentes oficiales españolas e internacionales
- Contexto completo y posible manipulación de datos reales
- Intención desinformativa vs. error involuntario
- Impacto potencial en la opinión pública española
- Conexión con narrativas más amplias de extrema derecha
- Urgencia de desmentir o contextualizar la información
""",
            
            'spanish_context': """
🇪🇸 CONTEXTO ESPAÑOL ESPECÍFICO:
Considera especialmente:
- Historia reciente: transición democrática, memoria histórica, franquismo
- Tensiones actuales: inmigración, nacionalismos territoriales, crisis económicas
- Partidos y movimientos: VOX, España 2000, Hogar Social, grupos neonazis
- Eventos recientes: manifestaciones, atentados, crisis políticas
- Medios y redes: desinformación en español, canales de Telegram, influencers
- Legislación: leyes contra discurso de odio, memoria histórica, libertad de expresión
"""
        }
    
    def _get_output_schemas(self) -> Dict[str, Dict]:
        """JSON schemas for different analysis types."""
        return {
            AnalysisType.COMPREHENSIVE: {
                "schema": {
                    "sesgo_politico": "str: 'extrema_izquierda' | 'izquierda' | 'centro_izquierda' | 'centro' | 'centro_derecha' | 'derecha' | 'extrema_derecha' | 'indefinido'",
                    "confianza_sesgo": "float: 0.0-1.0",
                    "nivel_amenaza": "str: 'critico' | 'alto' | 'medio' | 'bajo' | 'minimo'",
                    "riesgo_desinformacion": "str: 'critico' | 'alto' | 'medio' | 'bajo' | 'minimo'",
                    "tecnicas_manipulacion": "list[str]: técnicas de propaganda/manipulación detectadas",
                    "grupos_objetivo": "list[str]: grupos específicos atacados o mencionados",
                    "llamadas_accion": {
                        "presentes": "bool",
                        "tipo": "str: tipo de acción solicitada",
                        "urgencia": "str: 'inmediata' | 'corto_plazo' | 'largo_plazo'",
                        "especificidad": "str: 'muy_especifica' | 'especifica' | 'general' | 'vaga'"
                    },
                    "verificacion_afirmaciones": "list[dict]: análisis de afirmaciones verificables",
                    "indicadores_extremismo": "list[str]: indicadores específicos de extremismo detectados",
                    "potencial_viralizacion": "str: 'alto' | 'medio' | 'bajo'",
                    "contexto_historico": "str: referencias históricas o contextuales relevantes",
                    "recomendaciones": {
                        "accion_inmediata": "str: acción recomendada",
                        "monitorizacion": "bool: requiere seguimiento",
                        "fact_checking": "str: prioridad de verificación"
                    },
                    "explicacion": "str: explicación detallada del análisis"
                }
            },
            
            AnalysisType.THREAT_ASSESSMENT: {
                "schema": {
                    "nivel_amenaza": "str: 'critico' | 'alto' | 'medio' | 'bajo'",
                    "tipo_amenaza": "list[str]: tipos de amenaza detectados",
                    "inmediatez": "str: 'inmediata' | 'corto_plazo' | 'medio_plazo' | 'largo_plazo'",
                    "objetivos_amenaza": "list[str]: objetivos o blancos de las amenazas",
                    "escalada_potencial": "bool: potencial de escalada a violencia",
                    "acciones_especificas": "list[str]: acciones específicas mencionadas",
                    "contexto_amenaza": "str: contexto que amplifica la amenaza",
                    "mitigacion_urgente": "bool: requiere acción inmediata",
                    "explicacion": "str: justificación detallada"
                }
            },
            
            AnalysisType.MISINFORMATION: {
                "schema": {
                    "es_desinformacion": "bool",
                    "nivel_confianza": "float: 0.0-1.0",
                    "tipos_desinformacion": "list[str]: tipos específicos detectados",
                    "afirmaciones_falsas": "list[dict]: afirmaciones específicamente falsas",
                    "tecnicas_empleadas": "list[str]: técnicas de desinformación usadas",
                    "fuentes_citadas": "list[str]: fuentes mencionadas o implicadas",
                    "verificabilidad": "str: 'alta' | 'media' | 'baja' | 'nula'",
                    "impacto_potencial": "str: 'alto' | 'medio' | 'bajo'",
                    "narrativas_conectadas": "list[str]: narrativas más amplias relacionadas",
                    "recomendacion_verificacion": "str: estrategia de verificación recomendada",
                    "explicacion": "str: análisis detallado"
                }
            }
        }
    
    def generate_prompt(self, 
                       text: str, 
                       analysis_type: AnalysisType,
                       context: Optional[PromptContext] = None,
                       complexity_level: str = "full") -> str:
        """
        Generate a prompt based on the analysis type, context, and model complexity level.
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis to perform
            context: Optional context from prior analysis
            complexity_level: "simple", "medium", or "full" - adapts prompt to model capabilities
        """
        if complexity_level == "simple":
            return self._generate_simple_prompt(text, analysis_type)
        elif complexity_level == "medium":
            return self._generate_medium_prompt(text, analysis_type, context)
        else:
            return self._generate_full_prompt(text, analysis_type, context)
    
    def _generate_simple_prompt(self, text: str, analysis_type: AnalysisType) -> str:
        """Generate a simplified prompt for smaller models using example-based approach."""
        
        if analysis_type == AnalysisType.THREAT_ASSESSMENT:
            return f"""Ejemplo: "Vamos a por vosotros mañana"
Análisis: Este texto contiene amenazas directas e inmediatas de violencia.
Nivel: alto

Texto: "{text}"
Análisis:"""
        
        elif analysis_type == AnalysisType.MISINFORMATION:
            return f"""Ejemplo: "Las vacunas contienen chips de control mental"
Análisis: Este texto propaga teorías conspirativas sin evidencia científica.
Tipo: desinformación

Texto: "{text}"
Análisis:"""
        
        else:  # COMPREHENSIVE
            return f"""Ejemplo: "Los inmigrantes nos roban el trabajo y son criminales"
Análisis: Este texto promueve xenofobia y estereotipos negativos infundados contra inmigrantes.
Categoría: discurso de odio
Nivel: alto

Texto: "{text}"
Análisis:"""
    
    def _generate_medium_prompt(self, text: str, analysis_type: AnalysisType, context: Optional[PromptContext] = None) -> str:
        """Generate a medium complexity prompt for mid-tier models."""
        
        base_instructions = """
CONTEXTO: Eres un experto analista especializado en detectar activismo de extrema derecha en redes sociales españolas.

DETECTA ESPECÍFICAMENTE:
- Desinformación y teorías conspiratorias (anti-inmigración, anti-globalización)
- Incitación al odio contra minorías, inmigrantes, políticos
- Retórica nacionalista extrema o nostalgia franquista
- Llamadas a la movilización o acción
- Sesgos políticos y manipulación emocional

MANTÉN OBJETIVIDAD:
- Base el análisis en evidencia textual específica
- Distingue entre opinión política legítima y extremismo
- Considera el contexto político español actual
"""
        
        # Add simplified context if available
        context_section = ""
        if context and context.far_right_score > 0.5:
            context_section = f"\n⚠️ CONTEXTO: Contenido con puntuación de extrema derecha {context.far_right_score:.2f}"
        
        # Get simplified analysis instructions
        analysis_instructions = self._get_medium_analysis_instructions(analysis_type)
        
        # Get simplified output format
        output_format = self._get_medium_output_format(analysis_type)
        
        return f"""{base_instructions}{context_section}

{analysis_instructions}

{output_format}

TEXTO A ANALIZAR:
"{text}"

RESPUESTA (JSON válido únicamente):"""
    
    def _generate_full_prompt(self, text: str, analysis_type: AnalysisType, context: Optional[PromptContext] = None) -> str:
        """Generate the full sophisticated prompt for advanced models."""
        # This is the original implementation
        # Start with base instructions
        prompt_parts = [self.base_instructions]
        
        # Add contextual information if available
        if context:
            prompt_parts.append(self._generate_context_section(context, analysis_type))
        
        # Add specific instructions for analysis type
        prompt_parts.append(self._get_analysis_instructions(analysis_type))
        
        # Add output format specification
        prompt_parts.append(self._get_output_format(analysis_type))
        
        # Add the text to analyze
        prompt_parts.append(f'\n\nTEXTO A ANALIZAR:\n"""{text}"""\n')
        
        # Add final response instruction
        prompt_parts.append("RESPUESTA (JSON válido únicamente):")
        
        return "\n".join(prompt_parts)
    
    def _get_medium_analysis_instructions(self, analysis_type: AnalysisType) -> str:
        """Get medium complexity analysis instructions."""
        instructions = {
            AnalysisType.COMPREHENSIVE: """
🎯 ANÁLISIS INTEGRAL: Evalúa sistemáticamente:

1. SESGO POLÍTICO: Identifica posicionamiento en el espectro político español
2. AMENAZAS: Detecta amenazas directas/indirectas contra personas o grupos
3. DESINFORMACIÓN: Verifica afirmaciones y detecta teorías conspirativas
4. MOVILIZACIÓN: Identifica llamadas a manifestaciones, boicots, acciones
5. IMPACTO SOCIAL: Evalúa potencial de incitar odio o radicalización

Considera el contexto político español actual y las tensiones sociales.
""",
            
            AnalysisType.THREAT_ASSESSMENT: """
🚨 EVALUACIÓN DE AMENAZAS: Analiza:

1. TIPOLOGÍA: Directa, indirecta, condicional o implícita
2. TEMPORALIDAD: Inmediata, corto plazo, medio/largo plazo
3. ESPECIFICIDAD: Personas, lugares, grupos, métodos mencionados
4. ESCALADA: Potencial de amplificación y violencia real
5. CONTEXTO ESPAÑOL: Marco legal y antecedentes históricos

Prioriza amenazas específicas, temporales y contra objetivos identificables.
""",
            
            AnalysisType.MISINFORMATION: """
🔍 ANÁLISIS DE DESINFORMACIÓN: Examina:

1. VERIFICACIÓN: Contrasta con fuentes oficiales españolas
2. TÉCNICAS: Identifica manipulación de datos y técnicas de propaganda
3. FUENTES: Evalúa credibilidad de referencias citadas
4. INTENCIONALIDAD: Desinformación deliberada vs. error honesto
5. IMPACTO ESPAÑOL: Efecto en tensiones sociales y procesos democráticos

Enfócate en desinformación que alimente extremismo de derecha.
"""
        }
        
        return instructions.get(analysis_type, instructions[AnalysisType.COMPREHENSIVE])
    
    def _get_medium_output_format(self, analysis_type: AnalysisType) -> str:
        """Get medium complexity output format."""
        if analysis_type == AnalysisType.THREAT_ASSESSMENT:
            schema = {
                "nivel_amenaza": "critico|alto|medio|bajo",
                "tipo_amenaza": ["lista de tipos detectados"],
                "inmediatez": "inmediata|corto_plazo|medio_plazo|largo_plazo",
                "objetivos_amenaza": ["objetivos o blancos identificados"],
                "explicacion": "justificación detallada"
            }
        elif analysis_type == AnalysisType.MISINFORMATION:
            schema = {
                "es_desinformacion": "true|false",
                "nivel_confianza": "0.0-1.0",
                "tipos_desinformacion": ["tipos específicos detectados"],
                "tecnicas_empleadas": ["técnicas de desinformación"],
                "explicacion": "análisis detallado"
            }
        else:  # COMPREHENSIVE
            schema = {
                "sesgo_politico": "extrema_izquierda|izquierda|centro|derecha|extrema_derecha|indefinido",
                "nivel_amenaza": "critico|alto|medio|bajo",
                "tecnicas_manipulacion": ["técnicas detectadas"],
                "grupos_objetivo": ["grupos atacados"],
                "llamadas_accion": {
                    "presentes": "true|false",
                    "tipo": "tipo de acción solicitada",
                    "urgencia": "inmediata|corto_plazo|largo_plazo"
                },
                "explicacion": "explicación detallada del análisis"
            }
        
        return f"""
FORMATO DE RESPUESTA:
Responde con un objeto JSON válido:

{json.dumps(schema, indent=2, ensure_ascii=False)}

REQUISITOS:
- JSON válido sin comentarios
- Explicaciones claras en español
- Máximo 100 palabras por explicación
"""
    
    def _generate_context_section(self, context: PromptContext, analysis_type: AnalysisType) -> str:
        """Generate contextual information section."""
        context_info = []
        
        if context.far_right_score > 0.5:
            template = self.context_templates['high_threat']
            patterns_text = ", ".join([p.get('category', 'unknown') for p in context.pattern_matches[:5]])
            groups_text = ", ".join(context.targeted_groups[:3])
            
            context_info.append(template.format(
                threat_level=context.threat_level,
                score=context.far_right_score,
                patterns=patterns_text or "Varios patrones detectados",
                groups=groups_text or "Múltiples grupos"
            ))
        
        if 'conspiración' in context.detected_topics:
            template = self.context_templates['conspiracy_focused']
            topics_text = ", ".join(context.detected_topics[:3])
            claims_text = f"{len(context.claims_detected)} afirmaciones"
            
            context_info.append(template.format(
                topics=topics_text,
                claims=claims_text
            ))
        
        if context.claims_detected:
            template = self.context_templates['claims_verification']
            main_claims = [claim.get('text', '')[:100] for claim in context.claims_detected[:3]]
            
            context_info.append(template.format(
                claim_count=len(context.claims_detected),
                main_claims="; ".join(main_claims)
            ))
        
        return "\n".join(context_info)
    
    def _get_analysis_instructions(self, analysis_type: AnalysisType) -> str:
        """Get specific instructions for each analysis type."""
        instructions = {
            AnalysisType.COMPREHENSIVE: """
🎯 ANÁLISIS INTEGRAL REQUERIDO. Evalúa sistemáticamente todos los aspectos:

1. SESGO POLÍTICO ESPECÍFICO:
   - Identifica posicionamiento en el espectro político español
   - Detecta extremismo vs. conservadurismo legítimo
   - Evalúa referencias a partidos, ideologías, líderes

2. AMENAZAS Y VIOLENCIA:
   - Amenazas directas/indirectas contra personas o grupos
   - Justificación o glorificación de violencia histórica
   - Llamadas a "resistencia armada" o "legítima defensa"

3. DESINFORMACIÓN Y CONSPIRACIONES:
   - Veracidad factual de afirmaciones específicas
   - Teorías conspirativas (gran reemplazo, agenda globalista, etc.)
   - Manipulación de datos o estadísticas

4. MOVILIZACIÓN Y ACCIÓN:
   - Llamadas específicas a manifestaciones, boicots, acciones
   - Coordinación de actividades grupales
   - Urgencia temporal y especificidad geográfica

5. IMPACTO SOCIAL Y RADICALIZACIÓN:
   - Potencial de incitar odio hacia grupos específicos
   - Capacidad de viralización y amplificación
   - Riesgo de normalización de extremismo

6. CONTEXTO HISTÓRICO ESPAÑOL:
   - Referencias al franquismo, Guerra Civil, transición
   - Nostalgia autoritaria o fascista
   - Reinterpretación histórica sesgada

Considera especialmente el contexto político español actual, tensiones migratorias, crisis económicas y polarización social.
""",
            
            AnalysisType.THREAT_ASSESSMENT: """
🚨 EVALUACIÓN PRIORITARIA DE AMENAZAS. Analiza sistemáticamente:

1. TIPOLOGÍA DE AMENAZA ESPECÍFICA:
   - Directa: "Vamos a por ti", "Te vamos a encontrar"
   - Indirecta: "Alguien debería hacer algo", "Se lo merecen"
   - Condicional: "Si siguen así...", "Cuando llegue el momento"
   - Implícita: Referencias históricas violentas, códigos

2. EVALUACIÓN TEMPORAL:
   - Inmediata: referencias temporales específicas (fechas, eventos)
   - Corto plazo: "pronto", "ya viene", "se acerca"
   - Medio/largo plazo: profecías, preparación gradual

3. ESPECIFICIDAD Y OBJETIVOS:
   - Personas identificables (nombres, cargos, descripción)
   - Lugares específicos (direcciones, instituciones, eventos)
   - Grupos amplios (inmigrantes, políticos, colectivos)
   - Métodos sugeridos (armas, tácticas, estrategias)

4. POTENCIAL DE ESCALADA:
   - Historial de violencia en contextos similares
   - Capacidad organizativa del emisor
   - Resonancia en comunidades extremistas
   - Legitimación progresiva de violencia

5. FACTORES CONTEXTUALES ESPAÑOLES:
   - Marco legal: delitos de odio, amenazas, apología
   - Antecedentes: atentados, violencia política, casos judiciales
   - Clima social: tensiones actuales, eventos desencadenantes

PRIORIZA amenazas con elementos específicos, temporales y contra objetivos identificables.
""",
            
            AnalysisType.MISINFORMATION: """
🔍 ANÁLISIS ESPECIALIZADO DE DESINFORMACIÓN. Examina meticulosamente:

1. VERIFICACIÓN FACTUAL RIGUROSA:
   - Contrasta afirmaciones con fuentes oficiales españolas
   - Identifica datos estadísticos manipulados o descontextualizados
   - Detecta fotografías, vídeos o citas falsas/manipuladas
   - Evalúa credibilidad de fuentes citadas

2. TÉCNICAS DE MANIPULACIÓN IDENTIFICADAS:
   - Cherry-picking: selección sesgada de datos
   - Correlación falsa: "después de esto, por esto"
   - Generalización abusiva: casos aislados como norma
   - Whataboutism: deflección hacia otros temas
   - Strawman: distorsión de posiciones contrarias

3. ANÁLISIS DE FUENTES:
   - Medios pseudocientíficos o conspiratorios
   - Cuentas anónimas o bots como "evidencia"
   - Autoridades falsas: "expertos" no cualificados
   - Círculos de retroalimentación: fuentes circulares

4. EVALUACIÓN DE INTENCIONALIDAD:
   - Desinformación deliberada vs. error honesto
   - Patrones repetitivos de falsedades
   - Beneficiarios políticos de la narrativa falsa
   - Timing estratégico (elecciones, crisis, eventos)

5. IMPACTO EN CONTEXTO ESPAÑOL:
   - Efecto en tensiones migratorias o territoriales
   - Influencia en procesos democráticos
   - Daño a cohesión social o institucional
   - Amplificación de prejuicios existentes

6. CONEXIONES CON NARRATIVAS EXTREMISTAS:
   - Gran reemplazo y teorías racistas
   - Antisemitismo y teorías globalistas
   - COVID-19 y conspiraciones sanitarias
   - Negacionismo histórico o revisionismo

Enfócate en desinformación que específicamente alimente extremismo de derecha en España.
""",
            
            AnalysisType.CLAIM_VERIFICATION: """
✅ VERIFICACIÓN SISTEMÁTICA DE AFIRMACIONES. Para cada afirmación detectada:

1. IDENTIFICACIÓN Y EXTRACCIÓN:
   - Separa hechos verificables de opiniones
   - Identifica afirmaciones estadísticas, históricas, científicas
   - Detecta predicciones o pronósticos presentados como hechos
   - Localiza citas atribuidas a personas o instituciones

2. CATEGORIZACIÓN ESPECÍFICA:
   - Estadísticas: cifras de inmigración, criminalidad, economía
   - Históricas: eventos del pasado, datos del franquismo, transición
   - Científicas: salud, clima, tecnología, estudios médicos
   - Políticas: declaraciones, programas, decisiones gubernamentales
   - Legales: leyes, procedimientos, casos judiciales

3. EVALUACIÓN DE VERIFICABILIDAD:
   - Alta: datos de INE, BOE, instituciones oficiales
   - Media: estudios académicos, organismos internacionales
   - Baja: encuestas privadas, medios no contrastados
   - Nula: rumores, testimonios anónimos, "fuentes reservadas"

4. ANÁLISIS CONTEXTUAL PROFUNDO:
   - Información omitida que cambia el significado
   - Periodo temporal específico de los datos
   - Definiciones y metodologías utilizadas
   - Comparaciones sesgadas o incompletas

5. PRIORIZACIÓN PARA FACT-CHECKING:
   - Crítica: afirmaciones que pueden incitar violencia
   - Alta: datos sobre inmigración, criminalidad, economía
   - Media: estadísticas históricas o sociales
   - Baja: opiniones disfrazadas de hechos

6. CONTEXTO ESPAÑOL ESPECÍFICO:
   - Contrasta con fuentes oficiales españolas (INE, ministerios)
   - Considera debates políticos actuales y sensibilidades sociales
   - Evalúa impacto en procesos democráticos o cohesión social
   - Identifica conexiones con narrativas extremistas recurrentes

Prioriza afirmaciones que puedan alimentar extremismo o afectar decisiones políticas importantes.
"""
        }
        
        return instructions.get(analysis_type, instructions[AnalysisType.COMPREHENSIVE])
    
    def _get_output_format(self, analysis_type: AnalysisType) -> str:
        """Get the output format specification for the analysis type."""
        schema = self.output_schemas.get(analysis_type, self.output_schemas[AnalysisType.COMPREHENSIVE])
        
        format_instruction = f"""
FORMATO DE RESPUESTA OBLIGATORIO:
Responde ÚNICAMENTE con un objeto JSON válido que siga esta estructura exacta:

{json.dumps(schema['schema'], indent=2, ensure_ascii=False)}

REQUISITOS:
- JSON válido sin comentarios ni texto adicional
- Todas las claves requeridas deben estar presentes
- Valores en español siguiendo las opciones especificadas
- Explicaciones claras y fundamentadas en evidencia
- Máximo 150 palabras por campo de texto
"""
        
        return format_instruction

def create_context_from_analysis(analysis_result: Dict) -> PromptContext:
    """Create prompt context from analysis results."""
    return PromptContext(
        content_type="social_media_post",
        far_right_score=analysis_result.get('far_right_score', 0.0),
        threat_level=analysis_result.get('threat_level', 'LOW'),
        detected_topics=[analysis_result.get('primary_topic', 'unknown')],
        claims_detected=analysis_result.get('verifiable_claims', []),
        pattern_matches=analysis_result.get('pattern_matches', []),
        targeted_groups=analysis_result.get('targeted_groups', [])
    )

def generate_enhanced_prompt(text: str, 
                           analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE,
                           prior_analysis: Optional[Dict] = None,
                           complexity_level: str = "full") -> str:
    """
    Convenience function to generate enhanced prompts.
    """
    generator = EnhancedPromptGenerator()
    
    context = None
    if prior_analysis:
        context = create_context_from_analysis(prior_analysis)
    
    return generator.generate_prompt(text, analysis_type, context, complexity_level)

# Test the prompt generator
if __name__ == "__main__":
    test_text = "Los inmigrantes ilegales están invadiendo España. Soros controla a los medios para ocultarlo. ¡Es hora de actuar!"
    
    generator = EnhancedPromptGenerator()
    
    # Test different analysis types
    for analysis_type in [AnalysisType.COMPREHENSIVE, AnalysisType.THREAT_ASSESSMENT, AnalysisType.MISINFORMATION]:
        print(f"\n{'='*60}")
        print(f"PROMPT FOR {analysis_type.value.upper()}")
        print(f"{'='*60}")
        
        prompt = generator.generate_prompt(test_text, analysis_type)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
