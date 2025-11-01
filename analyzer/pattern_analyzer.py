"""
Unified Pattern Analyzer: Merged topic classification and far-right detection.
Eliminates redundancy between SpanishPoliticalTopicClassifier and FarRightAnalyzer.
"""

import re
from typing import Dict, List, Pattern
from dataclasses import dataclass
from .categories import Categories



@dataclass
class PatternMatch:
    category: str
    matched_text: str
    description: str
    context: str

@dataclass
class AnalysisResult:
    categories: List[str]
    pattern_matches: List[PatternMatch]
    primary_category: str
    political_context: List[str]
    keywords: List[str]

class PatternAnalyzer:
    """
    Pattern analyzer combining topic classification and far-right pattern detection.
    Eliminates redundant processing while providing comprehensive content analysis.
    """
    
    def __init__(self):
        self.patterns = self._initialize_unified_patterns()
        self.political_entities = self._initialize_political_entities()
    
    def _initialize_unified_patterns(self) -> Dict[str, Dict]:
        """Initialize consolidated pattern detection with pre-compiled regex patterns."""
        return {
            Categories.HATE_SPEECH: {
                'patterns': [
                    # Racial/ethnic hate speech
                    re.compile(r'\b(?:raza\s+inferior|sangre\s+pura|superioridad\s+racial)\b', re.IGNORECASE),
                    re.compile(r'\b(?:hay\s+que\s+|debemos\s+|vamos\s+a\s+|queremos\s+)?(?:eliminar|deportar|expulsar)\s+(?:a\s+)?(?:los?\s+|todas?\s+|todos\s+los?\s+)?(?:musulmanes?|gitanos?|moros?|negros?)\b', re.IGNORECASE),
                    re.compile(r'\b(?:musulmanes?|gitanos?|moros?|negros?)\b.*\b(?:hay\s+que\s+|debemos\s+|queremos\s+)?(?:eliminar|deportar|expulsar)(?:los)?(?:\s+a\s+todos?)?\b', re.IGNORECASE),
                    # Dehumanizing language
                    re.compile(r'\b(?:alimañas?|parásitos?|escoria|basura)\s+(?:musulman|gitana|mora|negra)', re.IGNORECASE),
                    re.compile(r'\b(?:invasión|plaga|epidemia)\s+(?:de\s+)?(?:musulmanes|moros|gitanos)', re.IGNORECASE),
                    # Direct hate speech with context
                    re.compile(r'\b(?:siento\s+|tengo\s+|profeso\s+)?(?:odio|rencor|resentimiento)\s+(?:hacia\s+|contra\s+)?(?:los?\s+)?(?:musulmanes|inmigrantes|gitanos)\b', re.IGNORECASE),
                    re.compile(r'\b(?:apoyo\s+la\s+|defiendo\s+la\s+)?(?:discriminación|segregación)\s+(?:de\s+|hacia\s+|contra\s+)(?:los?\s+)?(?:musulmanes|inmigrantes|gitanos)\b', re.IGNORECASE),
                    # Homophobic and sexist slurs
                    re.compile(r'\b(?:maricas?|maricones?)\b', re.IGNORECASE),
                    re.compile(r'\b(?:feminazis?)\b', re.IGNORECASE),
                    # MERGED: Xenophobic patterns (formerly separate category)
                    re.compile(r'\bfuera\s+(?:los?\s+|las?\s+)?(?:moros?|extranjeros?|ilegales?|inmigrantes?|menas?)\b', re.IGNORECASE),
                    re.compile(r'\b(?:moros?|extranjeros?|ilegales?|inmigrantes?|menas?)\s+fuera\s+(?:de\s+)?(?:España|Europa)\b', re.IGNORECASE),
                    re.compile(r'\b(?:expuls\w+|deport\w+)\s+(?:a\s+)?(?:los?\s+)?(?:moros?|extranjeros?|ilegales?)\b', re.IGNORECASE),
                    re.compile(r'\b(?:moros?|musulmanes?|inmigrantes?)\s+(?:nos\s+)?(?:están\s+)?(?:invadiendo|invaden)\b', re.IGNORECASE),
                    re.compile(r'\b(?:invasión|oleada|avalancha)\s+(?:de\s+)?(?:inmigrantes?|musulmanes?|moros?)\b', re.IGNORECASE),
                    re.compile(r'\b(?:nos\s+)?(?:están\s+)?(?:invadiendo|invaden)\s+(?:España|Europa|nuestro\s+país|nuestra\s+patria)\b', re.IGNORECASE),
                    # ANTI-IMMIGRANT SCAPEGOATING - Essential patterns (LLM alone insufficient)
                    re.compile(r'\b(?:inmigrantes?|extranjeros?|ilegales?|menas?)\b.*\b(?:saturan?|colapsan?|saturando|colapsando)\b.*\b(?:servicios?|sanidad|hospitales?|colegios?)\b', re.IGNORECASE),
                    re.compile(r'\b(?:servicios?|sanidad|hospitales?|colegios?)\b.*\b(?:saturad[oa]s?|colapsad[oa]s?)\b.*\b(?:inmigrantes?|extranjeros?|ilegales?)\b', re.IGNORECASE),
                    re.compile(r'\b(?:menos|sin)\s+(?:viviendas?|casas?|pisos?|médicos?|recursos?|ayudas?)\b.*\b(?:por\s+)?(?:culpa\s+de\s+|debido\s+a\s+)?(?:los?\s+)?(?:inmigrantes?|extranjeros?)\b', re.IGNORECASE),
                    re.compile(r'\b(?:inmigrantes?|extranjeros?)\b.*\b(?:nos\s+)?(?:quitan|roban|arrebatan)\b.*\b(?:viviendas?|trabajos?|empleos?|ayudas?|recursos?)\b', re.IGNORECASE),
                    re.compile(r'\b(?:españoles?)\b.*\b(?:condenados?|obligados?)\s+(?:a\s+)?(?:listas?\s+de\s+espera|esperar|sufrir)\b.*\b(?:inmigrantes?|extranjeros?)\b', re.IGNORECASE),
                    re.compile(r'\b(?:hacen|hacer)\s+negocio\s+(?:con\s+)?(?:trayéndolos?|importándolos?|ellos?)\b', re.IGNORECASE),
                    re.compile(r'\b(?:promover|promovido|promueven)\s+una\s+invasión\b', re.IGNORECASE),
                    # Dehumanizing economic language
                    re.compile(r'\b(?:traer|importar|meter)\s+(?:inmigrantes?|extranjeros?)\b', re.IGNORECASE),
                    re.compile(r'\b(?:los?\s+)?(?:traen|importan|meten)\b.*\b(?:inmigrantes?|extranjeros?)\b', re.IGNORECASE),
                    # Derogatory language about specific groups
                    re.compile(r'\b(?:estos?\s+)?menas?\b.*\b(?:robar|traficar|delinquir|criminal\w*)\b', re.IGNORECASE),
                    re.compile(r'\b(?:harto|cansado)\s+(?:de\s+)?(?:estos?\s+)?(?:menas?|inmigrantes?|extranjeros?)\b', re.IGNORECASE),
                    re.compile(r'\bdevolvé[dr]los?\s+(?:a\s+)?(?:su\s+país|África|donde\s+vinieron)\b', re.IGNORECASE),
                    # MERGED: Violence threat patterns (formerly separate category)
                    re.compile(r'\b(?:matar|asesinar|eliminar|acabar\s+con)\s+(?:a\s+)?(?:los?\s+)?(?:rojos|marxistas|separatistas)\b', re.IGNORECASE),
                    re.compile(r'\b(?:exterminio|eliminar|acabar\s+con)\s+(?:los\s+)?(?:rojos|marxistas|separatistas)\b', re.IGNORECASE),
                    re.compile(r'\b(?:colgar|fusilar|paredón|guillotina)\b', re.IGNORECASE),
                    # Additional patterns from former secondary
                    re.compile(r'\b(?:racismo|xenofobia|homofobia)\b', re.IGNORECASE),
                    re.compile(r'\b(?:intolerancia|fanatismo|sectarismo)\b', re.IGNORECASE),
                    re.compile(r'\b(?:insulto|vejación|humillación)\b', re.IGNORECASE),
                    re.compile(r'\b(?:integración|asimilación)\s+(?:imposible|fracasada)\b', re.IGNORECASE),
                    re.compile(r'\b(?:guetos?|paralelas?)\s+(?:culturales?|religiosas?)\b', re.IGNORECASE),
                    re.compile(r'\b(?:violen(?:cia|to)|agresión|ataque)\b', re.IGNORECASE),
                    re.compile(r'\b(?:amenaza|amenazar|intimidar)\b', re.IGNORECASE),
                    re.compile(r'\b(?:armas?|armado|armamento)\b', re.IGNORECASE),
                    re.compile(r'\b(?:justicia|venganza)\s+(?:por\s+(?:su\s+)?propia\s+mano|popular)\b', re.IGNORECASE),
                    re.compile(r'\b(?:linchamiento|linchamient[oa])\b', re.IGNORECASE),
                ],
                'keywords': ['odio', 'racismo', 'discriminación', 'xenofobia', 'violencia', 'amenaza'],
                'description': 'Hate speech, xenophobia, and violent threats targeting specific groups'
            },
            
            Categories.DISINFORMATION: {
                'patterns': [
                    # General disinformation - more specific patterns to avoid flagging anti-misinformation content
                    re.compile(r'\b(?:esto\s+es\s+|es\s+pura\s+|típica\s+)?(?:desinformación|fake\s+news|noticias?\s+falsas?)\b(?!\s+(?:es\s+mala|debe\s+combatirse|hay\s+que\s+evitar))', re.IGNORECASE),
                    re.compile(r'\b(?:difunden|propagan|extienden)\s+(?:bulos?|rumores?|mentiras?)\b', re.IGNORECASE),
                    re.compile(r'\b(?:esto\s+es\s+un\s+|puro\s+|típico\s+)?(?:bulo|rumor)\b(?!\s+(?:es\s+falso|está\s+desmentido))', re.IGNORECASE),
                    re.compile(r'\b(?:es\s+)?(?:mentira|falsedad)\s+(?:que\s+)?(?:digan|afirmen|crean)\b', re.IGNORECASE),
                    re.compile(r'\b(?:campaña\s+de\s+|operación\s+de\s+)?(?:manipulación|tergiversación|distorsión)\b', re.IGNORECASE),
                    # MERGED: Health disinformation patterns (formerly separate category)
                    re.compile(r'\b(?:vacunas?|vacunación)\s+(?:con\s+)?(?:microchips?|chips?|5g|bill\s+gates|control\s+mental)', re.IGNORECASE),
                    re.compile(r'\b(?:vacunas?)\s+(?:tienen|contienen|llevan)\s+(?:microchips?|chips?|nanobots?)', re.IGNORECASE),
                    re.compile(r'\b(?:bill\s+gates)\s+(?:y\s+las\s+)?(?:vacunas?|control)', re.IGNORECASE),
                    re.compile(r'\b(?:covid|coronavirus)\s+(?:es\s+)?(?:mentira|falso|inexistente|inventado)', re.IGNORECASE),
                    re.compile(r'\b(?:plandemia|dictadura\s+(?:sanitaria|digital)|farmafia)\b', re.IGNORECASE),
                    # Specific patterns found in failing tests
                    re.compile(r'\b(?:grafeno)\b.*\b(?:controlarnos|control)\b', re.IGNORECASE),
                    re.compile(r'\b(?:5g)\b.*\b(?:controlarnos|control)\b', re.IGNORECASE),
                    re.compile(r'\b(?:vacunas?).*\b(?:grafeno|5g)\b', re.IGNORECASE),
                    re.compile(r'\b(?:\d+\s+de\s+cada\s+\d+)\s+(?:casos?)\s+(?:de\s+covid|son\s+inventados?)\b', re.IGNORECASE),
                    # SPECIFIC CLAIM PATTERNS - Only very specific false statistical claims
                    re.compile(r'\b(?:el\s+)?(?:100|90|80)\s*%\s+(?:de\s+)?(?:los\s+)?(?:médicos|científicos)\s+(?:están\s+)?(?:comprados?|sobornados?|pagados?)\b', re.IGNORECASE),
                    re.compile(r'\b(?:cero|0)\s+(?:muertes?|fallecidos?)\s+(?:por\s+)?(?:covid|coronavirus)\s+(?:real(?:es|mente)?|verdad)\b', re.IGNORECASE),
                    # Medical claims that could be false
                    re.compile(r'\b(?:vacunas?|medicamentos?|tratamientos?)\s+(?:causan?|provocan?|generan?)\s+(\w+)', re.IGNORECASE),
                    re.compile(r'\b(?:covid|coronavirus|pandemia)\s+(?:es|fue|será)\s+(\w+)', re.IGNORECASE),
                    re.compile(r'\b(?:efectos?\s+(?:secundarios?|adversos?))\s+(?:de\s+)?(?:vacunas?)\s+(?:son\s+)?(?:ocultados?|negados?|minimizados?)\b', re.IGNORECASE),
                    # Economic disinformation - more specific patterns
                    re.compile(r'\b(?:inflación|desempleo|paro)\s+(?:real\s+)?(?:es\s+del?\s+|alcanza\s+el\s+)(?:50|60|70|80|90)\s*%', re.IGNORECASE),
                    re.compile(r'\b(?:cambio\s+climático|calentamiento\s+global)\s+(?:es\s+)?(?:mentira|falso|inexistente|inventado|hoax)\b', re.IGNORECASE),
                    # Demographic disinformation - more specific claims
                    re.compile(r'\b(?:inmigrantes?|extranjeros?)\s+(?:ya\s+)?(?:representan|son)\s+(?:más\s+del\s+|el\s+)?(?:50|60|70|80|90)\s*%\s+(?:de\s+la\s+población|de\s+los\s+habitantes)\b', re.IGNORECASE),
                    # Additional patterns from former secondary
                    re.compile(r'\b(?:propaganda|adoctrinamiento|lavado\s+de\s+cerebro)\b', re.IGNORECASE),
                    re.compile(r'\b(?:censura|silenciamiento|ocultación)\b', re.IGNORECASE),
                    re.compile(r'\b(?:verdad\s+(?:oculta|alternativa)|realidad\s+alternativa)\b', re.IGNORECASE),
                    # Government decrees/laws without official sources
                    re.compile(r'\b(?:el\s+gobierno|gobierno)\s+(?:ha\s+aprobado|aprueba|ha\s+firmado|firma)\s+(?:un\s+)?decreto\b', re.IGNORECASE),
                    re.compile(r'\b(?:decreto\s+(?:aprobado|firmado|promulgado)|ley\s+aprobada)\s+(?:que\s+)?(?:prohíbe|obliga|impone|restringe)\b', re.IGNORECASE),
                    re.compile(r'\b(?:ya\s+está\s+(?:firmado|aprobado|promulgado)|ya\s+está\s+firmado)\s+(?:el\s+)?decreto\b', re.IGNORECASE),
                    # Political dismissals/resignations without sources
                    re.compile(r'\b(?:ha\s+(?:sido\s+)?(?:destituido|cesado|dimiti[dt]o)|abandona\s+el\s+cargo)\s+(?:por|después\s+de)\b', re.IGNORECASE),
                    re.compile(r'\b(?:ministro|ministra|director|directora|líder\s+de\s+(?:la\s+)?oposición).*?(?:ha\s+(?:sido\s+)?)?(?:destituido|destituida|cesado|cesada|dimiti[dt]o|dimitida)\b', re.IGNORECASE),
                    re.compile(r'\b(?:exclusiva|confirmado)\s*:\s*(?:el\s+)?(?:líder|ministro|ministra|director|directora)\s+(?:ha\s+(?:sido\s+)?)?(?:destituido|cesado|dimiti[dt]o)\b', re.IGNORECASE),
                    # Official claims without specific sources
                    re.compile(r'\b(?:ya\s+es\s+oficial|es\s+oficial|confirmado|según\s+fuentes\s+(?:oficiales?|del\s+gobierno))\b', re.IGNORECASE),
                    re.compile(r'\b(?:según\s+fuentes\s+del\s+(?:ministerio|palacio|gobierno))\s+(?:sin\s+especificar|que\s+no\s+quieren\s+identificarse)\b', re.IGNORECASE),
                    # Fake official source citations
                    re.compile(r'\b(?:boe|boletín\s+oficial)\s+(?:ya\s+)?(?:publicado|publicada)\b(?!\s+(?:\d+|número|num\.|\d{1,2}/\d{4}))', re.IGNORECASE),
                    re.compile(r'\b(?:boe|boletín\s+oficial)\s+(?:ya\s+)?(?:publicado|publicada)\b(?!\s+(?:del?\s+)?\d{1,2}\s+de\s+\w+\s+de\s+\d{4})', re.IGNORECASE),
                    re.compile(r'ya\s+está\s+publicado\s+en\s+el\s+boe\b(?!\s+(?:\d+|número|num\.|\d{1,2}/\d{4}))', re.IGNORECASE),
                    re.compile(r'boe\s+ya\s+publicado\b(?!\s+(?:\d+|número|num\.|\d{1,2}/\d{4}))', re.IGNORECASE),
                    # International agreements without sources
                    re.compile(r'\b(?:acuerdo\s+(?:secreto|confidencial)|pacto\s+secreto)\s+(?:con|entre)\b', re.IGNORECASE),
                    re.compile(r'\b(?:ha\s+firmado|firmó)\s+(?:un\s+)?acuerdo\s+(?:secreto|confidencial)\b', re.IGNORECASE),
                    # Mandatory measures without sources
                    re.compile(r'\b(?:obliga\s+a\s+(?:todos|todas)\s+(?:los\s+)?ciudadanos?\s+a|será\s+obligatorio)\b', re.IGNORECASE),
                    re.compile(r'\b(?:bajo\s+pena\s+de\s+(?:multa|sanciones?|castigo))\b', re.IGNORECASE),
                    # Border/immigration measures without sources
                    re.compile(r'\b(?:cierra\s+(?:todas\s+)?(?:las\s+)?fronteras|cerrar\s+(?:todas\s+)?(?:las\s+)?fronteras)\b', re.IGNORECASE),
                    re.compile(r'\b(?:por\s+tiempo\s+indefinido|de\s+forma\s+permanente)\b', re.IGNORECASE),
                    # Digital dictatorship patterns
                    re.compile(r'\b(?:dictadura\s+digital|imponen\s+(?:la\s+)?dictadura\s+digital|quieren\s+(?:una\s+)?dictadura\s+digital)\b', re.IGNORECASE),
                ],
                'keywords': ['desinformación', 'bulo', 'mentira', 'manipulación', 'vacunas', 'covid', 'estadísticas', 'estudios', 'decreto', 'oficial', 'confirmado', 'fuentes', 'gobierno', 'aprobado', 'firmado'],
                'description': 'False information including health, statistical, and factual disinformation'
            },
            
            Categories.CONSPIRACY_THEORY: {
                'patterns': [
                    # Classic conspiracy theories
                    re.compile(r'\b(?:plan\s+kalergi|gran\s+reemplazo|reemplaz[oa]\s+populacional)\b', re.IGNORECASE),
                    re.compile(r'\b(?:nuevo\s+orden\s+mundial|deep\s+state|estado\s+profundo)\b', re.IGNORECASE),
                    re.compile(r'\b(?:soros|globalistas?|masoner[íi]a|illuminati|bilderberg)\b', re.IGNORECASE),
                    re.compile(r'\b(?:agenda\s+(?:2030|oculta)|gran\s+reset(?:eo)?)\b', re.IGNORECASE),
                    # Conspiracy language
                    re.compile(r'\b(?:conspiración|complot|conjura)\b', re.IGNORECASE),
                    re.compile(r'\b(?:élite|oligarqu[íi]a)\s+(?:global|mundial|oculta)\b', re.IGNORECASE),
                    # Additional patterns from former secondary
                    re.compile(r'\b(?:manipulación|control)\s+(?:mental|social)\b', re.IGNORECASE),
                    re.compile(r'\b(?:medios\s+(?:de\s+)?comunicación|prensa)\s+(?:manipulad[ao]|comprad[ao])\b', re.IGNORECASE),
                ],
                'keywords': ['conspiración', 'élite', 'soros', 'agenda'],
                'description': 'Conspiracy theories and hidden agenda narratives'
            },
            
            Categories.ANTI_IMMIGRATION: {
                'patterns': [
                    # Immigration invasion narratives
                    re.compile(r'\b(?:invasión|nos\s+están\s+inundando|nos\s+están\s+invadiendo)\b', re.IGNORECASE),
                    re.compile(r'\b(?:nos\s+están\s+borrando|nos\s+quieren\s+borrar|sustitución\s+cultural)\b', re.IGNORECASE),
                    re.compile(r'\b(?:gran\s+sustitución|teoría\s+de\s+la\s+sustitución)\b', re.IGNORECASE),
                    re.compile(r'\b(?:inmigrantes?\s+ilegales?|ilegales\s+inmigrantes?)\b', re.IGNORECASE),
                    re.compile(r'\b(?:control\s+de\s+fronteras?|fronteras\s+abiertas?|fronteras\s+protegidas?)\b', re.IGNORECASE),
                    # Economic burden claims
                    re.compile(r'\b(?:nos\s+quitan\s+el\s+trabajo|nos\s+quitan\s+nuestros?\s+empleos?)\b', re.IGNORECASE),
                    re.compile(r'\b(?:carga\s+económica|nos\s+cuestan\s+mucho|nos\s+arruinan)\b', re.IGNORECASE),
                    re.compile(r'\b(?:vienen\s+a\s+vivir\s+de\s+nuestras?\s+prestaciones?)\b', re.IGNORECASE),
                    # Cultural incompatibility
                    re.compile(r'\b(?:no\s+se\s+integran|no\s+quieren\s+integrarse)\b', re.IGNORECASE),
                    re.compile(r'\b(?:imponen\s+sus\s+costumbres|imponen\s+su\s+religión)\b', re.IGNORECASE),
                    re.compile(r'\b(?:compatibilidad\s+cultural|incompatible\s+culturalmente)\b', re.IGNORECASE),
                ],
                'keywords': ['inmigración', 'invasión', 'fronteras', 'sustitución', 'ilegales'],
                'description': 'Anti-immigration rhetoric and xenophobia'
            },
            
            Categories.ANTI_LGBTQ: {
                'patterns': [
                    # Gender ideology attacks - clear, obvious patterns
                    re.compile(r'\b(?:ideología\s+de\s+género|doctrina\s+de\s+género)\b', re.IGNORECASE),
                    re.compile(r'\b(?:agenda\s+lgbt|LGBT\s+nos\s+quiere|quieren\s+adoctrinar)\b', re.IGNORECASE),
                    re.compile(r'\b(?:quieren\s+convertir\s+a\s+nuestros?\s+hijos?|adoctrinamiento\s+infantil)\b', re.IGNORECASE),
                    re.compile(r'\b(?:van\s+a\s+por\s+los\s+niños?|van\s+a\s+por\s+nuestros?\s+hijos?)\b', re.IGNORECASE),
                    # Traditional family defense - clear patterns
                    re.compile(r'\b(?:defensa\s+de\s+la\s+familia\s+tradicional)\b', re.IGNORECASE),
                    re.compile(r'\b(?:familia\s+tradicional\s+en\s+peligro|amenaza\s+a\s+la\s+familia)\b', re.IGNORECASE),
                    re.compile(r'\b(?:valores\s+cristianos?\s+atacados?|tradiciones\s+familiares)\b', re.IGNORECASE),
                    # Only the most obvious anti-trans patterns
                    re.compile(r'\b(?:hombres\s+con\s+vestidos?|mujeres\s+con\s+pantalones?)\b', re.IGNORECASE),
                    re.compile(r'\b(?:biología\s+binaria|hay\s+solo\s+dos\s+géneros)\b', re.IGNORECASE),
                    re.compile(r'\b(?:deporte\s+femenino\s+contaminado|deporte\s+femenino\s+invadido)\b', re.IGNORECASE),
                ],
                'keywords': ['lgbt', 'género', 'trans', 'familia', 'tradicional', 'niños'],
                'description': 'Anti-LGBTQ rhetoric and gender ideology attacks'
            },
            
            Categories.ANTI_FEMINISM: {
                'patterns': [
                    # Feminazi rhetoric
                    re.compile(r'\b(?:feminazis?|feministas?\s+radicales?|feministas?\s+extremas?)\b', re.IGNORECASE),
                    re.compile(r'\b(?:feminismo\s+es\s+odio|el\s+feminismo\s+destruye)\b', re.IGNORECASE),
                    re.compile(r'\b(?:feminismo\s+radical|ideología\s+feminista)\b', re.IGNORECASE),
                    # Traditional gender roles
                    re.compile(r'\b(?:mujeres\s+en\s+casa|mujeres\s+para\s+la\s+cocina)\b', re.IGNORECASE),
                    re.compile(r'\b(?:hombres\s+proveedores?|mujeres\s+amas\s+de\s+casa)\b', re.IGNORECASE),
                    re.compile(r'\b(?:roles\s+tradicionales?\s+de\s+género)\b', re.IGNORECASE),
                    # False accusations
                    re.compile(r'\b(?:falsas\s+acusaciones?\s+de\s+violación|violación\s+falsa)\b', re.IGNORECASE),
                    re.compile(r'\b(?:caza\s+de\s+brujas\s+feminista|acoso\s+a\s+hombres)\b', re.IGNORECASE),
                    re.compile(r'\b(?:machismo\s+inverso|matriarcado\s+opresivo)\b', re.IGNORECASE),
                ],
                'keywords': ['feminismo', 'feminazi', 'género', 'machismo', 'tradicional'],
                'description': 'Anti-feminism and traditional gender role promotion'
            },
            
            Categories.CALL_TO_ACTION: {
                'patterns': [
                    # Direct mobilization calls
                    re.compile(r'\b(?:movilizaos|organizaos|organicen|retirad|sacad|actuad\s+ya)\b', re.IGNORECASE),
                    re.compile(r'\b(?:todos\s+a\s+(?:las\s+calles|protestar|manifestar|movilizar))\b', re.IGNORECASE),
                    re.compile(r'\b(?:convocatoria|difunde\s+(?:esta|la)\s+(?:convocatoria|manifestación))\b', re.IGNORECASE),
                    re.compile(r'\b(?:concentración|manifestación|protesta|marcha)\s+(?:el\s+)?\w+', re.IGNORECASE),
                    re.compile(r'\b(?:boicot|boicotear|boicoteemos)\b', re.IGNORECASE),
                    # Action language
                    re.compile(r'\b(?:revolución|rebelión|alzamiento|resistencia)\b', re.IGNORECASE),
                    re.compile(r'\b(?:a\s+las\s+calles|hay\s+que\s+actuar|salir\s+a\s+protestar)\b', re.IGNORECASE),
                    # Subtle calls to action
                    re.compile(r'\b(?:hagan\s+algo|hay\s+que\s+hacer\s+algo)\b', re.IGNORECASE),
                    re.compile(r'\b(?:no\s+(?:podemos|podéis)\s+quedarnos?\s+(?:de\s+)?brazos\s+cruzados)\b', re.IGNORECASE),
                    # Additional patterns from former secondary
                    re.compile(r'\b(?:activismo|militancia|compromiso)\b', re.IGNORECASE),
                    re.compile(r'\b(?:organización|coordinación|planificación)\b', re.IGNORECASE),
                ],
                'keywords': ['movilización', 'protesta', 'manifestación', 'boicot'],
                'description': 'Calls to action and mobilization'
            },
            
            Categories.NATIONALISM: {
                'patterns': [
                    # Spanish nationalism - more specific patterns to avoid false positives
                    re.compile(r'\b(?:españa|europa)\s+(?:primero|por\s+encima\s+de\s+todo|antes\s+que\s+nada)\b', re.IGNORECASE),
                    re.compile(r'\b(?:españa|europa)\s+(?:para\s+los?\s+)?(?:españoles?|europeos?|autóctonos?|nativos?)\s+(?:solamente|sólo|únicamente)\b', re.IGNORECASE),
                    re.compile(r'\b(?:patria|patriotas?|patriot(?:a|ismo)|nación|nacional)\s+(?:verdader[ao]|auténtic[ao]|puri?[ao])\b', re.IGNORECASE),
                    re.compile(r'\b(?:identidad|esencia|raíces?|origen(?:es)?)\s+(?:española|nacional|racial)\s+(?:amenazada|en\s+peligro|atacada)\b', re.IGNORECASE),
                    re.compile(r'\b(?:pureza|autenticidad)\s+(?:española|nacional|racial|cultural)\b', re.IGNORECASE),
                    # Nationalist rhetoric - requires context of superiority or exclusion
                    re.compile(r'\b(?:nosotros?\s+los?\s+)?(?:españoles?|autóctonos?|nativos?)\s+(?:somos\s+)?(?:superiores?|mejores?)\b', re.IGNORECASE),
                    re.compile(r'\b(?:soberan[íi]a|independencia)\s+(?:nacional|española)\s+(?:amenazada|en\s+peligro)\b', re.IGNORECASE),
                    re.compile(r'\b(?:hispanidad|españolidad)\s+(?:amenazada|perdida|recuperar)\b', re.IGNORECASE),
                    re.compile(r'\b(?:bandera|himno|símbolos?\s+nacional(?:es)?)\s+(?:españoles?|auténticos?|verdaderos?)\b', re.IGNORECASE),
                    # Nationalist calls to action or superiority
                    re.compile(r'\b(?:reconquista|recuperar)\s+(?:españa|nuestra\s+patria|nuestro\s+país)\b', re.IGNORECASE),
                    re.compile(r'\b(?:tradición|tradiciones|ancestr(?:al|os))\s+(?:españoles?|auténtic[ao]s?|puri?[ao]s?)\b', re.IGNORECASE),
                    # Avoid single words like "patriota" without nationalist context
                ],
                'keywords': ['nacionalismo', 'patria', 'identidad', 'soberanía', 'españoles superiores'],
                'description': 'Nationalist rhetoric emphasizing national superiority or identity threats'
            },

            
            Categories.ANTI_GOVERNMENT: {
                'patterns': [
                    # Government as illegitimate
                    re.compile(r'\b(?:régimen|dictadura|tiranía)\s+(?:de\s+)?(?:sánchez|socialista)\b', re.IGNORECASE),
                    re.compile(r'\b(?:gobierno|administración)\s+(?:corrupt[oa]|ilegítim[oa]|dictatorial)\b', re.IGNORECASE),
                    re.compile(r'\b(?:golpe\s+de\s+estado|derrocar|derribar)\s+(?:al\s+)?gobierno\b', re.IGNORECASE),
                    # Anti-left rhetoric (now part of anti_government category)
                    re.compile(r'\b(?:socialistas?|comunistas?|marxistas?|rojos?)\s+(?:han\s+)?(?:destruido|destruyen|arruinado|arruinan)\s+(?:españa|el\s+país)', re.IGNORECASE),
                    re.compile(r'\b(?:agenda|ideología)\s+(?:marxista|comunista|progre)\s+(?:está\s+)?(?:destruyendo|infectando)', re.IGNORECASE),
                    # Anti-state rhetoric
                    re.compile(r'\b(?:estado\s+profundo|deep\s+state)\b', re.IGNORECASE),
                    re.compile(r'\b(?:traición|traidor|vendepat(?:ria|rias))\b', re.IGNORECASE),
                    # Additional patterns from former secondary
                    re.compile(r'\b(?:oposición|resistencia|contestación)\b', re.IGNORECASE),
                    re.compile(r'\b(?:democracia|libertad|derechos)\s+(?:amenazad[ao]|en\s+peligro)\b', re.IGNORECASE),
                    # Subtle anti-government patterns
                    re.compile(r'\b(?:sistema|instituciones?)\s+(?:podrid[ao]|corrupt[ao]|ilegítim[ao])\b', re.IGNORECASE),
                    re.compile(r'\b(?:los\s+que\s+mandan|élites?|casta\s+política)\s+(?:no\s+)?(?:representan?|son)\b', re.IGNORECASE),
                    re.compile(r'\b(?:desde\s+dentro|por\s+dentro)\s+(?:está|están)\s+(?:podrid[ao]|corrupt[ao])\b', re.IGNORECASE),
                    re.compile(r'\b(?:no\s+representan?\s+al\s+pueblo|pueblo\s+real)\b', re.IGNORECASE),
                    re.compile(r'\b(?:deslegitimación|ilegítim[ao]|ilegítim[ao])\b', re.IGNORECASE),
                ],
                'keywords': ['régimen', 'dictadura', 'gobierno', 'traición', 'sistema', 'podrido', 'socialistas', 'rojos'],
                'description': 'Anti-government rhetoric and delegitimization'
            },
            
            Categories.HISTORICAL_REVISIONISM: {
                'patterns': [
                    # Franco rehabilitation
                    re.compile(r'\b(?:franco|franquismo)\s+(?:fue\s+)?(?:necesario|salvador|salvó|héroe|gran\s+líder|mejor)\b', re.IGNORECASE),
                    re.compile(r'\b(?:dictadura|régimen)\s+(?:franquista|de\s+franco)\s+(?:fue\s+)?(?:mejor|próspera|gloriosa)\b', re.IGNORECASE),
                    re.compile(r'\b(?:valle\s+de\s+los\s+caídos|fundación\s+franco)\b', re.IGNORECASE),
                    re.compile(r'\b(?:con\s+)?franco\s+(?:se\s+)?(?:vivía|estaba|iba)\s+mejor\b', re.IGNORECASE),
                    # Historical denial
                    re.compile(r'\b(?:república|guerra\s+civil)\s+(?:fue\s+)?(?:criminal|marxista|comunista)\b', re.IGNORECASE),
                    re.compile(r'\b(?:víctimas\s+(?:del\s+)?franquismo)\s+(?:son\s+)?(?:mentira|exageradas?)\b', re.IGNORECASE),
                    # Historical manipulation and hidden truth
                    re.compile(r'\b(?:historia\s+oficial|historia\s+manipulada)\s+(?:nos\s+)?(?:oculta|esconde|manipula)\b', re.IGNORECASE),
                    re.compile(r'\b(?:contar\s+los\s+hechos\s+reales|verdad\s+histórica)\b', re.IGNORECASE),
                    # Additional patterns from former secondary
                    re.compile(r'\b(?:memoria\s+histórica)\s+(?:es\s+)?(?:sectaria|revanchista)\b', re.IGNORECASE),
                    re.compile(r'\b(?:leyenda\s+negra|historia\s+manipulada)\b', re.IGNORECASE),
                    # Subtle revisionist patterns
                    re.compile(r'\b(?:historia\s+incompleta|historia\s+que\s+nos\s+enseñaron)\b', re.IGNORECASE),
                    re.compile(r'\b(?:hechos\s+que\s+nos\s+ocultaron|verdad\s+oculta)\b', re.IGNORECASE),
                    re.compile(r'\b(?:manipulación\s+histórica|reescritura\s+de\s+la\s+historia)\b', re.IGNORECASE),
                    re.compile(r'\b(?:versión\s+oficial|historia\s+establecida)\s+(?:es\s+)?(?:falsa|mentirosa)\b', re.IGNORECASE),
                ],
                'keywords': ['franco', 'franquismo', 'dictadura', 'memoria', 'historia', 'manipulada'],
                'description': 'Historical revisionism and dictatorship rehabilitation'
            },
            
            Categories.POLITICAL_GENERAL: {
                'patterns': [
                    # Specific political institutional terms
                    re.compile(r'\b(?:elecciones?|votar|sufragio|electoral)\b', re.IGNORECASE),
                    re.compile(r'\b(?:partidos?\s+políticos?|coalición|alianza)\b', re.IGNORECASE),
                    re.compile(r'\b(?:congreso|senado|parlamento|cortes)\b', re.IGNORECASE),
                    re.compile(r'\b(?:constitución|ley\s+electoral|jurídico\s+político)\b', re.IGNORECASE),
                    # Only match "gobierno" when it's clearly institutional/formal
                    re.compile(r'\b(?:gobierno\s+(?:de\s+)?(?:españa|nacional|central|autonómico))\b', re.IGNORECASE),
                    re.compile(r'\b(?:política\s+(?:exterior|interior|económica|social))\b', re.IGNORECASE),
                    # Remove overly broad single word patterns that catch everything
                ],
                'keywords': ['política', 'gobierno', 'democracia', 'elecciones'],
                'description': 'General political content without extremist elements'
            }
        }
    
    def _initialize_political_entities(self) -> Dict[str, List[str]]:
        """Initialize political entities for context detection."""
        return {
            'personas': [
                'sánchez', 'iglesias', 'abascal', 'casado', 'rivera', 'díaz ayuso',
                'franco', 'hitler', 'mussolini', 'soros', 'bill gates'
            ],
            'partidos': [
                'psoe', 'pp', 'vox', 'podemos', 'ciudadanos', 'falange',
                'democracia nacional', 'hogar social'
            ],
            'organizaciones': [
                'eu', 'ue', 'otan', 'onu', 'fmi', 'davos', 'bilderberg'
            ]
        }
    
    def analyze_content(self, text: str) -> AnalysisResult:
        """
        Unified content analysis combining topic classification and extremism detection.
        """
        if not text or len(text.strip()) < 5:
            return AnalysisResult(
                categories=[],
                pattern_matches=[],
                primary_category="non_political",
                political_context=[],
                keywords=[]
            )
        
        text_lower = text.lower()
        pattern_matches = []
        detected_categories = []
        all_keywords = set()
        
        # Analyze each category
        for category, config in self.patterns.items():
            category_matches = []
            
            # Check all patterns (no more primary/secondary distinction)
            for pattern in config['patterns']:
                matches = list(re.finditer(pattern, text_lower))
                for match in matches:
                    category_matches.append(PatternMatch(
                        category=category,
                        matched_text=match.group(),
                        description=config['description'],
                        context=text[max(0, match.start()-20):match.end()+20].strip()
                    ))
            
            if category_matches:
                detected_categories.append(category)
                pattern_matches.extend(category_matches)
                all_keywords.update(config['keywords'])
        
        # Detect political context
        political_context = self._detect_political_context(text)
        
        # Determine primary category - first detected category (no scoring)
        primary_category = detected_categories[0] if detected_categories else "non_political"
        
        
        return AnalysisResult(
            categories=detected_categories,
            pattern_matches=pattern_matches,
            primary_category=primary_category,
            political_context=political_context,
            keywords=list(all_keywords)
        )
    
    def _detect_political_context(self, text: str) -> List[str]:
        """Detect political entities and context."""
        context = []
        text_lower = text.lower()
        
        for entity_type, entities in self.political_entities.items():
            for entity in entities:
                # Use word boundaries to avoid false positives like "que" matching "ue"
                pattern = r'\b' + re.escape(entity) + r'\b'
                if re.search(pattern, text_lower):
                    context.append(f"{entity_type}:{entity}")
        
        return context[:10]  # Limit context items