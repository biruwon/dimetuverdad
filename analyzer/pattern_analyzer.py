"""
Unified Pattern Analyzer: Merged topic classification and far-right detection.
Eliminates redundancy between SpanishPoliticalTopicClassifier and FarRightAnalyzer.
"""

import re
from typing import Dict, List
from dataclasses import dataclass
from .categories import Categories



@dataclass
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
        """Initialize consolidated pattern detection with merged overlapping categories."""
        return {
            Categories.HATE_SPEECH: {
                'patterns': [
                    # Racial/ethnic hate speech
                    r'\b(?:raza\s+inferior|sangre\s+pura|superioridad\s+racial)\b',
                    r'\b(?:hay\s+que\s+|debemos\s+|vamos\s+a\s+|queremos\s+)?(?:eliminar|deportar|expulsar)\s+(?:a\s+)?(?:los?\s+|todas?\s+|todos\s+los?\s+)?(?:musulmanes?|gitanos?|moros?|negros?)\b',
                    r'\b(?:musulmanes?|gitanos?|moros?|negros?)\b.*\b(?:hay\s+que\s+|debemos\s+|queremos\s+)?(?:eliminar|deportar|expulsar)(?:los)?(?:\s+a\s+todos?)?\b',
                    # Dehumanizing language
                    r'\b(?:alimañas?|parásitos?|escoria|basura)\s+(?:musulman|gitana|mora|negra)',
                    r'\b(?:invasión|plaga|epidemia)\s+(?:de\s+)?(?:musulmanes|moros|gitanos)',
                    # Direct hate speech with context
                    r'\b(?:siento\s+|tengo\s+|profeso\s+)?(?:odio|rencor|resentimiento)\s+(?:hacia\s+|contra\s+)?(?:los?\s+)?(?:musulmanes|inmigrantes|gitanos)\b',
                    r'\b(?:apoyo\s+la\s+|defiendo\s+la\s+)?(?:discriminación|segregación)\s+(?:de\s+|hacia\s+|contra\s+)(?:los?\s+)?(?:musulmanes|inmigrantes|gitanos)\b',
                    # Homophobic and sexist slurs
                    r'\b(?:maricas?|maricones?)\b',
                    r'\b(?:feminazis?)\b',
                    # MERGED: Xenophobic patterns (formerly separate category)
                    r'\bfuera\s+(?:los?\s+|las?\s+)?(?:moros?|extranjeros?|ilegales?|inmigrantes?|menas?)\b',
                    r'\b(?:moros?|extranjeros?|ilegales?|inmigrantes?|menas?)\s+fuera\s+(?:de\s+)?(?:España|Europa)\b',
                    r'\b(?:expuls\w+|deport\w+)\s+(?:a\s+)?(?:los?\s+)?(?:moros?|extranjeros?|ilegales?)\b',
                    r'\b(?:moros?|musulmanes?|inmigrantes?)\s+(?:nos\s+)?(?:están\s+)?(?:invadiendo|invaden)\b',
                    r'\b(?:invasión|oleada|avalancha)\s+(?:de\s+)?(?:inmigrantes?|musulmanes?|moros?)\b',
                    r'\b(?:nos\s+)?(?:están\s+)?(?:invadiendo|invaden)\s+(?:España|Europa|nuestro\s+país|nuestra\s+patria)\b',
                    # ANTI-IMMIGRANT SCAPEGOATING - Essential patterns (LLM alone insufficient)
                    r'\b(?:inmigrantes?|extranjeros?|ilegales?|menas?)\b.*\b(?:saturan?|colapsan?|saturando|colapsando)\b.*\b(?:servicios?|sanidad|hospitales?|colegios?)\b',
                    r'\b(?:servicios?|sanidad|hospitales?|colegios?)\b.*\b(?:saturad[oa]s?|colapsad[oa]s?)\b.*\b(?:inmigrantes?|extranjeros?|ilegales?)\b',
                    r'\b(?:menos|sin)\s+(?:viviendas?|casas?|pisos?|médicos?|recursos?|ayudas?)\b.*\b(?:por\s+)?(?:culpa\s+de\s+|debido\s+a\s+)?(?:los?\s+)?(?:inmigrantes?|extranjeros?)\b',
                    r'\b(?:inmigrantes?|extranjeros?)\b.*\b(?:nos\s+)?(?:quitan|roban|arrebatan)\b.*\b(?:viviendas?|trabajos?|empleos?|ayudas?|recursos?)\b',
                    r'\b(?:españoles?)\b.*\b(?:condenados?|obligados?)\s+(?:a\s+)?(?:listas?\s+de\s+espera|esperar|sufrir)\b.*\b(?:inmigrantes?|extranjeros?)\b',
                    r'\b(?:hacen|hacer)\s+negocio\s+(?:con\s+)?(?:trayéndolos?|importándolos?|ellos?)\b',
                    r'\b(?:promover|promovido|promueven)\s+una\s+invasión\b',
                    # Dehumanizing economic language
                    r'\b(?:traer|importar|meter)\s+(?:inmigrantes?|extranjeros?)\b',
                    r'\b(?:los?\s+)?(?:traen|importan|meten)\b.*\b(?:inmigrantes?|extranjeros?)\b',
                    # Derogatory language about specific groups
                    r'\b(?:estos?\s+)?menas?\b.*\b(?:robar|traficar|delinquir|criminal\w*)\b',
                    r'\b(?:harto|cansado)\s+(?:de\s+)?(?:estos?\s+)?(?:menas?|inmigrantes?|extranjeros?)\b',
                    r'\bdevolvé[dr]los?\s+(?:a\s+)?(?:su\s+país|África|donde\s+vinieron)\b',
                    # MERGED: Violence threat patterns (formerly separate category)
                    r'\b(?:matar|asesinar|eliminar|acabar\s+con)\s+(?:a\s+)?(?:los?\s+)?(?:rojos|marxistas|separatistas)\b',
                    r'\b(?:exterminio|eliminar|acabar\s+con)\s+(?:los\s+)?(?:rojos|marxistas|separatistas)\b',
                    r'\b(?:colgar|fusilar|paredón|guillotina)\b',
                    # Additional patterns from former secondary
                    r'\b(?:racismo|xenofobia|homofobia)\b',
                    r'\b(?:intolerancia|fanatismo|sectarismo)\b',
                    r'\b(?:insulto|vejación|humillación)\b',
                    r'\b(?:integración|asimilación)\s+(?:imposible|fracasada)\b',
                    r'\b(?:guetos?|paralelas?)\s+(?:culturales?|religiosas?)\b',
                    r'\b(?:violen(?:cia|to)|agresión|ataque)\b',
                    r'\b(?:amenaza|amenazar|intimidar)\b',
                    r'\b(?:armas?|armado|armamento)\b',
                    r'\b(?:justicia|venganza)\s+(?:por\s+(?:su\s+)?propia\s+mano|popular)\b',
                    r'\b(?:linchamiento|linchamient[oa])\b',
                ],
                'keywords': ['odio', 'racismo', 'discriminación', 'xenofobia', 'violencia', 'amenaza'],
                'description': 'Hate speech, xenophobia, and violent threats targeting specific groups'
            },
            
            Categories.DISINFORMATION: {
                'patterns': [
                    # General disinformation - more specific patterns to avoid flagging anti-misinformation content
                    r'\b(?:esto\s+es\s+|es\s+pura\s+|típica\s+)?(?:desinformación|fake\s+news|noticias?\s+falsas?)\b(?!\s+(?:es\s+mala|debe\s+combatirse|hay\s+que\s+evitar))',
                    r'\b(?:difunden|propagan|extienden)\s+(?:bulos?|rumores?|mentiras?)\b',
                    r'\b(?:esto\s+es\s+un\s+|puro\s+|típico\s+)?(?:bulo|rumor)\b(?!\s+(?:es\s+falso|está\s+desmentido))',
                    r'\b(?:es\s+)?(?:mentira|falsedad)\s+(?:que\s+)?(?:digan|afirmen|crean)\b',
                    r'\b(?:campaña\s+de\s+|operación\s+de\s+)?(?:manipulación|tergiversación|distorsión)\b',
                    # MERGED: Health disinformation patterns (formerly separate category)
                    r'\b(?:vacunas?|vacunación)\s+(?:con\s+)?(?:microchips?|chips?|5g|bill\s+gates|control\s+mental)',
                    r'\b(?:vacunas?)\s+(?:tienen|contienen|llevan)\s+(?:microchips?|chips?|nanobots?)',
                    r'\b(?:bill\s+gates)\s+(?:y\s+las\s+)?(?:vacunas?|control)',
                    r'\b(?:covid|coronavirus)\s+(?:es\s+)?(?:mentira|falso|inexistente|inventado)',
                    r'\b(?:plandemia|dictadura\s+(?:sanitaria|digital)|farmafia)\b',
                    # Specific patterns found in failing tests
                    r'\b(?:grafeno)\b.*\b(?:controlarnos|control)\b',
                    r'\b(?:5g)\b.*\b(?:controlarnos|control)\b',
                    r'\b(?:vacunas?).*\b(?:grafeno|5g)\b',
                    r'\b(?:\d+\s+de\s+cada\s+\d+)\s+(?:casos?)\s+(?:de\s+covid|son\s+inventados?)\b',
                    # SPECIFIC CLAIM PATTERNS - Only very specific false statistical claims
                    r'\b(?:el\s+)?(?:100|90|80)\s*%\s+(?:de\s+)?(?:los\s+)?(?:médicos|científicos)\s+(?:están\s+)?(?:comprados?|sobornados?|pagados?)\b',
                    r'\b(?:cero|0)\s+(?:muertes?|fallecidos?)\s+(?:por\s+)?(?:covid|coronavirus)\s+(?:real(?:es|mente)?|verdad)\b',
                    # Medical claims that could be false
                    r'\b(?:vacunas?|medicamentos?|tratamientos?)\s+(?:causan?|provocan?|generan?)\s+(\w+)',
                    r'\b(?:covid|coronavirus|pandemia)\s+(?:es|fue|será)\s+(\w+)',
                    r'\b(?:efectos?\s+(?:secundarios?|adversos?))\s+(?:de\s+)?(?:vacunas?)\s+(?:son\s+)?(?:ocultados?|negados?|minimizados?)\b',
                    # Economic disinformation - more specific patterns
                    r'\b(?:inflación|desempleo|paro)\s+(?:real\s+)?(?:es\s+del?\s+|alcanza\s+el\s+)(?:50|60|70|80|90)\s*%',
                    r'\b(?:cambio\s+climático|calentamiento\s+global)\s+(?:es\s+)?(?:mentira|falso|inexistente|inventado|hoax)\b',
                    # Demographic disinformation - more specific claims
                    r'\b(?:inmigrantes?|extranjeros?)\s+(?:ya\s+)?(?:representan|son)\s+(?:más\s+del\s+|el\s+)?(?:50|60|70|80|90)\s*%\s+(?:de\s+la\s+población|de\s+los\s+habitantes)\b',
                    # Additional patterns from former secondary
                    r'\b(?:propaganda|adoctrinamiento|lavado\s+de\s+cerebro)\b',
                    r'\b(?:censura|silenciamiento|ocultación)\b',
                    r'\b(?:verdad\s+(?:oculta|alternativa)|realidad\s+alternativa)\b',
                    # Government decrees/laws without official sources
                    r'\b(?:el\s+gobierno|gobierno)\s+(?:ha\s+aprobado|aprueba|ha\s+firmado|firma)\s+(?:un\s+)?decreto\b',
                    r'\b(?:decreto\s+(?:aprobado|firmado|promulgado)|ley\s+aprobada)\s+(?:que\s+)?(?:prohíbe|obliga|impone|restringe)\b',
                    r'\b(?:ya\s+está\s+(?:firmado|aprobado|promulgado)|ya\s+está\s+firmado)\s+(?:el\s+)?decreto\b',
                    # Political dismissals/resignations without sources
                    r'\b(?:ha\s+(?:sido\s+)?(?:destituido|cesado|dimiti[dt]o)|abandona\s+el\s+cargo)\s+(?:por|después\s+de)\b',
                    r'\b(?:ministro|ministra|director|directora|líder\s+de\s+(?:la\s+)?oposición).*?(?:ha\s+(?:sido\s+)?)?(?:destituido|destituida|cesado|cesada|dimiti[dt]o|dimitida)\b',
                    r'\b(?:exclusiva|confirmado)\s*:\s*(?:el\s+)?(?:líder|ministro|ministra|director|directora)\s+(?:ha\s+(?:sido\s+)?)?(?:destituido|cesado|dimiti[dt]o)\b',
                    # Official claims without specific sources
                    r'\b(?:ya\s+es\s+oficial|es\s+oficial|confirmado|según\s+fuentes\s+(?:oficiales?|del\s+gobierno))\b',
                    r'\b(?:según\s+fuentes\s+del\s+(?:ministerio|palacio|gobierno))\s+(?:sin\s+especificar|que\s+no\s+quieren\s+identificarse)\b',
                    # Fake official source citations
                    r'\b(?:boe|boletín\s+oficial)\s+(?:ya\s+)?(?:publicado|publicada)\b(?!\s+(?:\d+|número|num\.|\d{1,2}/\d{4}))',
                    r'\b(?:boe|boletín\s+oficial)\s+(?:ya\s+)?(?:publicado|publicada)\b(?!\s+(?:del?\s+)?\d{1,2}\s+de\s+\w+\s+de\s+\d{4})',
                    r'ya\s+está\s+publicado\s+en\s+el\s+boe\b(?!\s+(?:\d+|número|num\.|\d{1,2}/\d{4}))',
                    r'boe\s+ya\s+publicado\b(?!\s+(?:\d+|número|num\.|\d{1,2}/\d{4}))',
                    # International agreements without sources
                    r'\b(?:acuerdo\s+(?:secreto|confidencial)|pacto\s+secreto)\s+(?:con|entre)\b',
                    r'\b(?:ha\s+firmado|firmó)\s+(?:un\s+)?acuerdo\s+(?:secreto|confidencial)\b',
                    # Mandatory measures without sources
                    r'\b(?:obliga\s+a\s+(?:todos|todas)\s+(?:los\s+)?ciudadanos?\s+a|será\s+obligatorio)\b',
                    r'\b(?:bajo\s+pena\s+de\s+(?:multa|sanciones?|castigo))\b',
                    # Border/immigration measures without sources
                    r'\b(?:cierra\s+(?:todas\s+)?(?:las\s+)?fronteras|cerrar\s+(?:todas\s+)?(?:las\s+)?fronteras)\b',
                    r'\b(?:por\s+tiempo\s+indefinido|de\s+forma\s+permanente)\b',
                    # Digital dictatorship patterns
                    r'\b(?:dictadura\s+digital|imponen\s+(?:la\s+)?dictadura\s+digital|quieren\s+(?:una\s+)?dictadura\s+digital)\b',
                ],
                'keywords': ['desinformación', 'bulo', 'mentira', 'manipulación', 'vacunas', 'covid', 'estadísticas', 'estudios', 'decreto', 'oficial', 'confirmado', 'fuentes', 'gobierno', 'aprobado', 'firmado'],
                'description': 'False information including health, statistical, and factual disinformation'
            },
            
            Categories.CONSPIRACY_THEORY: {
                'patterns': [
                    # Classic conspiracy theories
                    r'\b(?:plan\s+kalergi|gran\s+reemplazo|reemplaz[oa]\s+populacional)\b',
                    r'\b(?:nuevo\s+orden\s+mundial|deep\s+state|estado\s+profundo)\b',
                    r'\b(?:soros|globalistas?|masoner[íi]a|illuminati|bilderberg)\b',
                    r'\b(?:agenda\s+(?:2030|oculta)|gran\s+reset(?:eo)?)\b',
                    # Conspiracy language
                    r'\b(?:conspiración|complot|conjura)\b',
                    r'\b(?:élite|oligarqu[íi]a)\s+(?:global|mundial|oculta)\b',
                    # Additional patterns from former secondary
                    r'\b(?:manipulación|control)\s+(?:mental|social)\b',
                    r'\b(?:medios\s+(?:de\s+)?comunicación|prensa)\s+(?:manipulad[ao]|comprad[ao])\b',
                ],
                'keywords': ['conspiración', 'élite', 'soros', 'agenda'],
                'description': 'Conspiracy theories and hidden agenda narratives'
            },
            
            Categories.ANTI_IMMIGRATION: {
                'patterns': [
                    # Immigration invasion narratives
                    r'\b(?:invasión|nos\s+están\s+inundando|nos\s+están\s+invadiendo)\b',
                    r'\b(?:nos\s+están\s+borrando|nos\s+quieren\s+borrar|sustitución\s+cultural)\b',
                    r'\b(?:gran\s+sustitución|teoría\s+de\s+la\s+sustitución)\b',
                    r'\b(?:inmigrantes?\s+ilegales?|ilegales\s+inmigrantes?)\b',
                    r'\b(?:control\s+de\s+fronteras?|fronteras\s+abiertas?|fronteras\s+protegidas?)\b',
                    # Economic burden claims
                    r'\b(?:nos\s+quitan\s+el\s+trabajo|nos\s+quitan\s+nuestros?\s+empleos?)\b',
                    r'\b(?:carga\s+económica|nos\s+cuestan\s+mucho|nos\s+arruinan)\b',
                    r'\b(?:vienen\s+a\s+vivir\s+de\s+nuestras?\s+prestaciones?)\b',
                    # Cultural incompatibility
                    r'\b(?:no\s+se\s+integran|no\s+quieren\s+integrarse)\b',
                    r'\b(?:imponen\s+sus\s+costumbres|imponen\s+su\s+religión)\b',
                    r'\b(?:compatibilidad\s+cultural|incompatible\s+culturalmente)\b',
                ],
                'keywords': ['inmigración', 'invasión', 'fronteras', 'sustitución', 'ilegales'],
                'description': 'Anti-immigration rhetoric and xenophobia'
            },
            
            Categories.ANTI_LGBTQ: {
                'patterns': [
                    # Gender ideology attacks
                    r'\b(?:ideología\s+de\s+género|doctrina\s+de\s+género)\b',
                    r'\b(?:agenda\s+lgbt|LGBT\s+nos\s+quiere|quieren\s+adoctrinar)\b',
                    r'\b(?:quieren\s+convertir\s+a\s+nuestros?\s+hijos?|adoctrinamiento\s+infantil)\b',
                    r'\b(?:van\s+a\s+por\s+los\s+niños?|van\s+a\s+por\s+nuestros?\s+hijos?)\b',
                    # Traditional family defense
                    r'\b(?:defensa\s+de\s+la\s+familia\s+tradicional)\b',
                    r'\b(?:familia\s+tradicional\s+en\s+peligro|amenaza\s+a\s+la\s+familia)\b',
                    r'\b(?:valores\s+cristianos?\s+atacados?|tradiciones\s+familiares)\b',
                    # Anti-trans rhetoric
                    r'\b(?:hombres\s+con\s+vestidos?|mujeres\s+con\s+pantalones?)\b',
                    r'\b(?:biología\s+binaria|hay\s+solo\s+dos\s+géneros)\b',
                    r'\b(?:deporte\s+femenino\s+contaminado|deporte\s+femenino\s+invadido)\b',
                ],
                'keywords': ['lgbt', 'género', 'trans', 'familia', 'tradicional', 'niños'],
                'description': 'Anti-LGBTQ rhetoric and gender ideology attacks'
            },
            
            Categories.ANTI_FEMINISM: {
                'patterns': [
                    # Feminazi rhetoric
                    r'\b(?:feminazis?|feministas?\s+radicales?|feministas?\s+extremas?)\b',
                    r'\b(?:feminismo\s+es\s+odio|el\s+feminismo\s+destruye)\b',
                    r'\b(?:feminismo\s+radical|ideología\s+feminista)\b',
                    # Traditional gender roles
                    r'\b(?:mujeres\s+en\s+casa|mujeres\s+para\s+la\s+cocina)\b',
                    r'\b(?:hombres\s+proveedores?|mujeres\s+amas\s+de\s+casa)\b',
                    r'\b(?:roles\s+tradicionales?\s+de\s+género)\b',
                    # False accusations
                    r'\b(?:falsas\s+acusaciones?\s+de\s+violación|violación\s+falsa)\b',
                    r'\b(?:caza\s+de\s+brujas\s+feminista|acoso\s+a\s+hombres)\b',
                    r'\b(?:machismo\s+inverso|matriarcado\s+opresivo)\b',
                ],
                'keywords': ['feminismo', 'feminazi', 'género', 'machismo', 'tradicional'],
                'description': 'Anti-feminism and traditional gender role promotion'
            },
            
            Categories.CALL_TO_ACTION: {
                'patterns': [
                    # Direct mobilization calls
                    r'\b(?:movilizaos|organizaos|organicen|retirad|sacad|actuad\s+ya)\b',
                    r'\b(?:todos\s+a\s+(?:las\s+calles|protestar|manifestar|movilizar))\b',
                    r'\b(?:convocatoria|difunde\s+(?:esta|la)\s+(?:convocatoria|manifestación))\b',
                    r'\b(?:concentración|manifestación|protesta|marcha)\s+(?:el\s+)?\w+',
                    r'\b(?:boicot|boicotear|boicoteemos)\b',
                    # Action language
                    r'\b(?:revolución|rebelión|alzamiento|resistencia)\b',
                    r'\b(?:a\s+las\s+calles|hay\s+que\s+actuar|salir\s+a\s+protestar)\b',
                    # Subtle calls to action
                    r'\b(?:hagan\s+algo|hay\s+que\s+hacer\s+algo)\b',
                    r'\b(?:no\s+(?:podemos|podéis)\s+quedarnos?\s+(?:de\s+)?brazos\s+cruzados)\b',
                    # Additional patterns from former secondary
                    r'\b(?:activismo|militancia|compromiso)\b',
                    r'\b(?:organización|coordinación|planificación)\b',
                    r'\b(?:solidaridad|apoyo|respaldo)\b',
                ],
                'keywords': ['movilización', 'protesta', 'manifestación', 'boicot'],
                'description': 'Calls to action and mobilization'
            },
            
            Categories.NATIONALISM: {
                'patterns': [
                    # Spanish nationalism
                    r'\b(?:españa|europa)\s+primero\b',
                    r'\b(?:españa|europa)\s+para\s+(?:los\s+)?españoles?\b',
                    r'\b(?:patria|patriotas?|patriot(?:a|ismo)|nación|nacional)\b',
                    r'\b(?:identidad|esencia)\s+(?:española|nacional)\s+(?:amenazada|en\s+peligro)\b',
                    r'\b(?:pureza|autenticidad)\s+(?:española|nacional|racial)\b',
                    # Nationalist language
                    r'\b(?:soberan[íi]a|independencia)\b',
                    r'\b(?:hispanidad|españolidad)\b',
                    r'\b(?:bandera|himno|símbolos?\s+nacional(?:es)?)\b',
                    # Additional patterns from former secondary
                    r'\b(?:reconquista|recuperar\s+españa)\b',
                    r'\b(?:tradición|tradiciones|ancestr(?:al|os))\b',
                ],
                'keywords': ['nacionalismo', 'patria', 'identidad', 'soberanía'],
                'description': 'Nationalist rhetoric and identity politics'
            },

            
            Categories.ANTI_GOVERNMENT: {
                'patterns': [
                    # Government as illegitimate
                    r'\b(?:régimen|dictadura|tiranía)\s+(?:de\s+)?(?:sánchez|socialista)\b',
                    r'\b(?:gobierno|administración)\s+(?:corrupt[oa]|ilegítim[oa]|dictatorial)\b',
                    r'\b(?:golpe\s+de\s+estado|derrocar|derribar)\s+(?:al\s+)?gobierno\b',
                    # Anti-left rhetoric (now part of anti_government category)
                    r'\b(?:socialistas?|comunistas?|marxistas?|rojos?)\s+(?:han\s+)?(?:destruido|destruyen|arruinado|arruinan)\s+(?:españa|el\s+país)',
                    r'\b(?:agenda|ideología)\s+(?:marxista|comunista|progre)\s+(?:está\s+)?(?:destruyendo|infectando)',
                    # Anti-state rhetoric
                    r'\b(?:estado\s+profundo|deep\s+state)\b',
                    r'\b(?:traición|traidor|vendepat(?:ria|rias))\b',
                    # Additional patterns from former secondary
                    r'\b(?:oposición|resistencia|contestación)\b',
                    r'\b(?:democracia|libertad|derechos)\s+(?:amenazad[ao]|en\s+peligro)\b',
                    # Subtle anti-government patterns
                    r'\b(?:sistema|instituciones?)\s+(?:podrid[ao]|corrupt[ao]|ilegítim[ao])\b',
                    r'\b(?:los\s+que\s+mandan|élites?|casta\s+política)\s+(?:no\s+)?(?:representan?|son)\b',
                    r'\b(?:desde\s+dentro|por\s+dentro)\s+(?:está|están)\s+(?:podrid[ao]|corrupt[ao])\b',
                    r'\b(?:no\s+representan?\s+al\s+pueblo|pueblo\s+real)\b',
                    r'\b(?:deslegitimación|ilegítim[ao]|ilegítim[ao])\b',
                ],
                'keywords': ['régimen', 'dictadura', 'gobierno', 'traición', 'sistema', 'podrido', 'socialistas', 'rojos'],
                'description': 'Anti-government rhetoric and delegitimization'
            },
            
            Categories.HISTORICAL_REVISIONISM: {
                'patterns': [
                    # Franco rehabilitation
                    r'\b(?:franco|franquismo)\s+(?:fue\s+)?(?:necesario|salvador|salvó|héroe|gran\s+líder|mejor)\b',
                    r'\b(?:dictadura|régimen)\s+(?:franquista|de\s+franco)\s+(?:fue\s+)?(?:mejor|próspera|gloriosa)\b',
                    r'\b(?:valle\s+de\s+los\s+caídos|fundación\s+franco)\b',
                    r'\b(?:con\s+)?franco\s+(?:se\s+)?(?:vivía|estaba|iba)\s+mejor\b',
                    # Historical denial
                    r'\b(?:república|guerra\s+civil)\s+(?:fue\s+)?(?:criminal|marxista|comunista)\b',
                    r'\b(?:víctimas\s+(?:del\s+)?franquismo)\s+(?:son\s+)?(?:mentira|exageradas?)\b',
                    # Historical manipulation and hidden truth
                    r'\b(?:historia\s+oficial|historia\s+manipulada)\s+(?:nos\s+)?(?:oculta|esconde|manipula)\b',
                    r'\b(?:contar\s+los\s+hechos\s+reales|verdad\s+histórica)\b',
                    # Additional patterns from former secondary
                    r'\b(?:memoria\s+histórica)\s+(?:es\s+)?(?:sectaria|revanchista)\b',
                    r'\b(?:leyenda\s+negra|historia\s+manipulada)\b',
                    # Subtle revisionist patterns
                    r'\b(?:historia\s+incompleta|historia\s+que\s+nos\s+enseñaron)\b',
                    r'\b(?:hechos\s+que\s+nos\s+ocultaron|verdad\s+oculta)\b',
                    r'\b(?:manipulación\s+histórica|reescritura\s+de\s+la\s+historia)\b',
                    r'\b(?:versión\s+oficial|historia\s+establecida)\s+(?:es\s+)?(?:falsa|mentirosa)\b',
                ],
                'keywords': ['franco', 'franquismo', 'dictadura', 'memoria', 'historia', 'manipulada'],
                'description': 'Historical revisionism and dictatorship rehabilitation'
            },
            
            Categories.POLITICAL_GENERAL: {
                'patterns': [
                    # Specific political institutional terms
                    r'\b(?:elecciones?|votar|sufragio|electoral)\b',
                    r'\b(?:partidos?\s+políticos?|coalición|alianza)\b',
                    r'\b(?:congreso|senado|parlamento|cortes)\b',
                    r'\b(?:constitución|ley\s+electoral|jurídico\s+político)\b',
                    # Only match "gobierno" when it's clearly institutional/formal
                    r'\b(?:gobierno\s+(?:de\s+)?(?:españa|nacional|central|autonómico))\b',
                    r'\b(?:política\s+(?:exterior|interior|económica|social))\b',
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
                matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
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
        
        # If no extremist patterns but has political context, mark as political_general
        if not detected_categories and political_context:
            primary_category = Categories.POLITICAL_GENERAL
            detected_categories = [Categories.POLITICAL_GENERAL]
        
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