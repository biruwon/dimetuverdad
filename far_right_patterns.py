"""
Far-right activism detection system for Spanish content analysis.
Detects patterns, language, and indicators common in far-right discourse.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class FarRightCategory(Enum):
    HATE_SPEECH = "hate_speech"
    XENOPHOBIA = "xenophobia" 
    NATIONALISM = "nationalism"
    CONSPIRACY = "conspiracy"
    VIOLENCE_INCITEMENT = "violence_incitement"
    ANTI_GOVERNMENT = "anti_government"
    HISTORICAL_REVISIONISM = "historical_revisionism"
    HEALTH_DISINFORMATION = "health_disinformation"
    FAR_RIGHT_BIAS = "far_right_bias"
    CALL_TO_ACTION = "call_to_action"
    GENERAL = "general"

@dataclass
class PatternMatch:
    category: str
    pattern: str
    confidence: float
    context: str

class FarRightAnalyzer:
    """
    Analyzes text for far-right patterns, rhetoric, and indicators.
    Focused on Spanish far-right discourse patterns without scoring.
    """
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[str, List[Dict]]:
        """Initialize far-right detection patterns."""
        return {
            'hate_speech': [
                {
                    'pattern': r'\b(?:raza\s+inferior|sangre\s+pura|superioridad\s+racial)\b',
                    'description': 'Lenguaje de superioridad racial'
                },
                {
                    'pattern': r'\b(?:eliminar|deportar|expulsar)\s+.*\b(?:musulmanes?|gitanos?|moros?|negros?)\b|\b(?:musulmanes?|gitanos?|moros?|negros?)\b.*\b(?:eliminar|deportar|expulsar)\b',
                    'description': 'Llamadas a eliminación de grupos'
                },
                {
                    'pattern': r'\b(?:virus|infectan|plaga|invasión)\s+(?:musulmán|gitana|extranjera)\b',
                    'description': 'Deshumanización de grupos'
                },
                {
                    'pattern': r'\b(?:inmigrantes?|extranjeros?|musulmanes?|moros?)\b.*\b(?:plaga|virus|amenaza|invasión)\b|\b(?:plaga|virus|amenaza|invasión)\b.*\b(?:inmigrantes?|extranjeros?|musulmanes?|moros?)\b',
                    'description': 'Deshumanización de inmigrantes'
                },
                {
                    'pattern': r'\b(?:genéticamente|biológicamente)\s+inferior(?:es)?\b',
                    'description': 'Determinismo biológico'
                },
                {
                    'pattern': r'\b(?:gitanos?|negros?|árabes?)\s+(?:son\s+)?(?:genéticamente|por\s+naturaleza)\s+(?:ladrones?|mentirosos?|violentos?|criminales?)',
                    'description': 'Determinismo racial/étnico'
                },
                {
                    'pattern': r'\bfeminazis?\b.*\b(?:plaga|virus|exterminar|eliminar)\b|\b(?:plaga|virus|exterminar|eliminar)\b.*\bfeminazis?\b',
                    'description': 'Deshumanización de feministas con lenguaje violento'
                }
            ],
            'xenophobia': [
                {
                    'pattern': r'\b(?:invasión|invad\w+|avalancha|ola)\b.*\b(?:inmigrantes?|extranjeros?|moros?|musulmanes?|menas?)\b|\b(?:inmigrantes?|extranjeros?|moros?|musulmanes?|menas?)\b.*\b(?:invasión|invad\w+|avalancha|ola)\b',
                    'description': 'Inmigración como invasión'
                },
                {
                    'pattern': r'\b(?:inmigración\s+masiva|invasión)\s+(?:es\s+(?:una\s+)?)?(?:planificada|organizada|coordinada)',
                    'description': 'Inmigración como plan coordinado'
                },
                {
                    'pattern': r'\b(?:España|Europa)\s+para\s+(?:los\s+)?españoles?\b',
                    'description': 'Nacionalismo excluyente'
                },
                {
                    'pattern': r'\b(?:fuera|expuls\w+|deport\w+)\b.*\b(?:moros?|extranjeros?|ilegales?|inmigrantes?|menas?)\b|\b(?:moros?|extranjeros?|ilegales?|inmigrantes?|menas?)\b.*\b(?:fuera|expuls\w+|deport\w+)\b',
                    'description': 'Llamadas a expulsión'
                },
                {
                    'pattern': r'\b(?:menas?|sudacas?)\b.*\b(?:robar|traficar|vagos?|delincuentes?)\b|\b(?:robar|traficar|vagos?|delincuentes?)\b.*\b(?:menas?|sudacas?)\b',
                    'description': 'Estereotipos criminales sobre inmigrantes'
                }
            ],
            'nationalism': [
                {
                    'pattern': r'\b(?:reconquista|recuperar)\s+(?:España|la\s+patria)\b',
                    'description': 'Nacionalismo reconquistador'
                },
                {
                    'pattern': r'\b(?:gloria|grandeza)\s+(?:nacional|de\s+España|patria)\b',
                    'description': 'Nacionalismo glorificador'
                },
                {
                    'pattern': r'\b(?:pureza|autenticidad)\s+(?:española|nacional|racial)\b',
                    'description': 'Purismo étnico/nacional'
                },
                {
                    'pattern': r'\b(?:identidad|esencia)\s+(?:española|nacional)\s+(?:amenazada|en\s+peligro)\b',
                    'description': 'Identidad amenazada'
                }
            ],
            'conspiracy': [
                {
                    'pattern': r'\b(?:plan\s+kalergi|gran\s+reemplazo|reemplaz[oa]\s+populacional)\b',
                    'description': 'Teoría del reemplazo'
                },
                {
                    'pattern': r'\b(?:soros|élite\s+globalista|nuevo\s+orden\s+mundial)\b',
                    'description': 'Conspiración globalista'
                },
                {
                    'pattern': r'\b(?:agenda\s+(?:2030|globalista)|gran\s+reset)\b',
                    'description': 'Conspiración agenda global'
                },
                {
                    'pattern': r'\b(?:illuminati|masoner[íi]a|deep\s+state)\b',
                    'description': 'Conspiración sociedades secretas'
                },
                {
                    'pattern': r'\b(?:microchips?|controlarnos?|manipul[ao]r|vigilarnos)\b',
                    'description': 'Teorías de control tecnológico'
                },
                {
                    'pattern': r'\b(?:datos\s+oficiales\s+son\s+mentira|gobierno\s+oculta|medios\s+mainstream\s+ocultan)\b',
                    'description': 'Desconfianza en información oficial'
                },
                {
                    'pattern': r'\b(?:medios|prensa|televisión)\s+(?:ocultan|esconden|mienten|manipulan)\b.*\b(?:verdad|realidad|datos)\b|\b(?:verdad|realidad|datos)\b.*\b(?:medios|prensa|televisión)\s+(?:ocultan|esconden|mienten|manipulan)\b',
                    'description': 'Medios ocultan información'
                },
                {
                    'pattern': r'\b(?:chemtrails?|estelas\s+químicas)\b.*(?:población|control|veneno|aluminio)',
                    'description': 'Teoría de chemtrails'
                },
                {
                    'pattern': r'\b(?:industria\s+farmacéutica|big\s+pharma)\s+(?:oculta|esconde|no\s+quiere)',
                    'description': 'Conspiración farmacéutica'
                },
                {
                    'pattern': r'\b(?:medios|empresas|corporaciones)\s+(?:están\s+)?(?:comprados|compradas|controlados|controladas)\b',
                    'description': 'Medios/empresas controlados por élites'
                },
                {
                    'pattern': r'\b(?:cambio\s+climático|calentamiento\s+global)\s+(?:es\s+(?:un\s+)?(?:invento|mentira|farsa))',
                    'description': 'Negación del cambio climático'
                },
                {
                    'pattern': r'\b(?:mayor|gran|gran)\s+(?:engaño|mentira|farsa|estafa)\s+de\s+la\s+historia',
                    'description': 'Gran engaño/mentira histórica'
                },
                {
                    'pattern': r'\b(?:engaño|mentira|farsa)\s+(?:de|del)\s+(?:la\s+historia|siglo|mundo)',
                    'description': 'Engaño histórico/mundial'
                },
                {
                    'pattern': r'\b(?:científicos\s+ocultan|estudios\s+independientes|la\s+verdad\s+que\s+no\s+te\s+cuentan)',
                    'description': 'Desconfianza en ciencia oficial'
                },
                {
                    'pattern': r'\b(?:experimento\s+social|control\s+mental|población\s+mundial)',
                    'description': 'Teorías de control poblacional'
                },
                {
                    'pattern': r'\b(?:élite\s+global|reptilianos|nuevo\s+orden)\s+(?:controla|domina|manipula)',
                    'description': 'Conspiración élite global'
                },
                {
                    'pattern': r'\b(?:no\s+es\s+casualidad|no\s+es\s+coincidencia)\s+que\b.*\b(?:siempre|siempre\s+están|benefician|mismos)\b',
                    'description': 'Patrones de intencionalidad oculta'
                },
                {
                    'pattern': r'\b(?:mismos?\s+(?:grupos?|empresas?|personas?|corporaciones?))\s+(?:que\s+)?(?:se\s+)?(?:benefician|controlan|dominan)\b.*\b(?:eventos?|situaciones?|crisis)\b',
                    'description': 'Beneficiarios recurrentes sospechosos'
                },
                {
                    'pattern': r'\b(?:siempre\s+están|siempre\s+aparecen|siempre\s+salen\s+ganando)\s+en\s+(?:posiciones?\s+de\s+)?(?:control|poder|ventaja)\b',
                    'description': 'Patrones de control sistemático'
                },
                {
                    'pattern': r'\b(?:calentamiento\s+global|cambio\s+climático)\s+(?:es\s+(?:una\s+)?)?(?:mentira|farsa|engaño)',
                    'description': 'Negación del cambio climático específica'
                },
                {
                    'pattern': r'\b(?:big\s+pharma|industria\s+farmacéutica)\s+(?:lleva\s+\d+\s+años\s+)?(?:ocultando|escondiendo)\s+(?:la\s+cura)',
                    'description': 'Big Pharma oculta curas'
                }
            ],
            'violence_incitement': [
                {
                    'pattern': r'\b(?:a\s+las\s+armas|tomar\s+las\s+armas|armarse)\b',
                    'description': 'Llamada directa a violencia'
                },
                {
                    'pattern': r'\b(?:colgar|fusilar|al\s+paredón)\s+(?:a\s+)?(?:los\s+)?(?:traidores|comunistas|separatistas)\b',
                    'description': 'Amenazas de muerte específicas'
                },
                {
                    'pattern': r'\b(?:guerra|lucha\s+armada|resistencia\s+armada)\s+(?:civil|nacional)\b',
                    'description': 'Incitación a guerra civil'
                },
                {
                    'pattern': r'\b(?:exterminio|eliminar|acabar\s+con)\s+(?:los\s+)?(?:rojos|marxistas|separatistas)\b',
                    'description': 'Llamadas a exterminio'
                }
            ],
            'anti_government': [
                {
                    'pattern': r'\b(?:régimen|dictadura)\s+(?:de\s+)?(?:sánchez|socialista|comunista)\b',
                    'description': 'Gobierno como dictadura'
                },
                {
                    'pattern': r'\b(?:golpe\s+de\s+estado|derrocar|derribar)\s+(?:al\s+)?gobierno\b',
                    'description': 'Llamadas a golpe'
                },
                {
                    'pattern': r'\b(?:traidor|vendepat(?:ria|rias))\s+(?:sánchez|gobierno|socialistas?)\b',
                    'description': 'Acusaciones de traición'
                },
                {
                    'pattern': r'\b(?:resistencia|desobediencia)\s+(?:civil|nacional|patriótica)\b',
                    'description': 'Llamadas a resistencia'
                }
            ],
            'historical_revisionism': [
                {
                    'pattern': r'\b(?:franco|franquismo)\s+(?:salvador|liberador|necesario)\b',
                    'description': 'Apología del franquismo'
                },
                {
                    'pattern': r'\b(?:holocausto|shoah)\s+(?:mentira|falso|exagerado)\b',
                    'description': 'Negación del holocausto'
                },
                {
                    'pattern': r'\b(?:leyenda\s+negra|anti-españolismo)\s+(?:histórico|sistemático)\b',
                    'description': 'Revisionismo histórico español'
                },
                {
                    'pattern': r'\b(?:memoria\s+histórica)\s+(?:falsa|manipulada|sectaria)\b',
                    'description': 'Negación memoria histórica'
                }
            ],
            'health_disinformation': [
                {
                    'pattern': r'\b(?:suplemento|remedio)\s+(?:natural|milagroso)\s+(?:cura|trata)\s+(?:el\s+)?(?:cáncer|diabetes|covid)',
                    'description': 'Curas milagrosas fraudulentas'
                },
                {
                    'pattern': r'\b(?:este|mi)\s+(?:tratamiento|medicina|producto)\s+(?:cura|elimina)\s+(?:el\s+)?(?:cáncer|vih|diabetes)\s+en\s+\d+\s+días',
                    'description': 'Promesas de cura rápida'
                },
                {
                    'pattern': r'\b(?:medicina\s+alternativa|terapia\s+natural)\s+(?:que\s+)?(?:los\s+)?médicos\s+(?:no\s+quieren|ocultan|prohíben)',
                    'description': 'Medicina alternativa vs medicina oficial'
                },
                {
                    'pattern': r'\b(?:la\s+verdad\s+sobre|secreto\s+de)\s+(?:las\s+)?(?:vacunas|medicamentos|tratamientos)',
                    'description': 'Revelaciones sobre medicina'
                },
                {
                    'pattern': r'\b(?:estudios\s+(?:independientes|secretos|censurados))\s+(?:demuestran|revelan|prueban)',
                    'description': 'Estudios supuestamente censurados'
                },
                {
                    'pattern': r'\b(?:vacunas?)\s+(?:causan|provocan)\s+(?:autismo|cáncer|esterilidad)',
                    'description': 'Desinformación sobre vacunas'
                },
                {
                    'pattern': r'\b(?:vacunas?|medicamentos?)\s+.*(?:según\s+estudios\s+que|investigaciones\s+que)\s+.*(?:médicos|gobierno)\s+(?:no\s+quieren|ocultan)',
                    'description': 'Vacunas con teorías conspirativas'
                }
            ],
            'far_right_bias': [
                {
                    'pattern': r'\b(?:socialistas?|comunistas?|marxistas?|rojos?)\s+(?:han\s+)?(?:destruido|destruyen|arruinado|arruinan|convertido)\s+(?:España|el\s+país|nuestro\s+país)',
                    'description': 'Retórica anti-izquierda extrema'
                },
                {
                    'pattern': r'\b(?:feminazis?|feministas?\s+extremistas?)\b.*\b(?:destruido|destruyen|arruinado|han\s+destruido)\b|\b(?:destruido|destruyen|arruinado|han\s+destruido)\b.*\b(?:feminazis?|feministas?\s+extremistas?)\b',
                    'description': 'Retórica anti-feminista extrema'
                },
                {
                    'pattern': r'\b(?:solo|únicamente)\s+(?:vox|partido\s+patriótico)\s+(?:puede\s+)?(?:salvar|defender|proteger)\s+(?:España|la\s+patria)',
                    'description': 'Exclusivismo político extremo'
                },
                {
                    'pattern': r'\b(?:agenda|ideología)\s+(?:marxista|comunista|progre)\s+(?:está\s+)?(?:destruyendo|infectando|corrompiendo)',
                    'description': 'Demonización ideológica'
                },
                {
                    'pattern': r'\b(?:invasión|dictadura)\s+(?:comunista|socialista|progre)\s+(?:en\s+)?(?:España|nuestro\s+país)',
                    'description': 'Retórica de invasión política'
                },
                {
                    'pattern': r'\b(?:han\s+)?convertido\s+(?:España|el\s+país)\s+en\s+(?:venezuela|cuba|la\s+urss)',
                    'description': 'Comparaciones con regímenes socialistas'
                },
                {
                    'pattern': r'\b(?:traidores?|vendidos?)\s+(?:al\s+)?(?:globalismo|comunismo|soros)',
                    'description': 'Acusaciones de traición política'
                },
                {
                    'pattern': r'\b(?:régimen|dictadura)\s+(?:de\s+)?(?:sánchez|socialista|rojo)',
                    'description': 'Gobierno como dictadura'
                },
                {
                    'pattern': r'\b(?:agenda|movimiento|ideología)\s+(?:progresista|izquierdista|de\s+izquierdas?)\s+(?:está\s+)?(?:transformando|cambiando|modificando|alterando)\b',
                    'description': 'Transformación ideológica percibida como amenaza'
                },
                {
                    'pattern': r'\b(?:agenda|ideología)\s+(?:woke|progre|de\s+género)\s+(?:está\s+)?(?:destruyendo|arruinando|infectando)',
                    'description': 'Agenda woke/progresista destructiva'
                },
                {
                    'pattern': r'\b(?:woke|progre)\s+(?:está\s+)?(?:destruyendo|arruinando|infectando|atacando)',
                    'description': 'Ideología woke destructiva'
                },
                {
                    'pattern': r'\b(?:instituciones?\s+)?(?:tradicionales?|históricas?|establecidas?)\s+(?:están\s+siendo|son)\s+(?:atacadas?|destruidas?|modificadas?|transformadas?)\b',
                    'description': 'Instituciones tradicionales bajo amenaza'
                },
                {
                    'pattern': r'\b(?:feminismo|feministas?)\s+(?:es\s+)?(?:cáncer|plaga|virus|destrucción)',
                    'description': 'Feminismo como enfermedad/destrucción'
                },
                {
                    'pattern': r'\b(?:mujeres?|tías)\s+(?:están\s+para)\s+(?:fregar|parir|callarse|servir)',
                    'description': 'Roles de género tradicionales extremos'
                },
                {
                    'pattern': r'\b(?:nuestras?\s+)?(?:instituciones?|estructuras?|bases?)\s+(?:fundamentales?|básicas?|tradicionales?)\s+(?:están\s+en\s+peligro|bajo\s+ataque)\b',
                    'description': 'Instituciones fundamentales amenazadas'
                }
            ],
            'call_to_action': [
                {
                    'pattern': r'\b(?:concentración|manifestación|protesta|convocatoria)\s+(?:urgente\s+)?(?:hoy|mañana|el\s+\w+)?\s*(?:a\s+las\s+)?(?:\d{1,2}:\d{2}[hH]?|\d{1,2}[hH])?',
                    'description': 'Convocatoria específica a movilización'
                },
                {
                    'pattern': r'\btodos\s+a\s+\w+\s+(?:hoy|mañana|ahora)',
                    'description': 'Convocatoria masiva con ubicación y tiempo'
                },
                {
                    'pattern': r'\b(?:hay\s+que|tenemos\s+que|debemos)\s+(?:salir\s+a\s+las\s+calles|movilizarnos|actuar)',
                    'description': 'Llamada a movilización'
                },
                {
                    'pattern': r'\b(?:únete|participa|ven)\s+(?:a\s+la\s+)?(?:manifestación|concentración|protesta)',
                    'description': 'Invitación directa a protesta'
                },
                {
                    'pattern': r'\b(?:todos?\s+unidos?|juntos\s+podemos|la\s+unión\s+hace\s+la\s+fuerza)',
                    'description': 'Llamada a unidad y acción'
                },
                {
                    'pattern': r'\b(?:marchemos|vamos)\s+(?:todos?\s+)?(?:contra|a)\b',
                    'description': 'Llamada directa a marchar o actuar'
                },
                {
                    'pattern': r'\b(?:revolución|rebelión|alzamiento)\s+(?:ya|ahora|hoy)\b',
                    'description': 'Llamada a revolución inmediata'
                },
                {
                    'pattern': r'\b(?:defender|proteger|salvar)\s+(?:España|la\s+patria|nuestro\s+país)\s+(?:de\s+la\s+)?(?:invasión|destrucción)',
                    'description': 'Llamada a defensa nacional'
                },
                {
                    'pattern': r'\b(?:es\s+hora\s+de|momento\s+de)\s+(?:actuar|levantarse|despertar|luchar)',
                    'description': 'Urgencia de acción'
                },
                {
                    'pattern': r'\b(?:comparte|difunde|reenvía)\s+(?:para\s+que|hasta\s+que)\s+(?:todos?\s+)?(?:sepan|despierten)',
                    'description': 'Llamada a difusión'
                },
                {
                    'pattern': r'\b(?:es\s+)?(?:momento|hora|tiempo)\s+(?:de\s+)?(?:que\s+)?(?:los\s+)?(?:ciudadanos?|personas?|gente)\s+(?:responsables?|comprometidas?|conscientes?)\s+(?:tomen|hagamos?|realicemos?)\s+(?:medidas?|acciones?|algo)\b',
                    'description': 'Llamada a responsabilidad cívica activa'
                },
                {
                    'pattern': r'\b(?:tomar|adoptar|implementar)\s+(?:medidas?|acciones?|decisiones?)\s+(?:para\s+)?(?:proteger|defender|salvaguardar)\s+(?:nuestras?\s+)?(?:comunidades?|familias?|sociedad)\b',
                    'description': 'Medidas de protección comunitaria'
                },
                {
                    'pattern': r'\b(?:ciudadanos?|personas?|gente)\s+(?:responsables?|comprometidas?|conscientes?)\s+(?:deben|tienen\s+que|necesitan)\s+(?:actuar|hacer\s+algo|tomar\s+acción)\b',
                    'description': 'Deber cívico de actuar'
                },
                {
                    'pattern': r'\b(?:patriotas?|españoles?\s+de\s+bien)\s*!+\s+(?:es\s+hora\s+de|momento\s+de)\s+(?:organizarse|unirse|actuar)',
                    'description': 'Llamada a organización patriótica'
                },
                {
                    'pattern': r'\b(?:colegios?|escuelas?)\s+(?:están\s+)?(?:adoctrinando|manipulando|lavando\s+el\s+cerebro)',
                    'description': 'Adoctrinamiento en educación'
                },
                {
                    'pattern': r'\b(?:adoctrinando|adoctrinan|adoctrinar)\s+(?:a\s+)?(?:nuestros?\s+)?(?:niños?|hijos?)',
                    'description': 'Adoctrinamiento infantil'
                },
                {
                    'pattern': r'\b(?:retirad|sacad)\s+(?:a\s+)?(?:vuestros?\s+)?(?:hijos?|niños?)\s+(?:de\s+(?:los\s+)?colegios?)',
                    'description': 'Llamada a retirar hijos de colegios'
                },
                {
                    'pattern': r'\b(?:basta\s+ya|ya\s+basta)\b.*\b(?:unirse|salir\s+a\s+las\s+calles|actuar)',
                    'description': 'Llamada urgente a acción callejera'
                },
                {
                    'pattern': r'\b(?:verdaderos?\s+españoles?|patriotas?)\s+(?:debemos|tenemos\s+que)\s+(?:recuperar|defender)',
                    'description': 'Llamada a acción patriótica'
                },
                {
                    'pattern': r'\bmovilizaos\b|\bmovilízaos\b',
                    'description': 'Llamada directa a movilización'
                },
                {
                    'pattern': r'\bsacad\s+(?:a\s+)?(?:vuestros?\s+)?(?:hijos?|niños?)\s+de\s+(?:esos\s+)?(?:colegios?)',
                    'description': 'Retirar hijos de colegios específicos'
                }
            ]
        }
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text for far-right patterns and return detected categories.
        """
        if not text or len(text.strip()) < 5:
            return {
                'categories': [],
                'pattern_matches': [],
                'has_patterns': False
            }
        
        text_lower = text.lower()
        pattern_matches = []
        detected_categories = set()
        
        # Analyze each category
        for category, patterns in self.patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info['pattern']
                description = pattern_info['description']
                
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    detected_categories.add(category)
                    
                    pattern_matches.append({
                        'category': category,
                        'matched_text': match.group(),
                        'description': description,
                        'context': text[max(0, match.start()-20):match.end()+20].strip()
                    })
        
        return {
            'categories': list(detected_categories),
            'pattern_matches': pattern_matches,
            'has_patterns': len(pattern_matches) > 0,
            'total_matches': len(pattern_matches)
        }
    
    def has_far_right_content(self, text: str) -> bool:
        """Determine if content contains far-right patterns."""
        result = self.analyze_text(text)
        return result['has_patterns']
    
    def get_detected_categories(self, text: str) -> List[str]:
        """Get all detected far-right categories for the content."""
        result = self.analyze_text(text)
        return result['categories']

# Test function
if __name__ == "__main__":
    test_texts = [
        "Los musulmanes son una raza inferior que viene a destruir nuestra cultura cristiana",
        "España para los españoles, fuera los moros invasores",
        "Soros financia la inmigración para reemplazar a los europeos según el plan Kalergi",
        "¡A las armas patriotas! Es hora de la reconquista nacional",
        "El régimen de Sánchez es una dictadura comunista que hay que derrocar",
        "Franco fue el salvador de España frente al comunismo internacional",
        "Hoy hace sol, perfecto para pasear por el parque"
    ]
    
    analyzer = FarRightAnalyzer()
    
    for text in test_texts:
        print(f"\n--- Texto: {text}")
        result = analyzer.analyze_text(text)
        print(f"Tiene patrones: {result['has_patterns']}")
        print(f"Categorías: {result['categories']}")
        if result['pattern_matches']:
            print("Patrones detectados:")
            for match in result['pattern_matches'][:3]:
                print(f"  - {match['category']}: {match['description']}")
