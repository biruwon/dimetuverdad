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
                    'pattern': r'\b(?:eliminar|deportar|expulsar)\s+(?:a\s+)?(?:los\s+)?(?:musulmanes|gitanos|moros|negros)\b',
                    'description': 'Llamadas a eliminación de grupos'
                },
                {
                    'pattern': r'\b(?:virus|infectan|plaga|invasión)\s+(?:musulmán|gitana|extranjera)\b',
                    'description': 'Deshumanización de grupos'
                },
                {
                    'pattern': r'\b(?:los\s+)?(?:inmigrantes|extranjeros|musulmanes)\s+(?:son\s+)?(?:una\s+)?(?:plaga|virus|amenaza|invasión)',
                    'description': 'Deshumanización de inmigrantes'
                },
                {
                    'pattern': r'\b(?:genéticamente|biológicamente)\s+inferior(?:es)?\b',
                    'description': 'Determinismo biológico'
                }
            ],
            'xenophobia': [
                {
                    'pattern': r'\b(?:invasión|avalancha|ola)\s+(?:de\s+)?(?:inmigrantes|extranjeros)\b',
                    'description': 'Inmigración como invasión'
                },
                {
                    'pattern': r'\b(?:España|Europa)\s+para\s+(?:los\s+)?españoles?\b',
                    'description': 'Nacionalismo excluyente'
                },
                {
                    'pattern': r'\b(?:fuera|expulsar|deportar)\s+(?:a\s+)?(?:los\s+)?(?:moros|extranjeros|ilegales)\b',
                    'description': 'Llamadas a expulsión'
                },
                {
                    'pattern': r'\b(?:reemplaz|sustitu)(?:ar|ción)\s+(?:población|demográfica?)\b',
                    'description': 'Teoría del gran reemplazo'
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
                    'pattern': r'\b(?:chemtrails?|estelas\s+químicas)\b.*(?:población|control|veneno|aluminio)',
                    'description': 'Teoría de chemtrails'
                },
                {
                    'pattern': r'\b(?:industria\s+farmacéutica|big\s+pharma)\s+(?:oculta|esconde|no\s+quiere)',
                    'description': 'Conspiración farmacéutica'
                },
                {
                    'pattern': r'\b(?:cambio\s+climático|calentamiento\s+global)\s+(?:es\s+(?:un\s+)?(?:invento|mentira|farsa))',
                    'description': 'Negación del cambio climático'
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
