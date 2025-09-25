"""
Unified Pattern Analyzer: Merged topic classification and far-right detection.
Eliminates redundancy between SpanishPoliticalTopicClassifier and FarRightAnalyzer.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from categories import Categories



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
                    r'\b(?:eliminar|deportar|expulsar)\s+.*\b(?:musulmanes?|gitanos?|moros?|negros?)\b',
                    r'\b(?:musulmanes?|gitanos?|moros?|negros?)\b.*\b(?:eliminar|deportar|expulsar)\b',
                    # Dehumanizing language
                    r'\b(?:alimañas?|parásitos?|escoria|basura)\s+(?:musulman|gitana|mora|negra)',
                    r'\b(?:invasión|plaga|epidemia)\s+(?:de\s+)?(?:musulmanes|moros|gitanos)',
                    # Direct hate speech
                    r'\b(?:odio|rencor|resentimiento)\b.*\b(?:musulmanes|inmigrantes|gitanos)',
                    r'\b(?:discriminación|segregación|apartheid)\b',
                    # Homophobic and sexist slurs
                    r'\b(?:maricas?|maricones?)\b',
                    r'\b(?:feminazis?)\b',
                    # MERGED: Xenophobic patterns (formerly separate category)
                    r'\bfuera\s+(?:los?\s+|las?\s+)?(?:moros?|extranjeros?|ilegales?|inmigrantes?|menas?)\b',
                    r'\b(?:moros?|extranjeros?|ilegales?|inmigrantes?|menas?)\s+fuera\s+(?:de\s+)?(?:España|Europa)\b',
                    r'\b(?:expuls\w+|deport\w+)\s+(?:a\s+)?(?:los?\s+)?(?:moros?|extranjeros?|ilegales?)\b',
                    r'\b(?:moros?|musulmanes?|inmigrantes?)\s+(?:nos\s+)?(?:están\s+)?(?:invadiendo|invaden)\b',
                    r'\b(?:invasión|oleada|avalancha)\s+(?:de\s+)?(?:inmigrantes?|musulmanes?|moros?)\b',
                    r'\b(?:están\s+)?invadiendo\s+(?:España|Europa|nuestro\s+país)\b',
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
                    # General disinformation
                    r'\b(?:desinformación|fake\s+news|noticias?\s+falsas?)\b',
                    r'\b(?:bulo|rumor|mentira|falsedad)\b',
                    r'\b(?:manipulación|tergiversación|distorsión)\b',
                    # MERGED: Health disinformation patterns (formerly separate category)
                    r'\b(?:vacunas?|vacunación)\s+(?:con\s+)?(?:microchips?|chips?|5g|bill\s+gates|control\s+mental)',
                    r'\b(?:vacunas?)\s+(?:tienen|contienen|llevan)\s+(?:microchips?|chips?|nanobots?)',
                    r'\b(?:bill\s+gates)\s+(?:y\s+las\s+)?(?:vacunas?|control)',
                    r'\b(?:covid|coronavirus)\s+(?:es\s+)?(?:mentira|falso|inexistente|inventado)',
                    r'\b(?:plandemia|dictadura\s+sanitaria|farmafia)\b',
                    # Specific patterns found in failing tests
                    r'\b(?:grafeno)\b.*\b(?:controlarnos|control)\b',
                    r'\b(?:5g)\b.*\b(?:controlarnos|control)\b',
                    r'\b(?:vacunas?).*\b(?:grafeno|5g)\b',
                    r'\b(?:\d+\s+de\s+cada\s+\d+)\s+(?:casos?)\s+(?:de\s+covid|son\s+inventados?)\b',
                    # INTEGRATED CLAIM PATTERNS - Statistical claims that might be false
                    r'\b(\d+(?:[.,]\d+)*)\s*%\s+(?:de\s+)?(?:los\s+)?(\w+)',
                    r'\b(?:según|conforme\s+a|de\s+acuerdo\s+con)\s+(?:el\s+)?(\w+),?\s+(\d+(?:[.,]\d+)*)',
                    r'\b(\d+(?:[.,]\d+)*)\s+(?:millones?|miles?|euros?|personas?|casos?)\s+(?:de\s+)?(\w+)',
                    r'\b(?:aumentó|disminuyó|creció|bajó)\s+(?:un\s+)?(\d+(?:[.,]\d+)*)\s*%',
                    # Medical claims that could be false
                    r'\b(?:vacunas?|medicamentos?|tratamientos?)\s+(?:causan?|provocan?|generan?)\s+(\w+)',
                    r'\b(?:covid|coronavirus|pandemia)\s+(?:es|fue|será)\s+(\w+)',
                    r'\b(?:estudios?|investigación|ciencia)\s+(?:demuestra|prueba|confirma)\s+que\s+(.+)',
                    r'\b(?:efectos?\s+(?:secundarios?|adversos?))\s+(?:de\s+)?(.+)',
                    # Economic disinformation
                    r'\b(?:pib|inflación|desempleo|paro)\s+(?:es|está|alcanza)\s+(?:del?\s+)?(\d+(?:[.,]\d+)*)\s*%',
                    r'\b(?:salario|sueldo|pensión)\s+(?:medio|promedio)\s+(?:es|está|alcanza)\s+(\d+(?:[.,]\d+)*)\s*euros?',
                    # Scientific claims that could be false
                    r'\b(?:la\s+ciencia|científicos?|investigadores?)\s+(?:dice|afirma|demuestra)\s+que\s+(.+)',
                    r'\b(?:está\s+(?:científicamente\s+)?(?:probado|demostrado))\s+que\s+(.+)',
                    r'\b(?:cambio\s+climático|calentamiento\s+global)\s+(?:es|no\s+es)\s+(.+)',
                    # Demographic claims that might be inflated or false
                    r'\b(?:inmigrantes?|extranjeros?)\s+(?:representan|son)\s+(?:el\s+)?(\d+(?:[.,]\d+)*)\s*%',
                    r'\b(?:población|habitantes)\s+(?:de\s+)?(\w+)\s+(?:es|son|alcanza)\s+(\d+(?:[.,]\d+)*)',
                    # Additional patterns from former secondary
                    r'\b(?:propaganda|adoctrinamiento|lavado\s+de\s+cerebro)\b',
                    r'\b(?:censura|silenciamiento|ocultación)\b',
                    r'\b(?:verdad\s+(?:oculta|alternativa)|realidad\s+alternativa)\b',
                ],
                'keywords': ['desinformación', 'bulo', 'mentira', 'manipulación', 'vacunas', 'covid', 'estadísticas', 'estudios'],
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
            
            Categories.FAR_RIGHT_BIAS: {
                'patterns': [
                    # Anti-left rhetoric
                    r'\b(?:socialistas?|comunistas?|marxistas?|rojos?)\s+(?:han\s+)?(?:destruido|destruyen|arruinado|arruinan)\s+(?:España|el\s+país)',
                    r'\b(?:régimen|dictadura)\s+(?:de\s+)?(?:sánchez|socialista|rojo)\b',
                    r'\b(?:agenda|ideología)\s+(?:marxista|comunista|progre)\s+(?:está\s+)?(?:destruyendo|infectando)',
                    # Political extremism
                    r'\b(?:traidores?|vendidos?)\s+(?:al\s+)?(?:globalismo|comunismo|soros)\b',
                    r'\b(?:han\s+)?convertido\s+(?:España|el\s+país)\s+en\s+(?:venezuela|cuba|la\s+urss)\b',
                    # Anti-woke and cultural war patterns
                    r'\b(?:agenda\s+woke|ideología\s+woke)\b.*\b(?:destruyendo|destruye)\b',
                    r'\b(?:woke)\b.*\b(?:valores\s+cristianos|tradiciones|familia)\b',
                    r'\b(?:feminazis?|progres?|rojos?)\b.*\b(?:destruir|acabar|eliminar)\b',
                    # Additional patterns from former secondary
                    r'\b(?:ideolog[íi]a|doctrina)\s+(?:racial|étnica)\b',
                    r'\b(?:movimiento|organización)\s+(?:nacionalist[ao]|patriót[iao])\b',
                ],
                'keywords': ['extrema_derecha', 'nacionalismo', 'patriotismo'],
                'description': 'Far-right political bias and extremist rhetoric'
            },
            
            Categories.CALL_TO_ACTION: {
                'patterns': [
                    # Direct mobilization calls
                    r'\b(?:movilizaos|organizaos|retirad|sacad|actuad\s+ya)\b',
                    r'\b(?:todos\s+a|mañana|convocatoria|difunde)\b',
                    r'\b(?:concentración|manifestación|protesta|marcha)\s+(?:el\s+)?\w+',
                    r'\b(?:boicot|boicotear|boicoteemos)\b',
                    # Action language
                    r'\b(?:revolución|rebelión|alzamiento|resistencia)\b',
                    r'\b(?:a\s+las\s+calles|hay\s+que\s+actuar|salir\s+a\s+protestar)\b',
                    # Additional patterns from former secondary
                    r'\b(?:activismo|militancia|compromiso)\b',
                    r'\b(?:organización|coordinación|planificación)\b',
                    r'\b(?:solidaridad|apoyo|respaldo)\b',
                ],
                'keywords': ['movilización', 'protesta', 'manifestación', 'boicot'],
                'description': 'Calls to action and mobilization'
            },
            
            'nationalism': {
                'patterns': [
                    # Spanish nationalism
                    r'\b(?:España|Europa)\s+para\s+(?:los\s+)?españoles?\b',
                    r'\b(?:patria|patriot(?:a|ismo)|nación|nacional)\b',
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
            
            'anti_government': {
                'patterns': [
                    # Government as illegitimate
                    r'\b(?:régimen|dictadura|tiranía)\s+(?:de\s+)?(?:sánchez|socialista)\b',
                    r'\b(?:gobierno|administración)\s+(?:corrupt[oa]|ilegítim[oa]|dictatorial)\b',
                    r'\b(?:golpe\s+de\s+estado|derrocar|derribar)\s+(?:al\s+)?gobierno\b',
                    # Anti-state rhetoric
                    r'\b(?:estado\s+profundo|deep\s+state)\b',
                    r'\b(?:traición|traidor|vendepat(?:ria|rias))\b',
                    # Additional patterns from former secondary
                    r'\b(?:oposición|resistencia|contestación)\b',
                    r'\b(?:democracia|libertad|derechos)\s+(?:amenazad[ao]|en\s+peligro)\b',
                ],
                'keywords': ['régimen', 'dictadura', 'gobierno', 'traición'],
                'description': 'Anti-government rhetoric and delegitimization'
            },
            
            'historical_revisionism': {
                'patterns': [
                    # Franco rehabilitation
                    r'\b(?:franco|franquismo)\s+(?:fue\s+)?(?:necesario|salvador|héroe|gran\s+líder|mejor)\b',
                    r'\b(?:dictadura|régimen)\s+(?:franquista|de\s+franco)\s+(?:fue\s+)?(?:mejor|próspera|gloriosa)\b',
                    r'\b(?:valle\s+de\s+los\s+caídos|fundación\s+franco)\b',
                    r'\b(?:con\s+)?franco\s+(?:se\s+)?(?:vivía|estaba|iba)\s+mejor\b',
                    # Historical denial
                    r'\b(?:república|guerra\s+civil)\s+(?:fue\s+)?(?:criminal|marxista|comunista)\b',
                    r'\b(?:víctimas\s+(?:del\s+)?franquismo)\s+(?:son\s+)?(?:mentira|exageradas?)\b',
                    # Additional patterns from former secondary
                    r'\b(?:memoria\s+histórica)\s+(?:es\s+)?(?:sectaria|revanchista)\b',
                    r'\b(?:leyenda\s+negra|historia\s+manipulada)\b',
                ],
                'keywords': ['franco', 'franquismo', 'dictadura', 'memoria'],
                'description': 'Historical revisionism and dictatorship rehabilitation'
            },
            
            'political_general': {
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
            primary_category = "political_general"
            detected_categories = ["political_general"]
        
        return AnalysisResult(
            categories=detected_categories,
            pattern_matches=pattern_matches,
            primary_category=primary_category,
            political_context=political_context,
            keywords=list(all_keywords)
        )
    
    def _detect_political_context(self, text: str) -> List[str]:
        """Detect political entities and context."""
        import re
        context = []
        text_lower = text.lower()
        
        for entity_type, entities in self.political_entities.items():
            for entity in entities:
                # Use word boundaries to avoid false positives like "que" matching "ue"
                pattern = r'\b' + re.escape(entity) + r'\b'
                if re.search(pattern, text_lower):
                    context.append(f"{entity_type}:{entity}")
        
        return context[:10]  # Limit context items


# Convenience functions for backward compatibility
def analyze_far_right_content(text: str) -> Dict:
    """Backward compatibility function for far-right analysis."""
    analyzer = PatternAnalyzer()
    result = analyzer.analyze_content(text)
    
    return {
        'categories': result.categories,
        'pattern_matches': [
            {
                'category': m.category,
                'matched_text': m.matched_text,
                'description': m.description,
                'context': m.context
            }
            for m in result.pattern_matches
        ],
        'has_patterns': len(result.pattern_matches) > 0
    }

def classify_political_topic(text: str) -> Dict:
    """Backward compatibility function for topic classification."""
    analyzer = PatternAnalyzer()
    result = analyzer.analyze_content(text)
    
    return {
        'primary_category': result.primary_category,
        'all_categories': result.categories,
        'political_context': result.political_context,
        'keywords': result.keywords
    }


# Test the unified analyzer
if __name__ == "__main__":
    test_texts = [
        "Los musulmanes son una raza inferior que viene a destruir nuestra cultura",
        "España para los españoles, fuera los moros invasores",
        "Soros financia la inmigración para reemplazar a los europeos según el plan Kalergi",
        "¡Movilizaos patriotas! Es hora de la reconquista nacional",
        "El régimen de Sánchez es una dictadura comunista que hay que derrocar",
        "Las vacunas tienen microchips de Bill Gates para controlarnos",
        "Hoy hace sol en Madrid, perfecto para pasear",
        "Manifestación mañana contra la dictadura sanitaria"
    ]
    
    analyzer = PatternAnalyzer()
    
    for text in test_texts:
        print(f"\n--- Texto: {text}")
        result = analyzer.analyze_content(text)
        print(f"Categoría principal: {result.primary_category}")
        print(f"Todas las categorías: {result.categories}")
        if result.pattern_matches:
            print("Patrones detectados:")
            for match in result.pattern_matches[:2]:
                print(f"  - {match.category}: {match.matched_text}")