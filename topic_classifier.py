"""
Advanced topic classification system for Spanish political content analysis.
Includes specialized models and comprehensive labeling schemes.
"""

from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass
from enum import Enum

class TopicCategory(Enum):
    POLITICS_GENERAL = "política_general"
    EXTREMISM = "extremismo"
    IMMIGRATION = "inmigración"
    NATIONALISM = "nacionalismo"
    CONSPIRACY = "conspiración"
    VIOLENCE_THREATS = "violencia_amenazas"
    MISINFORMATION = "desinformación"
    HATE_SPEECH = "discurso_odio"
    MOBILIZATION = "movilización"
    ANTI_GOVERNMENT = "anti_gobierno"
    SOCIAL_ISSUES = "temas_sociales"
    ECONOMY = "economía"
    GENDER_LGBTQ = "género_lgbtq"
    RELIGION = "religión"
    HISTORICAL_REVISIONISM = "revisionismo_histórico"
    MEDIA_CRITICISM = "crítica_medios"
    JUDICIAL_SYSTEM = "sistema_judicial"
    SECURITY = "seguridad"
    CULTURAL_ISSUES = "temas_culturales"
    NON_POLITICAL = "no_político"

@dataclass
class TopicResult:
    category: TopicCategory
    confidence: float
    subcategory: str
    keywords: List[str]
    context_indicators: List[str]

class SpanishPoliticalTopicClassifier:
    """
    Specialized topic classifier for Spanish political content with focus on extremism detection.
    Uses rule-based classification combined with contextual analysis.
    """
    
    def __init__(self):
        self.topic_patterns = self._initialize_patterns()
        self.context_amplifiers = self._initialize_context_amplifiers()
        self.political_entities = self._initialize_political_entities()
    
    def _initialize_patterns(self) -> Dict[TopicCategory, Dict]:
        """Initialize comprehensive topic detection patterns."""
        return {
            TopicCategory.EXTREMISM: {
                'primary_patterns': [
                    r'\b(?:extrema?|ultra)(?:\s+)?(?:derecha?|izquierda?)\b',
                    r'\b(?:nazi|fascist[ao]s?|hitler|holocaust[oa])\b',
                    r'\b(?:supremacist[ao]s?|supremac[íi]a)\b',
                    r'\b(?:falange|falangist[ao]s?)\b',
                    r'\b(?:neonazi|skinhead)s?\b',
                    r'\b(?:raza\s+(?:superior|pura|aria))\b',
                    r'\b(?:limpieza\s+(?:étnica|racial))\b'
                ],
                'secondary_patterns': [
                    r'\b(?:ideolog[íi]a|doctrina)\s+(?:racial|étnica)\b',
                    r'\b(?:pureza|superioridad)\s+(?:racial|étnica)\b',
                    r'\b(?:movimiento|organización)\s+(?:nacionalist[ao]|patriót[iao])\b'
                ],
                'keywords': ['extremismo', 'radicalización', 'fanatismo', 'sectarismo'],
                'weight': 1.0
            },
            
            TopicCategory.IMMIGRATION: {
                'primary_patterns': [
                    r'\b(?:inmigr(?:ante|ación)|migr(?:ante|ación))\b',
                    r'\b(?:extranjero|foráneo)s?\b',
                    r'\b(?:refugiado|asilado)s?\b',
                    r'\b(?:frontera|control\s+fronterizo)\b',
                    r'\b(?:deportación|expulsión|repatriación)\b',
                    r'\b(?:ilegal(?:es)?|sin\s+papeles)\b',
                    r'\b(?:pateras?|cayuco)s?\b'
                ],
                'secondary_patterns': [
                    r'\b(?:integración|asimilación)\s+(?:social|cultural)\b',
                    r'\b(?:multicultur(?:al|alismo)|diversidad\s+cultural)\b',
                    r'\b(?:política\s+migratoria|ley\s+(?:de\s+)?extranjería)\b'
                ],
                'keywords': ['inmigración', 'extranjería', 'frontera', 'asilo'],
                'weight': 1.0
            },
            
            TopicCategory.NATIONALISM: {
                'primary_patterns': [
                    r'\b(?:nación|nacional|nacionalismo)\b',
                    r'\b(?:patria|patriot(?:a|ismo))\b',
                    r'\b(?:soberan[íi]a|independencia)\b',
                    r'\b(?:identidad\s+(?:nacional|española|cultural))\b',
                    r'\b(?:hispanidad|españolidad)\b',
                    r'\b(?:bandera|himno|símbolos?\s+nacional(?:es)?)\b'
                ],
                'secondary_patterns': [
                    r'\b(?:reconquista|recuperar\s+españa)\b',
                    r'\b(?:tradición|tradiciones|ancestr(?:al|os))\b',
                    r'\b(?:lengua|idioma)\s+(?:español|castellano|nacional)\b'
                ],
                'keywords': ['nacionalismo', 'patriotismo', 'identidad', 'soberanía'],
                'weight': 1.0
            },
            
            TopicCategory.CONSPIRACY: {
                'primary_patterns': [
                    r'\b(?:conspiración|complot|conjura)\b',
                    r'\b(?:élite|oligarqu[íi]a)\s+(?:global|mundial|oculta)\b',
                    r'\b(?:nuevo\s+orden\s+mundial|deep\s+state|estado\s+profundo)\b',
                    r'\b(?:soros|globalistas?|masoner[íi]a)\b',
                    r'\b(?:illuminati|bilderberg|davos)\b',
                    r'\b(?:agenda\s+(?:2030|oculta)|gran\s+reset(?:eo)?)\b',
                    r'\b(?:plandemia|dictadura\s+sanitaria)\b'
                ],
                'secondary_patterns': [
                    r'\b(?:manipulación|control)\s+(?:mental|social)\b',
                    r'\b(?:medios\s+(?:de\s+)?comunicación|prensa)\s+(?:manipulad[ao]|comprad[ao])\b',
                    r'\b(?:desinformación|propaganda|lavado\s+de\s+cerebro)\b'
                ],
                'keywords': ['conspiración', 'élite', 'manipulación', 'control'],
                'weight': 1.0
            },
            
            TopicCategory.VIOLENCE_THREATS: {
                'primary_patterns': [
                    r'\b(?:violen(?:cia|to)|agresión|ataque)\b',
                    r'\b(?:amenaza|amenazar|intimidar)\b',
                    r'\b(?:asesinar|matar|eliminar|acabar\s+con)\b',
                    r'\b(?:colgar|fusilar|paredón|guillotina)\b',
                    r'\b(?:armas?|armado|armamento)\b',
                    r'\b(?:bomba|explosivo|atentado)\b',
                    r'\b(?:guerra|conflicto\s+armado|lucha\s+armada)\b'
                ],
                'secondary_patterns': [
                    r'\b(?:justicia|venganza)\s+(?:por\s+(?:su\s+)?propia\s+mano|popular)\b',
                    r'\b(?:linchamiento|linchamient[oa])\b',
                    r'\b(?:exterminio|genocidio|holocausto)\b'
                ],
                'keywords': ['violencia', 'amenaza', 'armas', 'agresión'],
                'weight': 1.0
            },
            
            TopicCategory.MISINFORMATION: {
                'primary_patterns': [
                    r'\b(?:desinformación|fake\s+news|noticias?\s+falsas?)\b',
                    r'\b(?:bulo|rumor|mentira|falsedad)\b',
                    r'\b(?:manipulación|tergiversación|distorsión)\b',
                    r'\b(?:verificación|fact[\-\s]?check|comprobación)\b',
                    r'\b(?:desmentir|desmentido|refutar)\b'
                ],
                'secondary_patterns': [
                    r'\b(?:propaganda|adoctrinamiento|lavado\s+de\s+cerebro)\b',
                    r'\b(?:censura|silenciamiento|ocultación)\b',
                    r'\b(?:verdad\s+(?:oculta|alternativa)|realidad\s+alternativa)\b'
                ],
                'keywords': ['desinformación', 'bulo', 'mentira', 'manipulación'],
                'weight': 1.0
            },
            
            TopicCategory.HATE_SPEECH: {
                'primary_patterns': [
                    r'\b(?:odio|rencor|resentimiento)\b',
                    r'\b(?:discriminación|segregación|apartheid)\b',
                    r'\b(?:racismo|xenofobia|homofobia)\b',
                    r'\b(?:intolerancia|fanatismo|sectarismo)\b',
                    r'\b(?:insulto|vejación|humillación)\b'
                ],
                'secondary_patterns': [
                    r'\b(?:desprecio|desdén|menosprecio)\b',
                    r'\b(?:estigma|prejuicio|estereotipo)\b',
                    r'\b(?:exclusión|marginación|ostracismo)\b'
                ],
                'keywords': ['odio', 'discriminación', 'racismo', 'intolerancia'],
                'weight': 1.0
            },
            
            TopicCategory.MOBILIZATION: {
                'primary_patterns': [
                    r'\b(?:manifestación|concentración|protesta)\b',
                    r'\b(?:movilización|convocatoria|llamamiento)\b',
                    r'\b(?:marcha|desfile|procesión)\b',
                    r'\b(?:huelga|paro|boicot)\b',
                    r'\b(?:resistencia|desobediencia|insumisión)\b',
                    r'\b(?:revolución|rebelión|alzamiento)\b'
                ],
                'secondary_patterns': [
                    r'\b(?:activismo|militancia|compromiso)\b',
                    r'\b(?:organización|coordinación|planificación)\b',
                    r'\b(?:solidaridad|apoyo|respaldo)\b'
                ],
                'keywords': ['movilización', 'protesta', 'manifestación', 'activismo'],
                'weight': 1.0
            },
            
            TopicCategory.ANTI_GOVERNMENT: {
                'primary_patterns': [
                    r'\b(?:gobierno|administración|ejecutivo)\s+(?:corrupt[oa]|ilegítim[oa]|dictatorial)\b',
                    r'\b(?:régimen|dictadura|tiranía|autocracia)\b',
                    r'\b(?:traición|traidor|vendepat(?:ria|rias))\b',
                    r'\b(?:corrupción|soborno|malversación)\b',
                    r'\b(?:dimisión|destitución|cese)\b'
                ],
                'secondary_patterns': [
                    r'\b(?:oposición|resistencia|contestación)\b',
                    r'\b(?:democracia|libertad|derechos)\s+(?:amenazad[ao]|en\s+peligro)\b',
                    r'\b(?:autoritarismo|totalitarismo|despotismo)\b'
                ],
                'keywords': ['gobierno', 'régimen', 'corrupción', 'oposición'],
                'weight': 1.0
            },
            
            TopicCategory.GENDER_LGBTQ: {
                'primary_patterns': [
                    r'\b(?:género|sexo|identidad\s+sexual)\b',
                    r'\b(?:lgbti?q?\+?|homosexual|gay|lesbiana|trans(?:género|sexual)?)\b',
                    r'\b(?:matrimonio\s+(?:igualitario|homosexual|gay))\b',
                    r'\b(?:ideología\s+de\s+género|lobby\s+gay)\b',
                    r'\b(?:feminismo|machismo|patriarcado)\b'
                ],
                'secondary_patterns': [
                    r'\b(?:igualdad|equidad|paridad)\s+(?:de\s+género|sexual)\b',
                    r'\b(?:derechos\s+(?:lgbti?|reproductivos))\b',
                    r'\b(?:violencia\s+(?:de\s+género|machista|doméstica))\b'
                ],
                'keywords': ['género', 'lgbtq', 'feminismo', 'igualdad'],
                'weight': 1.0
            }
        }
    
    def _initialize_context_amplifiers(self) -> List[Tuple[str, float]]:
        """Initialize context patterns that amplify topic confidence."""
        return [
            (r'\b(?:urgente|importante|crucial|vital)\b', 1.2),
            (r'\b(?:todos?|todas?|nadie|ningún)\b', 1.1),
            (r'[!]{2,}', 1.15),
            (r'[🔴⚠️🚨💥]', 1.1),
            (r'\b(?:siempre|nunca|jamás)\b', 1.1),
            (r'\b(?:debe(?:mos|n)?|hay\s+que|es\s+necesario)\b', 1.1)
        ]
    
    def _initialize_political_entities(self) -> Dict[str, List[str]]:
        """Initialize political entities and figures for context detection."""
        return {
            'parties': [
                'psoe', 'pp', 'vox', 'podemos', 'ciudadanos', 'cs',
                'erc', 'junts', 'pnv', 'bildu', 'más país', 'compromís'
            ],
            'leaders': [
                'sánchez', 'feijóo', 'abascal', 'iglesias', 'arrimadas',
                'rufián', 'puigdemont', 'otegi', 'urkullu', 'ayuso',
                'moreno bonilla', 'aragonès', 'puig'
            ],
            'institutions': [
                'congreso', 'senado', 'gobierno', 'ministerio', 'tribunal constitucional',
                'tribunal supremo', 'audiencia nacional', 'fiscalía', 'guardia civil',
                'policía nacional', 'mossos', 'ertzaintza'
            ]
        }
    
    def classify_topic(self, text: str) -> List[TopicResult]:
        """
        Classify text into political topics with confidence scores.
        Returns list of relevant topics sorted by confidence.
        """
        if not text or len(text.strip()) < 5:
            return [TopicResult(
                category=TopicCategory.NON_POLITICAL,
                confidence=0.0,
                subcategory="texto_vacio",
                keywords=[],
                context_indicators=[]
            )]
        
        text_lower = text.lower()
        results = []
        
        # Check each topic category
        for category, patterns in self.topic_patterns.items():
            confidence = 0.0
            matched_keywords = []
            context_indicators = []
            
            # Primary patterns (higher weight)
            for pattern in patterns['primary_patterns']:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                if matches > 0:
                    confidence += matches * 0.4
                    matched_keywords.extend(re.findall(pattern, text_lower, re.IGNORECASE))
            
            # Secondary patterns (lower weight)
            for pattern in patterns['secondary_patterns']:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                if matches > 0:
                    confidence += matches * 0.2
                    matched_keywords.extend(re.findall(pattern, text_lower, re.IGNORECASE))
            
            # Apply context amplifiers
            for amplifier_pattern, multiplier in self.context_amplifiers:
                if re.search(amplifier_pattern, text_lower, re.IGNORECASE):
                    confidence *= multiplier
                    context_indicators.append(amplifier_pattern)
            
            # Check for political entity mentions
            political_context = self._detect_political_context(text_lower)
            if political_context:
                confidence *= 1.1
                context_indicators.extend(political_context)
            
            # Normalize confidence (cap at 1.0)
            confidence = min(1.0, confidence)
            
            if confidence > 0.1:  # Only include significant matches
                results.append(TopicResult(
                    category=category,
                    confidence=round(confidence, 3),
                    subcategory=self._determine_subcategory(category, matched_keywords),
                    keywords=list(set(matched_keywords))[:10],  # Limit and deduplicate
                    context_indicators=context_indicators
                ))
        
        # Sort by confidence and return top results
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        # If no political topics detected, classify as non-political
        if not results:
            results.append(TopicResult(
                category=TopicCategory.NON_POLITICAL,
                confidence=0.9,
                subcategory="general",
                keywords=[],
                context_indicators=[]
            ))
        
        return results[:5]  # Return top 5 most relevant topics
    
    def _detect_political_context(self, text: str) -> List[str]:
        """Detect mentions of political entities to provide context."""
        context = []
        
        for entity_type, entities in self.political_entities.items():
            for entity in entities:
                if entity in text:
                    context.append(f"{entity_type}:{entity}")
        
        return context
    
    def _determine_subcategory(self, category: TopicCategory, keywords: List[str]) -> str:
        """Determine subcategory based on matched keywords."""
        subcategories = {
            TopicCategory.EXTREMISM: {
                'nazi|fascist': 'fascismo',
                'supremac': 'supremacismo',
                'falange': 'falangismo',
                'raza': 'racismo'
            },
            TopicCategory.IMMIGRATION: {
                'refugiado|asilo': 'refugiados',
                'ilegal|sin_papeles': 'inmigración_irregular',
                'frontera|control': 'control_fronterizo',
                'deportación|expulsión': 'deportación'
            },
            TopicCategory.NATIONALISM: {
                'reconquista': 'nacionalismo_agresivo',
                'identidad': 'identidad_nacional',
                'soberanía': 'soberanismo',
                'tradición': 'tradicionalismo'
            },
            TopicCategory.CONSPIRACY: {
                'soros|élite': 'teorías_élite',
                'plandemia': 'conspiración_covid',
                'agenda_2030': 'conspiración_global',
                'deep_state': 'estado_profundo'
            }
        }
        
        if category in subcategories:
            keywords_text = ' '.join(keywords).lower()
            for pattern, subcat in subcategories[category].items():
                if re.search(pattern, keywords_text):
                    return subcat
        
        return 'general'
    
    def get_primary_topic(self, text: str) -> TopicResult:
        """Get the single most relevant topic for the text."""
        results = self.classify_topic(text)
        return results[0] if results else TopicResult(
            category=TopicCategory.NON_POLITICAL,
            confidence=0.0,
            subcategory="unknown",
            keywords=[],
            context_indicators=[]
        )
    
    def is_political_content(self, text: str, threshold: float = 0.3) -> bool:
        """Determine if content is political based on confidence threshold."""
        primary_topic = self.get_primary_topic(text)
        return (primary_topic.category != TopicCategory.NON_POLITICAL and 
                primary_topic.confidence >= threshold)

# Convenience function for quick classification
def classify_spanish_political_content(text: str) -> List[TopicResult]:
    """Quick classification function."""
    classifier = SpanishPoliticalTopicClassifier()
    return classifier.classify_topic(text)

# Test the classifier
if __name__ == "__main__":
    test_texts = [
        "Los inmigrantes ilegales están invadiendo España, hay que defender nuestras fronteras",
        "Sánchez es un traidor que está vendiendo España a Soros y la élite globalista",
        "Manifestación el domingo contra la dictadura sanitaria y la agenda 2030",
        "Hoy hace sol en Madrid, perfecto para pasear",
        "La ideología de género está destruyendo a nuestros niños",
        "¡A las armas españoles! Es hora de la reconquista y la revolución nacional"
    ]
    
    classifier = SpanishPoliticalTopicClassifier()
    
    for text in test_texts:
        print(f"\n--- Texto: {text}")
        results = classifier.classify_topic(text)
        for i, result in enumerate(results[:3]):  # Show top 3 results
            print(f"{i+1}. {result.category.value}: {result.confidence:.3f}")
            print(f"   Subcategoría: {result.subcategory}")
            if result.keywords:
                print(f"   Palabras clave: {', '.join(result.keywords[:5])}")
