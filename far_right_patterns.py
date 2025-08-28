"""
Comprehensive patterns and motifs for detecting far-right activism in Spanish social networks.
This module contains advanced pattern detection, scoring algorithms, and contextual analysis.
"""

import re
import json
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum

class ThreatLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class PatternMatch:
    pattern: str
    category: str
    weight: float
    threat_level: ThreatLevel
    context: str = ""

# Comprehensive far-right motifs categorized by theme and severity
FAR_RIGHT_PATTERNS = {
    # Immigration and xenophobia (high weight due to central role in Spanish far-right discourse)
    'immigration_xenophobia': {
        'weight_multiplier': 2.0,
        'patterns': [
            # Direct terms
            ('inmigr(?:ante)?s?', 'immigration', 1.5, ThreatLevel.MEDIUM),
            ('migr(?:ación|antes)', 'immigration', 1.5, ThreatLevel.MEDIUM),
            ('extranjeros?', 'immigration', 1.0, ThreatLevel.LOW),
            ('refugiados?', 'immigration', 1.0, ThreatLevel.LOW),
            
            # Derogatory/dehumanizing terms
            ('ilegales?', 'dehumanization', 2.5, ThreatLevel.HIGH),
            ('invasor(?:es)?', 'invasion_rhetoric', 3.0, ThreatLevel.HIGH),
            ('invasión', 'invasion_rhetoric', 3.0, ThreatLevel.HIGH),
            ('sustitución', 'replacement_theory', 3.5, ThreatLevel.CRITICAL),
            ('reemplaz[ao]', 'replacement_theory', 3.5, ThreatLevel.CRITICAL),
            ('gran reemplaz[ao]', 'replacement_theory', 4.0, ThreatLevel.CRITICAL),
            ('tsunami migrator', 'invasion_rhetoric', 3.0, ThreatLevel.HIGH),
            ('avalancha', 'invasion_rhetoric', 2.5, ThreatLevel.HIGH),
            
            # Racialized language
            ('moro[as]?', 'racial_slurs', 3.5, ThreatLevel.CRITICAL),
            ('subsahariano[as]?', 'racialization', 2.0, ThreatLevel.MEDIUM),
            ('magreb[íi](?:es|s)?', 'racialization', 1.5, ThreatLevel.MEDIUM),
        ]
    },
    
    # Islamophobia (particularly relevant in Spanish context)
    'islamophobia': {
        'weight_multiplier': 2.0,
        'patterns': [
            ('islamiz[aá]r?', 'islamophobia', 3.0, ThreatLevel.HIGH),
            ('islamización', 'islamophobia', 3.0, ThreatLevel.HIGH),
            ('yihad(?:ismo|ista)?', 'religious_extremism', 2.5, ThreatLevel.HIGH),
            ('sharia', 'religious_extremism', 2.5, ThreatLevel.HIGH),
            ('califato', 'religious_extremism', 2.5, ThreatLevel.HIGH),
            ('muzulmanes?', 'religious_targeting', 1.0, ThreatLevel.LOW),
            ('islam(?:ico[as]?)?', 'religious_targeting', 0.5, ThreatLevel.LOW),
        ]
    },
    
    # Conspiracy theories and misinformation
    'conspiracy_theories': {
        'weight_multiplier': 1.8,
        'patterns': [
            ('soros', 'antisemitic_conspiracy', 3.0, ThreatLevel.HIGH),
            ('globalist[as]?', 'globalist_conspiracy', 2.5, ThreatLevel.HIGH),
            ('élite[s]? global(?:es|ista)?', 'globalist_conspiracy', 2.5, ThreatLevel.HIGH),
            ('nuevo orden mundial', 'conspiracy', 3.0, ThreatLevel.HIGH),
            ('deep state', 'conspiracy', 2.5, ThreatLevel.HIGH),
            ('estado profundo', 'conspiracy', 2.5, ThreatLevel.HIGH),
            ('agenda 2030', 'conspiracy', 2.0, ThreatLevel.MEDIUM),
            ('gran reseteo?', 'conspiracy', 2.5, ThreatLevel.HIGH),
            ('plandemia', 'covid_conspiracy', 2.5, ThreatLevel.HIGH),
            ('dictadura sanitaria', 'covid_conspiracy', 2.5, ThreatLevel.HIGH),
            ('microchips?', 'tech_conspiracy', 2.0, ThreatLevel.MEDIUM),
            ('control mental', 'conspiracy', 2.0, ThreatLevel.MEDIUM),
            ('genocidio blanco', 'white_genocide', 4.0, ThreatLevel.CRITICAL),
        ]
    },
    
    # Spanish nationalism and identity
    'nationalism_identity': {
        'weight_multiplier': 1.5,
        'patterns': [
            ('españa(?:les)? primero', 'nationalism', 2.5, ThreatLevel.HIGH),
            ('patria', 'nationalism', 1.0, ThreatLevel.LOW),
            ('nación española', 'nationalism', 1.0, ThreatLevel.LOW),
            ('hispanidad', 'cultural_nationalism', 1.5, ThreatLevel.MEDIUM),
            ('raza española', 'racial_nationalism', 3.5, ThreatLevel.CRITICAL),
            ('pureza racial', 'racial_ideology', 4.0, ThreatLevel.CRITICAL),
            ('sangre española', 'racial_nationalism', 3.0, ThreatLevel.HIGH),
            ('tradición', 'traditionalism', 0.5, ThreatLevel.LOW),
            ('tradiciones', 'traditionalism', 0.5, ThreatLevel.LOW),
            ('identidad nacional', 'nationalism', 1.5, ThreatLevel.MEDIUM),
            ('reconquista', 'historical_nationalism', 2.5, ThreatLevel.HIGH),
        ]
    },
    
    # Anti-government and anti-establishment
    'anti_government': {
        'weight_multiplier': 1.3,
        'patterns': [
            ('sánchez dictador', 'anti_government', 2.0, ThreatLevel.MEDIUM),
            ('dictadura socialista', 'anti_government', 2.5, ThreatLevel.HIGH),
            ('régimen', 'anti_government', 1.5, ThreatLevel.MEDIUM),
            ('tiranía', 'anti_government', 2.0, ThreatLevel.MEDIUM),
            ('traidor(?:es)?', 'betrayal_rhetoric', 2.5, ThreatLevel.HIGH),
            ('traición', 'betrayal_rhetoric', 2.5, ThreatLevel.HIGH),
            ('vendepat(?:ria|rias)', 'betrayal_rhetoric', 3.0, ThreatLevel.HIGH),
            ('antiespa[ñn]ol', 'anti_spanish', 2.5, ThreatLevel.HIGH),
            ('enemigo[as]? (?:de|del) pueblo', 'populist_rhetoric', 2.0, ThreatLevel.MEDIUM),
            ('casta política', 'anti_establishment', 1.5, ThreatLevel.MEDIUM),
        ]
    },
    
    # Violence and threats
    'violence_threats': {
        'weight_multiplier': 3.0,  # Highest multiplier for safety
        'patterns': [
            ('colgar(?:los|las)?', 'violence_threat', 4.0, ThreatLevel.CRITICAL),
            ('fusilamiento', 'violence_threat', 4.0, ThreatLevel.CRITICAL),
            ('paredón', 'violence_threat', 4.0, ThreatLevel.CRITICAL),
            ('a la cárcel', 'imprisonment_call', 2.0, ThreatLevel.MEDIUM),
            ('guillotina', 'violence_threat', 4.0, ThreatLevel.CRITICAL),
            ('acabar con', 'elimination_rhetoric', 3.0, ThreatLevel.HIGH),
            ('extermini[ao]', 'genocide_rhetoric', 4.5, ThreatLevel.CRITICAL),
            ('limpieza', 'ethnic_cleansing', 4.0, ThreatLevel.CRITICAL),
            ('depuración', 'purge_rhetoric', 3.5, ThreatLevel.CRITICAL),
        ]
    },
    
    # Calls to action and mobilization
    'calls_to_action': {
        'weight_multiplier': 2.5,
        'patterns': [
            ('a las armas', 'armed_action', 4.5, ThreatLevel.CRITICAL),
            ('revolución', 'revolutionary_call', 3.0, ThreatLevel.HIGH),
            ('levantamiento', 'uprising_call', 3.5, ThreatLevel.CRITICAL),
            ('insurr?ección', 'insurrection_call', 4.0, ThreatLevel.CRITICAL),
            ('tomar las calles', 'street_action', 2.5, ThreatLevel.HIGH),
            ('recuperar españa', 'nationalist_action', 2.0, ThreatLevel.MEDIUM),
            ('defender la patria', 'defensive_nationalism', 1.5, ThreatLevel.MEDIUM),
            ('resistencia', 'resistance_call', 2.0, ThreatLevel.MEDIUM),
            ('concentr(?:ación|emonos)', 'mobilization', 1.0, ThreatLevel.LOW),
            ('manifestación', 'protest_call', 0.5, ThreatLevel.LOW),
            ('¡despertad!', 'awakening_call', 1.5, ThreatLevel.MEDIUM),
            ('españa despierta', 'awakening_call', 1.5, ThreatLevel.MEDIUM),
        ]
    },
    
    # Historical references and symbols
    'historical_symbols': {
        'weight_multiplier': 1.8,
        'patterns': [
            ('falange', 'fascist_symbol', 3.5, ThreatLevel.CRITICAL),
            ('yugo y flechas', 'fascist_symbol', 3.5, ThreatLevel.CRITICAL),
            ('cara al sol', 'fascist_symbol', 3.5, ThreatLevel.CRITICAL),
            ('arriba españa', 'fascist_slogan', 3.0, ThreatLevel.HIGH),
            ('una grande y libre', 'francoist_slogan', 2.5, ThreatLevel.HIGH),
            ('18 de julio', 'coup_commemoration', 3.0, ThreatLevel.HIGH),
            ('20-n', 'franco_death', 2.0, ThreatLevel.MEDIUM),
            ('división azul', 'nazi_collaboration', 3.5, ThreatLevel.CRITICAL),
            ('cruzada', 'religious_war', 2.5, ThreatLevel.HIGH),
        ]
    },
    
    # Gender and LGBTQ+ issues
    'gender_lgbtq': {
        'weight_multiplier': 1.5,
        'patterns': [
            ('ideología de género', 'anti_gender', 2.0, ThreatLevel.MEDIUM),
            ('lobby[s]? (?:lgtb|gay)', 'anti_lgbtq', 2.0, ThreatLevel.MEDIUM),
            ('degener(?:ado[as]?|ación)', 'derogatory_gender', 2.5, ThreatLevel.HIGH),
            ('familia tradicional', 'traditional_family', 1.0, ThreatLevel.LOW),
            ('matrimonio natural', 'anti_marriage_equality', 1.5, ThreatLevel.MEDIUM),
            ('adoctrinamiento', 'indoctrination_claim', 1.5, ThreatLevel.MEDIUM),
        ]
    },
    
    # Economic nationalism
    'economic_nationalism': {
        'weight_multiplier': 1.2,
        'patterns': [
            ('españa para los españoles', 'economic_nationalism', 2.5, ThreatLevel.HIGH),
            ('primero los nuestros', 'preference_rhetoric', 2.0, ThreatLevel.MEDIUM),
            ('trabajo para españoles', 'job_nationalism', 2.0, ThreatLevel.MEDIUM),
            ('subsidios? a extranjeros', 'welfare_chauvinism', 1.5, ThreatLevel.MEDIUM),
        ]
    }
}

# Hate speech indicators (slurs and derogatory terms)
HATE_SPEECH_PATTERNS = [
    ('rojos?', 'political_slur', 1.5, ThreatLevel.MEDIUM),
    ('zurd[oa]s?', 'political_slur', 2.0, ThreatLevel.MEDIUM),
    ('commies?', 'political_slur', 1.5, ThreatLevel.MEDIUM),
    ('bolivariano[as]?', 'political_slur', 1.5, ThreatLevel.MEDIUM),
    ('progre[s]?', 'political_slur', 1.0, ThreatLevel.LOW),
    ('buenista[s]?', 'political_slur', 1.0, ThreatLevel.LOW),
    ('perroflauta[s]?', 'social_slur', 2.0, ThreatLevel.MEDIUM),
    ('okupa[s]?', 'social_slur', 1.5, ThreatLevel.MEDIUM),
]

# Contextual amplifiers - phrases that increase severity when combined with other patterns
CONTEXTUAL_AMPLIFIERS = [
    (r'\b(?:ya|es hora de|basta de|no más)\b', 1.3),  # Urgency markers
    (r'\b(?:todos?|todas?)\b.*(?:fuera|expuls)', 1.5),  # Collective expulsion
    (r'[!]{2,}', 1.2),  # Multiple exclamation marks
    (r'[🔴⚠️🚨]', 1.2),  # Alert emojis
    (r'\b(?:URGENT[E]?|ÚLTIMA HORA|BREAKING)\b', 1.3),  # Urgency keywords
    (r'#[A-Z]+', 1.1),  # Hashtag activism
]

# Minimizing patterns - contexts that might reduce the score
MINIMIZING_CONTEXTS = [
    (r'\b(?:no|nunca|jamás)\b', 0.7),  # Negation
    (r'\b(?:dices?|dice|dicen|afirma[ns]?)\b', 0.8),  # Reported speech
    (r'\b(?:según|supuestamente|aparentemente)\b', 0.8),  # Uncertainty markers
    (r'[¿?]', 0.9),  # Questions
]

class FarRightAnalyzer:
    """Advanced analyzer for far-right content in Spanish social media."""
    
    def __init__(self):
        self.compiled_patterns = self._compile_patterns()
        self.hate_patterns = [(re.compile(p, re.IGNORECASE | re.UNICODE), cat, w, tl) 
                             for p, cat, w, tl in HATE_SPEECH_PATTERNS]
        self.amplifiers = [(re.compile(p, re.IGNORECASE | re.UNICODE), mult) 
                          for p, mult in CONTEXTUAL_AMPLIFIERS]
        self.minimizers = [(re.compile(p, re.IGNORECASE | re.UNICODE), mult) 
                          for p, mult in MINIMIZING_CONTEXTS]
    
    def _compile_patterns(self) -> Dict:
        """Compile all regex patterns for efficient matching."""
        compiled = {}
        for category, data in FAR_RIGHT_PATTERNS.items():
            compiled[category] = {
                'weight_multiplier': data['weight_multiplier'],
                'patterns': [(re.compile(pattern, re.IGNORECASE | re.UNICODE), 
                            subcat, weight, threat) 
                           for pattern, subcat, weight, threat in data['patterns']]
            }
        return compiled
    
    def analyze_text(self, text: str) -> Dict:
        """
        Comprehensive analysis of text for far-right indicators.
        Returns detailed breakdown with threat assessment.
        """
        if not text or len(text.strip()) < 3:
            return self._empty_result()
        
        text_lower = text.lower()
        matches = []
        total_score = 0.0
        category_scores = {}
        threat_levels = []
        
        # Main pattern matching
        for category, data in self.compiled_patterns.items():
            category_score = 0.0
            category_matches = []
            
            for pattern, subcat, weight, threat in data['patterns']:
                found = pattern.findall(text)
                if found:
                    match_score = len(found) * weight * data['weight_multiplier']
                    category_score += match_score
                    threat_levels.append(threat)
                    
                    for match in found:
                        matches.append(PatternMatch(
                            pattern=match,
                            category=f"{category}:{subcat}",
                            weight=weight,
                            threat_level=threat,
                            context=self._extract_context(text, match)
                        ))
                        category_matches.append(match)
            
            if category_score > 0:
                category_scores[category] = {
                    'score': category_score,
                    'matches': category_matches,
                    'count': len(category_matches)
                }
                total_score += category_score
        
        # Hate speech analysis
        hate_score = 0.0
        hate_matches = []
        for pattern, subcat, weight, threat in self.hate_patterns:
            found = pattern.findall(text)
            if found:
                hate_score += len(found) * weight
                threat_levels.append(threat)
                hate_matches.extend(found)
        
        # Apply contextual modifiers
        amplification = self._calculate_amplification(text)
        minimization = self._calculate_minimization(text)
        
        # Final score calculation
        final_score = (total_score + hate_score) * amplification * minimization
        
        # Normalize to 0-1 scale (with cap at 1.0)
        max_possible_score = 50.0  # Reasonable maximum for normalization
        normalized_score = min(1.0, final_score / max_possible_score)
        
        # Determine overall threat level
        max_threat = max(threat_levels, key=lambda x: x.value) if threat_levels else ThreatLevel.LOW
        
        return {
            'score': round(normalized_score, 3),
            'raw_score': round(final_score, 2),
            'threat_level': max_threat.name,
            'category_breakdown': category_scores,
            'hate_speech_score': round(hate_score, 2),
            'hate_speech_matches': hate_matches,
            'total_matches': len(matches),
            'amplification_factor': round(amplification, 2),
            'minimization_factor': round(minimization, 2),
            'pattern_matches': [
                {
                    'pattern': m.pattern,
                    'category': m.category,
                    'weight': m.weight,
                    'threat_level': m.threat_level.name,
                    'context': m.context
                } for m in matches
            ],
            'risk_assessment': self._assess_risk(normalized_score, max_threat, matches)
        }
    
    def _extract_context(self, text: str, match: str, window: int = 30) -> str:
        """Extract surrounding context for a match."""
        try:
            start = max(0, text.lower().find(match.lower()) - window)
            end = min(len(text), text.lower().find(match.lower()) + len(match) + window)
            return text[start:end].strip()
        except:
            return ""
    
    def _calculate_amplification(self, text: str) -> float:
        """Calculate amplification factor based on contextual cues."""
        factor = 1.0
        for pattern, mult in self.amplifiers:
            if pattern.search(text):
                factor *= mult
        return min(factor, 2.0)  # Cap amplification
    
    def _calculate_minimization(self, text: str) -> float:
        """Calculate minimization factor based on negation and uncertainty."""
        factor = 1.0
        for pattern, mult in self.minimizers:
            if pattern.search(text):
                factor *= mult
        return max(factor, 0.3)  # Floor minimization
    
    def _assess_risk(self, score: float, threat_level: ThreatLevel, matches: List[PatternMatch]) -> str:
        """Provide human-readable risk assessment."""
        if score >= 0.8 or threat_level == ThreatLevel.CRITICAL:
            return "ALTO: Contenido con alto riesgo de extremismo y posibles amenazas"
        elif score >= 0.6 or threat_level == ThreatLevel.HIGH:
            return "MEDIO-ALTO: Contenido con retórica extremista significativa"
        elif score >= 0.4 or threat_level == ThreatLevel.MEDIUM:
            return "MEDIO: Contenido con elementos de retórica de extrema derecha"
        elif score >= 0.2:
            return "BAJO: Contenido con algunos indicadores menores"
        else:
            return "MÍNIMO: Sin indicadores significativos de extremismo"
    
    def _empty_result(self) -> Dict:
        """Return empty analysis result."""
        return {
            'score': 0.0,
            'raw_score': 0.0,
            'threat_level': 'LOW',
            'category_breakdown': {},
            'hate_speech_score': 0.0,
            'hate_speech_matches': [],
            'total_matches': 0,
            'amplification_factor': 1.0,
            'minimization_factor': 1.0,
            'pattern_matches': [],
            'risk_assessment': 'MÍNIMO: Sin indicadores significativos de extremismo'
        }

def analyze_far_right_content(text: str) -> Dict:
    """Convenience function for quick analysis."""
    analyzer = FarRightAnalyzer()
    return analyzer.analyze_text(text)

# Quick test
if __name__ == "__main__":
    test_texts = [
        "Los inmigrantes ilegales nos están invadiendo, es hora de defender España",
        "Sánchez es un traidor que nos está vendiendo a Soros y la élite globalista",
        "Hoy hace sol en Madrid",
        "A las armas, españoles! Reconquista y revolución! Fuera invasores!"
    ]
    
    analyzer = FarRightAnalyzer()
    for text in test_texts:
        result = analyzer.analyze_text(text)
        print(f"\nTexto: {text}")
        print(f"Puntuación: {result['score']}")
        print(f"Nivel de amenaza: {result['threat_level']}")
        print(f"Evaluación: {result['risk_assessment']}")
