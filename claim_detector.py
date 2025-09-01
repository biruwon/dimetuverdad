"""
Spanish claim detection system for identifying verifiable statements.
Detects factual claims, statistics, and verifiable assertions in Spanish text.
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class ClaimType(Enum):
    ESTADISTICA = "estadística"
    MEDICA = "médica"
    ECONOMICA = "económica"
    HISTORICA = "histórica"
    CIENTIFICA = "científica"
    POLITICA = "política"
    SOCIAL = "social"
    DEMOGRAFICA = "demográfica"
    GENERAL = "general"

class VerifiabilityLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class UrgencyLevel(Enum):
    URGENT = "urgent"
    NORMAL = "normal"
    LOW = "low"

@dataclass
class Claim:
    text: str
    claim_type: ClaimType
    verifiability: VerifiabilityLevel
    urgency: UrgencyLevel
    confidence: float
    key_entities: List[str]
    verification_keywords: List[str]

class SpanishClaimDetector:
    """
    Detects and classifies factual claims in Spanish text.
    Identifies statements that can be fact-checked and verified.
    """
    
    def __init__(self):
        self.claim_patterns = self._initialize_claim_patterns()
        self.entity_patterns = self._initialize_entity_patterns()
        self.urgency_indicators = self._initialize_urgency_indicators()
        
    def _initialize_claim_patterns(self) -> Dict[ClaimType, List[Dict]]:
        """Initialize patterns for detecting different types of claims."""
        return {
            ClaimType.ESTADISTICA: [
                {
                    'pattern': r'\b(\d+(?:[.,]\d+)*)\s*%\s+(?:de\s+)?(?:los\s+)?(\w+)',
                    'weight': 0.9,
                    'description': 'Porcentajes con grupos'
                },
                {
                    'pattern': r'\b(?:según|conforme\s+a|de\s+acuerdo\s+con)\s+(?:el\s+)?(\w+),?\s+(\d+(?:[.,]\d+)*)',
                    'weight': 0.8,
                    'description': 'Estadísticas con fuente'
                },
                {
                    'pattern': r'\b(\d+(?:[.,]\d+)*)\s+(?:millones?|miles?|euros?|personas?|casos?)\s+(?:de\s+)?(\w+)',
                    'weight': 0.7,
                    'description': 'Cifras absolutas'
                },
                {
                    'pattern': r'\b(?:aumentó|disminuyó|creció|bajó)\s+(?:un\s+)?(\d+(?:[.,]\d+)*)\s*%',
                    'weight': 0.8,
                    'description': 'Cambios porcentuales'
                }
            ],
            
            ClaimType.MEDICA: [
                {
                    'pattern': r'\b(?:vacunas?|medicamentos?|tratamientos?)\s+(?:causan?|provocan?|generan?)\s+(\w+)',
                    'weight': 0.9,
                    'description': 'Efectos médicos causales'
                },
                {
                    'pattern': r'\b(?:covid|coronavirus|pandemia)\s+(?:es|fue|será)\s+(\w+)',
                    'weight': 0.8,
                    'description': 'Afirmaciones sobre COVID'
                },
                {
                    'pattern': r'\b(?:estudios?|investigación|ciencia)\s+(?:demuestra|prueba|confirma)\s+que\s+(.+)',
                    'weight': 0.8,
                    'description': 'Referencias a estudios'
                },
                {
                    'pattern': r'\b(?:efectos?\s+(?:secundarios?|adversos?))\s+(?:de\s+)?(.+)',
                    'weight': 0.7,
                    'description': 'Efectos secundarios'
                }
            ],
            
            ClaimType.ECONOMICA: [
                {
                    'pattern': r'\b(?:pib|inflación|desempleo|paro)\s+(?:es|está|alcanza)\s+(?:del?\s+)?(\d+(?:[.,]\d+)*)\s*%',
                    'weight': 0.9,
                    'description': 'Indicadores económicos'
                },
                {
                    'pattern': r'\b(?:salario|sueldo|pensión)\s+(?:medio|promedio)\s+(?:es|está|alcanza)\s+(\d+(?:[.,]\d+)*)\s*euros?',
                    'weight': 0.8,
                    'description': 'Datos salariales'
                },
                {
                    'pattern': r'\b(?:presupuesto|gasto|inversión)\s+(?:de|en)\s+(.+?)\s+(?:es|será|alcanza)\s+(\d+(?:[.,]\d+)*)',
                    'weight': 0.8,
                    'description': 'Datos presupuestarios'
                }
            ],
            
            ClaimType.HISTORICA: [
                {
                    'pattern': r'\b(?:en\s+)?(\d{4})\s+(?:ocurrió|sucedió|pasó)\s+(.+)',
                    'weight': 0.8,
                    'description': 'Eventos históricos fechados'
                },
                {
                    'pattern': r'\b(?:franco|dictadura|guerra\s+civil)\s+(?:causó|mató|asesinó)\s+(\d+(?:[.,]\d+)*)',
                    'weight': 0.9,
                    'description': 'Cifras históricas polémicas'
                },
                {
                    'pattern': r'\b(?:durante|bajo)\s+(?:el\s+)?(\w+)\s+(?:murieron|fallecieron)\s+(\d+(?:[.,]\d+)*)',
                    'weight': 0.8,
                    'description': 'Víctimas históricas'
                }
            ],
            
            ClaimType.CIENTIFICA: [
                {
                    'pattern': r'\b(?:la\s+ciencia|científicos?|investigadores?)\s+(?:dice|afirma|demuestra)\s+que\s+(.+)',
                    'weight': 0.8,
                    'description': 'Afirmaciones científicas'
                },
                {
                    'pattern': r'\b(?:está\s+(?:científicamente\s+)?(?:probado|demostrado))\s+que\s+(.+)',
                    'weight': 0.9,
                    'description': 'Pruebas científicas'
                },
                {
                    'pattern': r'\b(?:cambio\s+climático|calentamiento\s+global)\s+(?:es|no\s+es)\s+(.+)',
                    'weight': 0.8,
                    'description': 'Afirmaciones climáticas'
                }
            ],
            
            ClaimType.POLITICA: [
                {
                    'pattern': r'\b(?:gobierno|ministro|presidente)\s+(\w+)\s+(?:dijo|afirmó|prometió)\s+(.+)',
                    'weight': 0.7,
                    'description': 'Declaraciones políticas'
                },
                {
                    'pattern': r'\b(?:ley|decreto|normativa)\s+(?:establece|dice|prohibe)\s+(.+)',
                    'weight': 0.8,
                    'description': 'Contenido legislativo'
                },
                {
                    'pattern': r'\b(?:encuesta|sondeo|polling)\s+(?:dice|muestra|revela)\s+(.+)',
                    'weight': 0.7,
                    'description': 'Resultados de encuestas'
                }
            ],
            
            ClaimType.DEMOGRAFICA: [
                {
                    'pattern': r'\b(?:población|habitantes)\s+(?:de\s+)?(\w+)\s+(?:es|son|alcanza)\s+(\d+(?:[.,]\d+)*)',
                    'weight': 0.8,
                    'description': 'Datos poblacionales'
                },
                {
                    'pattern': r'\b(?:inmigrantes?|extranjeros?)\s+(?:representan|son)\s+(?:el\s+)?(\d+(?:[.,]\d+)*)\s*%',
                    'weight': 0.9,
                    'description': 'Porcentajes de inmigración'
                },
                {
                    'pattern': r'\b(?:natalidad|mortalidad|fertilidad)\s+(?:es|está)\s+(?:en\s+)?(\d+(?:[.,]\d+)*)',
                    'weight': 0.8,
                    'description': 'Tasas demográficas'
                }
            ]
        }
    
    def _initialize_entity_patterns(self) -> List[str]:
        """Initialize patterns for extracting key entities from claims."""
        return [
            r'\b(?:españa|madrid|barcelona|valencia|sevilla|bilbao)\b',
            r'\b(?:europa|eeuu|china|rusia|francia|alemania)\b',
            r'\b(?:covid|coronavirus|omicron|delta|vacuna)\b',
            r'\b(?:psoe|pp|vox|podemos|ciudadanos)\b',
            r'\b(?:sánchez|feijóo|abascal|iglesias)\b',
            r'\b(?:gobierno|congreso|senado|tribunal)\b',
            r'\b(?:oms|ue|otan|onu|fmi)\b',
            r'\b\d{4}\b',  # Years
            r'\b\d+(?:[.,]\d+)*\s*(?:%|euros?|millones?|miles?)\b'  # Numbers with units
        ]
    
    def _initialize_urgency_indicators(self) -> List[Dict]:
        """Initialize patterns for detecting urgency in claims."""
        return [
            {'pattern': r'\b(?:urgente|inmediatamente|ya|ahora)\b', 'weight': 1.0},
            {'pattern': r'\b(?:crisis|emergencia|peligro|riesgo)\b', 'weight': 0.8},
            {'pattern': r'\b(?:importante|crucial|vital|necesario)\b', 'weight': 0.6},
            {'pattern': r'[!]{2,}', 'weight': 0.7},
            {'pattern': r'\b(?:hoy|mañana|esta\s+semana)\b', 'weight': 0.5}
        ]
    
    def detect_claims(self, text: str) -> List[Claim]:
        """
        Detect and classify factual claims in the given text.
        """
        if not text or len(text.strip()) < 10:
            return []
        
        text_lower = text.lower()
        claims = []
        
        # Detect claims by type
        for claim_type, patterns in self.claim_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info['pattern']
                weight = pattern_info['weight']
                description = pattern_info['description']
                
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for match in matches:
                    claim_text = match.group(0)
                    
                    # Extract key entities
                    entities = self._extract_entities(claim_text)
                    
                    # Determine verifiability
                    verifiability = self._assess_verifiability(claim_type, claim_text, entities)
                    
                    # Determine urgency
                    urgency = self._assess_urgency(claim_text)
                    
                    # Generate verification keywords
                    verification_keywords = self._generate_verification_keywords(claim_type, entities)
                    
                    claim = Claim(
                        text=claim_text.strip(),
                        claim_type=claim_type,
                        verifiability=verifiability,
                        urgency=urgency,
                        confidence=round(weight, 3),
                        key_entities=entities,
                        verification_keywords=verification_keywords
                    )
                    
                    claims.append(claim)
        
        # Remove duplicates and sort by confidence
        unique_claims = self._deduplicate_claims(claims)
        unique_claims.sort(key=lambda x: x.confidence, reverse=True)
        
        return unique_claims[:10]  # Return top 10 most confident claims
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from claim text."""
        entities = []
        text_lower = text.lower()
        
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            entities.extend(matches)
        
        return list(set(entities))[:5]  # Return unique entities, max 5
    
    def _assess_verifiability(self, claim_type: ClaimType, text: str, entities: List[str]) -> VerifiabilityLevel:
        """Assess how verifiable a claim is."""
        text_lower = text.lower()
        
        # High verifiability indicators
        high_indicators = [
            r'\b(?:según|conforme|de\s+acuerdo\s+con)\s+(?:el\s+)?(?:ine|oms|ue|gobierno|ministerio)\b',
            r'\b(?:estudio|investigación|informe)\s+(?:de|del)\s+(\w+)\b',
            r'\b\d{4}\b',  # Specific years
            r'\b\d+(?:[.,]\d+)*\s*(?:%|euros?|millones?)\b'  # Specific numbers
        ]
        
        # Low verifiability indicators
        low_indicators = [
            r'\b(?:dicen|se\s+dice|se\s+comenta|rumores?)\b',
            r'\b(?:parece|aparentemente|probablemente)\b',
            r'\b(?:algunos|muchos|varios)\s+(?:expertos?|estudios?)\b'
        ]
        
        high_score = sum(1 for pattern in high_indicators if re.search(pattern, text_lower))
        low_score = sum(1 for pattern in low_indicators if re.search(pattern, text_lower))
        
        if high_score >= 2 or (high_score >= 1 and claim_type in [ClaimType.ESTADISTICA, ClaimType.ECONOMICA]):
            return VerifiabilityLevel.HIGH
        elif low_score >= 1:
            return VerifiabilityLevel.LOW
        else:
            return VerifiabilityLevel.MEDIUM
    
    def _assess_urgency(self, text: str) -> UrgencyLevel:
        """Assess the urgency level of a claim."""
        text_lower = text.lower()
        urgency_score = 0.0
        
        for indicator in self.urgency_indicators:
            if re.search(indicator['pattern'], text_lower):
                urgency_score += indicator['weight']
        
        if urgency_score >= 1.0:
            return UrgencyLevel.URGENT
        elif urgency_score >= 0.5:
            return UrgencyLevel.NORMAL
        else:
            return UrgencyLevel.LOW
    
    def _generate_verification_keywords(self, claim_type: ClaimType, entities: List[str]) -> List[str]:
        """Generate keywords useful for fact-checking the claim."""
        keywords = list(entities)
        
        type_keywords = {
            ClaimType.ESTADISTICA: ['estadística', 'datos', 'cifras', 'ine'],
            ClaimType.MEDICA: ['medicina', 'salud', 'oms', 'sanidad'],
            ClaimType.ECONOMICA: ['economía', 'finanzas', 'banco', 'pib'],
            ClaimType.HISTORICA: ['historia', 'archivo', 'documento', 'fecha'],
            ClaimType.CIENTIFICA: ['ciencia', 'investigación', 'estudio', 'peer-review'],
            ClaimType.POLITICA: ['política', 'gobierno', 'congreso', 'oficial'],
            ClaimType.DEMOGRAFICA: ['población', 'censo', 'demografía', 'ine']
        }
        
        if claim_type in type_keywords:
            keywords.extend(type_keywords[claim_type])
        
        return list(set(keywords))[:8]  # Return unique keywords, max 8
    
    def _deduplicate_claims(self, claims: List[Claim]) -> List[Claim]:
        """Remove duplicate claims based on text similarity."""
        unique_claims = []
        seen_texts = set()
        
        for claim in claims:
            # Simple deduplication based on first 50 characters
            text_key = claim.text[:50].lower().strip()
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_claims.append(claim)
        
        return unique_claims
    
    def has_verifiable_claims(self, text: str) -> bool:
        """Check if text contains any verifiable claims."""
        claims = self.detect_claims(text)
        return len(claims) > 0
    
    def get_primary_claim_type(self, text: str) -> Optional[ClaimType]:
        """Get the most prominent claim type in the text."""
        claims = self.detect_claims(text)
        if not claims:
            return None
        return claims[0].claim_type

# Test function
if __name__ == "__main__":
    test_texts = [
        "El 85% de los inmigrantes no trabajan según el INE",
        "Las vacunas COVID causan miocarditis en el 2% de los casos",
        "El PIB español creció un 3.2% el año pasado",
        "Franco mató a 500.000 personas durante la dictadura",
        "Los científicos demuestran que el cambio climático es una mentira",
        "Sánchez prometió bajar los impuestos pero los subió un 15%",
        "La población de Madrid alcanza los 7 millones de habitantes",
        "Estudios secretos confirman que los microchips controlan la mente"
    ]
    
    detector = SpanishClaimDetector()
    
    for text in test_texts:
        print(f"\n--- Texto: {text}")
        claims = detector.detect_claims(text)
        for claim in claims:
            print(f"Tipo: {claim.claim_type.value}")
            print(f"Verificabilidad: {claim.verifiability.value}")
            print(f"Urgencia: {claim.urgency.value}")
            print(f"Confianza: {claim.confidence}")
            if claim.key_entities:
                print(f"Entidades: {', '.join(claim.key_entities)}")
