"""
Advanced claim detection and fact-checking pipeline for Spanish content.
Specializes in detecting verifiable claims, misinformation patterns, and fact-check opportunities.
"""

import re
import json
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class ClaimType(Enum):
    STATISTICAL = "estadística"
    FACTUAL = "factual"
    PREDICTION = "predicción"
    HISTORICAL = "histórica"
    SCIENTIFIC = "científica"
    LEGAL = "legal"
    ECONOMIC = "económica"
    MEDICAL = "médica"
    ELECTORAL = "electoral"
    POLICY = "política_pública"
    CONSPIRACY = "conspiración"
    TESTIMONIAL = "testimonial"

class VerifiabilityLevel(Enum):
    HIGH = "alta"          # Easily verifiable with official sources
    MEDIUM = "media"       # Verifiable with some research
    LOW = "baja"          # Difficult to verify
    NONE = "no_verificable" # Opinion or subjective statement

class ClaimUrgency(Enum):
    CRITICAL = "crítica"   # Immediate fact-checking needed
    HIGH = "alta"         # Should be fact-checked soon
    MEDIUM = "media"      # Worth fact-checking
    LOW = "baja"          # Less urgent

@dataclass
class DetectedClaim:
    text: str
    claim_type: ClaimType
    verifiability: VerifiabilityLevel
    urgency: ClaimUrgency
    confidence: float
    key_entities: List[str]
    numerical_data: List[str]
    temporal_references: List[str]
    source_indicators: List[str]
    verification_keywords: List[str]
    context: str

class SpanishClaimDetector:
    """
    Advanced claim detection system optimized for Spanish political and social content.
    Focuses on identifying verifiable statements that may require fact-checking.
    """
    
    def __init__(self):
        self.claim_patterns = self._initialize_claim_patterns()
        self.entity_patterns = self._initialize_entity_patterns()
        self.temporal_patterns = self._initialize_temporal_patterns()
        self.source_patterns = self._initialize_source_patterns()
        self.urgency_amplifiers = self._initialize_urgency_amplifiers()
        self.misinformation_flags = self._initialize_misinformation_flags()
    
    def _initialize_claim_patterns(self) -> Dict[ClaimType, List[Tuple[str, float]]]:
        """Initialize patterns for different types of claims."""
        return {
            ClaimType.STATISTICAL: [
                (r'\b\d+(?:[,.]\d+)*\s*%\b', 0.9),  # Percentages
                (r'\b\d+(?:[,.]\d+)*\s*(?:millones?|miles?|billones?)\b', 0.8),  # Large numbers
                (r'\b(?:aumentó|disminuyó|creció|bajó)\s+(?:un?\s+)?\d+', 0.8),  # Statistical changes
                (r'\b(?:datos?|estadísticas?|cifras?|números?)\s+(?:oficiales?|del?\s+gobierno)\b', 0.7),
                (r'\b(?:según|de\s+acuerdo\s+a)\s+(?:estudios?|informes?|encuestas?)\b', 0.7),
                (r'\b\d+\s+de\s+cada\s+\d+\b', 0.8),  # Ratios
                (r'\b(?:la\s+mayoría|la\s+minoría|el\s+\d+%)\s+de\s+(?:los?\s+)?españoles\b', 0.7)
            ],
            
            ClaimType.FACTUAL: [
                (r'\b(?:es\s+(?:un\s+)?hecho|es\s+verdad|es\s+falso|es\s+mentira)\s+que\b', 0.9),
                (r'\b(?:confirmó|desmintió|reveló|anunció|declaró)\s+que\b', 0.8),
                (r'\b(?:se\s+ha\s+(?:confirmado|demostrado|probado))\s+que\b', 0.8),
                (r'\b(?:la\s+realidad\s+es|lo\s+cierto\s+es|en\s+realidad)\b', 0.7),
                (r'\b(?:está\s+(?:confirmado|demostrado|probado))\s+que\b', 0.8),
                (r'\b(?:nadie\s+puede\s+negar|es\s+innegable)\s+que\b', 0.7)
            ],
            
            ClaimType.PREDICTION: [
                (r'\b(?:va\s+a|van\s+a|será|serán|ocurrirá|pasará)\b', 0.6),
                (r'\b(?:en\s+(?:el\s+)?futuro|dentro\s+de|para\s+el?\s+año)\b', 0.7),
                (r'\b(?:predicción|pronóstico|previsión|estimación)\b', 0.8),
                (r'\b(?:se\s+espera|se\s+prevé|se\s+anticipa)\s+que\b', 0.7),
                (r'\b(?:muy\s+pronto|en\s+breve|próximamente)\b', 0.6)
            ],
            
            ClaimType.HISTORICAL: [
                (r'\b(?:en\s+el?\s+año\s+\d{4}|hace\s+\d+\s+años?)\b', 0.7),
                (r'\b(?:durante\s+(?:el?\s+)?(?:franquismo|dictadura|transición))\b', 0.8),
                (r'\b(?:historia|históricamente|en\s+el?\s+pasado)\b', 0.6),
                (r'\b(?:guerra\s+civil|segunda\s+república|restauración)\b', 0.8),
                (r'\b(?:nunca\s+antes|por\s+primera\s+vez|sin\s+precedentes)\b', 0.7)
            ],
            
            ClaimType.MEDICAL: [
                (r'\b(?:vacunas?|vacunación|inmunización)\b', 0.8),
                (r'\b(?:coronavirus|covid|pandemia|epidemia)\b', 0.8),
                (r'\b(?:efectos?\s+secundarios?|reacciones?\s+adversas?)\b', 0.8),
                (r'\b(?:medicamentos?|fármacos?|tratamientos?)\b', 0.7),
                (r'\b(?:oms|organización\s+mundial\s+de\s+la\s+salud)\b', 0.7),
                (r'\b(?:estudios?\s+(?:médicos?|clínicos?|científicos?))\b', 0.8)
            ],
            
            ClaimType.LEGAL: [
                (r'\b(?:ley|leyes|legislación|normativa)\b', 0.7),
                (r'\b(?:tribunal|juzgado|sentencia|fallo)\b', 0.8),
                (r'\b(?:constitución|constitucional|inconstitucional)\b', 0.8),
                (r'\b(?:delito|crimen|ilegal|legal|jurídico)\b', 0.7),
                (r'\b(?:fiscal|fiscalía|juez|magistrado)\b', 0.7),
                (r'\b(?:demanda|querella|denuncia|acusación)\b', 0.8)
            ],
            
            ClaimType.ECONOMIC: [
                (r'\b(?:pib|producto\s+interior\s+bruto)\b', 0.8),
                (r'\b(?:inflación|deflación|crisis\s+económica)\b', 0.8),
                (r'\b(?:presupuestos?|déficit|deuda\s+pública)\b', 0.8),
                (r'\b(?:euros?|millones?\s+de\s+euros?|miles\s+de\s+millones)\b', 0.7),
                (r'\b(?:impuestos?|tributos?|hacienda)\b', 0.7),
                (r'\b(?:desempleo|paro|empleo|trabajo)\b', 0.7)
            ],
            
            ClaimType.ELECTORAL: [
                (r'\b(?:elecciones?|electorales?|votación|votos?)\b', 0.8),
                (r'\b(?:candidatos?|partidos?\s+políticos?)\b', 0.7),
                (r'\b(?:encuestas?\s+electorales?|sondeos?)\b', 0.8),
                (r'\b(?:campaña\s+electoral|propaganda\s+electoral)\b', 0.8),
                (r'\b(?:escaños?|diputados?|senadores?)\b', 0.7),
                (r'\b(?:gobierno|oposición|coalición)\b', 0.6)
            ],
            
            ClaimType.CONSPIRACY: [
                (r'\b(?:conspiración|complot|encubrimiento)\b', 0.8),
                (r'\b(?:ocultan|esconden|manipulan)\s+(?:la\s+)?(?:verdad|información)\b', 0.8),
                (r'\b(?:no\s+quieren\s+que\s+sepas|te\s+ocultan)\b', 0.9),
                (r'\b(?:élite|élites)\s+(?:mundial|global|oculta)\b', 0.8),
                (r'\b(?:control\s+mental|lavado\s+de\s+cerebro|manipulación\s+mediática)\b', 0.9)
            ]
        }
    
    def _initialize_entity_patterns(self) -> List[Tuple[str, str]]:
        """Initialize patterns for detecting key entities in claims."""
        return [
            # Government entities
            (r'\b(?:gobierno|ejecutivo|ministerio|ministro)\b', 'government'),
            (r'\b(?:congreso|senado|parlamento|cortes)\b', 'legislature'),
            (r'\b(?:tribunal\s+(?:supremo|constitucional)|audiencia\s+nacional)\b', 'judiciary'),
            
            # Political parties
            (r'\b(?:psoe|pp|vox|podemos|ciudadanos|cs)\b', 'political_party'),
            
            # International entities
            (r'\b(?:ue|unión\s+europea|europa|bruselas)\b', 'international'),
            (r'\b(?:otan|nato|estados\s+unidos|eeuu)\b', 'international'),
            (r'\b(?:oms|onu|naciones\s+unidas)\b', 'international_org'),
            
            # Institutions
            (r'\b(?:banco\s+(?:de\s+)?españa|bce|banco\s+central)\b', 'financial'),
            (r'\b(?:ine|instituto\s+nacional\s+de\s+estadística)\b', 'statistical'),
            (r'\b(?:sanidad|ministerio\s+de\s+sanidad)\b', 'health'),
            
            # Media
            (r'\b(?:medios?\s+de\s+comunicación|prensa|televisión|radio)\b', 'media'),
            (r'\b(?:rtve|antena\s+3|telecinco|la\s+sexta)\b', 'media_outlet'),
            
            # Geographic
            (r'\b(?:españa|madrid|barcelona|valencia|sevilla)\b', 'geographic'),
            (r'\b(?:cataluña|euskadi|galicia|andalucía)\b', 'region'),
        ]
    
    def _initialize_temporal_patterns(self) -> List[str]:
        """Initialize patterns for detecting temporal references."""
        return [
            r'\b(?:hoy|ayer|mañana|ahora|actualmente)\b',
            r'\b(?:este|esta|próximo|próxima)\s+(?:año|mes|semana|lunes|martes|miércoles|jueves|viernes|sábado|domingo)\b',
            r'\b(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+(?:de\s+)?\d{4}\b',
            r'\b\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b',
            r'\b(?:hace|dentro\s+de)\s+\d+\s+(?:días?|semanas?|meses?|años?)\b',
            r'\b(?:desde|hasta|entre)\s+(?:el?\s+)?\d{4}\b',
            r'\b(?:en\s+)?(?:el?\s+)?(?:año\s+)?\d{4}\b'
        ]
    
    def _initialize_source_patterns(self) -> List[Tuple[str, float]]:
        """Initialize patterns for detecting source citations and credibility indicators."""
        return [
            (r'\b(?:según|de\s+acuerdo\s+(?:a|con)|conforme\s+a)\b', 0.8),
            (r'\b(?:fuentes?\s+(?:oficiales?|gubernamentales?|fidedignas?))\b', 0.9),
            (r'\b(?:estudio|informe|investigación|análisis)\s+(?:oficial|del?\s+gobierno)\b', 0.9),
            (r'\b(?:datos?\s+(?:oficiales?|del?\s+ine|del?\s+gobierno))\b', 0.9),
            (r'\b(?:ha\s+(?:dicho|declarado|afirmado|confirmado))\b', 0.7),
            (r'\b(?:medios?\s+de\s+comunicación|prensa|periodistas?)\b', 0.6),
            (r'\b(?:redes?\s+sociales?|twitter|facebook|telegram)\b', 0.3),
            (r'\b(?:he\s+leído|me\s+han\s+dicho|dicen\s+que)\b', 0.2),
            (r'\b(?:expertos?|especialistas?|científicos?)\b', 0.8),
            (r'\b(?:universidades?|centros?\s+de\s+investigación)\b', 0.8)
        ]
    
    def _initialize_urgency_amplifiers(self) -> List[Tuple[str, float]]:
        """Initialize patterns that increase claim urgency."""
        return [
            (r'\b(?:urgente|emergencia|crisis|alerta)\b', 2.0),
            (r'\b(?:inmediatamente|ahora\s+mismo|ya|cuanto\s+antes)\b', 1.5),
            (r'\b(?:peligro|riesgo|amenaza|grave)\b', 1.8),
            (r'\b(?:todos?\s+(?:los?\s+)?españoles?\s+deben\s+saber)\b', 1.7),
            (r'\b(?:ocultan|esconden|censuran)\b', 1.6),
            (r'\b(?:última\s+hora|breaking|noticia\s+urgente)\b', 1.8),
            (r'[!]{3,}', 1.3),
            (r'[🔴⚠️🚨💥]', 1.2)
        ]
    
    def _initialize_misinformation_flags(self) -> List[Tuple[str, float]]:
        """Initialize patterns that flag potential misinformation."""
        return [
            (r'\b(?:no\s+quieren\s+que\s+sepas|te\s+ocultan|la\s+verdad\s+que)\b', 0.9),
            (r'\b(?:medios?\s+(?:manipulados?|comprados?|vendidos?))\b', 0.8),
            (r'\b(?:(?:ellos?|élites?)\s+controlan)\b', 0.8),
            (r'\b(?:despierta|abre\s+los\s+ojos|no\s+seas?\s+borrego)\b', 0.7),
            (r'\b(?:dictadura\s+(?:sanitaria|mediática))\b', 0.8),
            (r'\b(?:nueva\s+normalidad|nuevo\s+orden\s+mundial)\b', 0.8),
            (r'\b(?:plandemia|casodemia|bozalemia)\b', 0.9),
            (r'\b(?:microchips?\s+en\s+las?\s+vacunas?)\b', 0.9),
            (r'\b(?:5g\s+(?:mata|controla|manipula))\b', 0.9)
        ]
    
    def detect_claims(self, text: str) -> List[DetectedClaim]:
        """
        Detect and analyze verifiable claims in Spanish text.
        Returns list of detected claims with analysis.
        """
        if not text or len(text.strip()) < 10:
            return []
        
        claims = []
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            if len(sentence.strip()) < 15:  # Skip very short sentences
                continue
                
            claim_analysis = self._analyze_sentence_for_claims(sentence, text)
            if claim_analysis:
                claims.extend(claim_analysis)
        
        # Deduplicate and sort by confidence
        unique_claims = self._deduplicate_claims(claims)
        unique_claims.sort(key=lambda x: x.confidence, reverse=True)
        
        return unique_claims[:10]  # Return top 10 claims
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for individual analysis."""
        # Simple sentence splitting for Spanish
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _analyze_sentence_for_claims(self, sentence: str, full_context: str) -> List[DetectedClaim]:
        """Analyze a single sentence for verifiable claims."""
        sentence_lower = sentence.lower()
        detected_claims = []
        
        # Check each claim type
        for claim_type, patterns in self.claim_patterns.items():
            type_confidence = 0.0
            matched_patterns = []
            
            for pattern, weight in patterns:
                matches = re.findall(pattern, sentence_lower, re.IGNORECASE)
                if matches:
                    type_confidence += len(matches) * weight
                    matched_patterns.extend(matches)
            
            if type_confidence > 0.5:  # Threshold for considering it a claim
                # Extract additional information
                entities = self._extract_entities(sentence)
                numerical_data = self._extract_numerical_data(sentence)
                temporal_refs = self._extract_temporal_references(sentence)
                source_indicators = self._extract_source_indicators(sentence)
                verification_keywords = self._extract_verification_keywords(sentence)
                
                # Determine verifiability
                verifiability = self._assess_verifiability(
                    claim_type, entities, numerical_data, source_indicators
                )
                
                # Calculate urgency
                urgency = self._calculate_urgency(sentence, full_context, claim_type)
                
                # Final confidence calculation
                final_confidence = min(1.0, type_confidence / 2.0)  # Normalize
                
                detected_claims.append(DetectedClaim(
                    text=sentence.strip(),
                    claim_type=claim_type,
                    verifiability=verifiability,
                    urgency=urgency,
                    confidence=round(final_confidence, 3),
                    key_entities=entities,
                    numerical_data=numerical_data,
                    temporal_references=temporal_refs,
                    source_indicators=source_indicators,
                    verification_keywords=verification_keywords,
                    context=self._extract_context(full_context, sentence)
                ))
        
        return detected_claims
    
    def _extract_entities(self, sentence: str) -> List[str]:
        """Extract key entities from the sentence."""
        entities = []
        sentence_lower = sentence.lower()
        
        for pattern, entity_type in self.entity_patterns:
            matches = re.findall(pattern, sentence_lower, re.IGNORECASE)
            for match in matches:
                entities.append(f"{entity_type}:{match}")
        
        return entities[:5]  # Limit to top 5
    
    def _extract_numerical_data(self, sentence: str) -> List[str]:
        """Extract numerical data that could be fact-checked."""
        patterns = [
            r'\b\d+(?:[,.]\d+)*\s*%',  # Percentages
            r'\b\d+(?:[,.]\d+)*\s*(?:millones?|miles?|billones?)',  # Large numbers
            r'\b\d+(?:[,.]\d+)*\s*euros?',  # Money
            r'\b\d{1,2}(?:[,.]\d+)?\s*(?:grados?|°C)',  # Temperature
            r'\b\d+\s+de\s+cada\s+\d+',  # Ratios
            r'\b\d{4}',  # Years
        ]
        
        numerical_data = []
        for pattern in patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            numerical_data.extend(matches)
        
        return numerical_data[:5]
    
    def _extract_temporal_references(self, sentence: str) -> List[str]:
        """Extract temporal references for verification context."""
        temporal_refs = []
        sentence_lower = sentence.lower()
        
        for pattern in self.temporal_patterns:
            matches = re.findall(pattern, sentence_lower, re.IGNORECASE)
            temporal_refs.extend(matches)
        
        return temporal_refs[:3]
    
    def _extract_source_indicators(self, sentence: str) -> List[str]:
        """Extract source credibility indicators."""
        source_indicators = []
        sentence_lower = sentence.lower()
        
        for pattern, credibility in self.source_patterns:
            matches = re.findall(pattern, sentence_lower, re.IGNORECASE)
            for match in matches:
                source_indicators.append(f"{credibility}:{match}")
        
        return source_indicators[:3]
    
    def _extract_verification_keywords(self, sentence: str) -> List[str]:
        """Extract keywords relevant for verification."""
        verification_patterns = [
            r'\b(?:confirmó|desmintió|verificó|comprobó|investigó)\b',
            r'\b(?:según|conforme|de\s+acuerdo)\b',
            r'\b(?:oficial|oficialmente|autoridades?)\b',
            r'\b(?:datos?|estadísticas?|informes?|estudios?)\b'
        ]
        
        keywords = []
        sentence_lower = sentence.lower()
        
        for pattern in verification_patterns:
            matches = re.findall(pattern, sentence_lower, re.IGNORECASE)
            keywords.extend(matches)
        
        return keywords[:5]
    
    def _assess_verifiability(
        self, 
        claim_type: ClaimType, 
        entities: List[str], 
        numerical_data: List[str], 
        source_indicators: List[str]
    ) -> VerifiabilityLevel:
        """Assess how easily verifiable a claim is."""
        score = 0
        
        # Statistical and factual claims with numbers are highly verifiable
        if claim_type in [ClaimType.STATISTICAL, ClaimType.ECONOMIC, ClaimType.ELECTORAL]:
            score += 3
        
        # Claims with numerical data are more verifiable
        if numerical_data:
            score += 2
        
        # Claims with official entities are more verifiable
        official_entities = [e for e in entities if 'government' in e or 'statistical' in e]
        if official_entities:
            score += 2
        
        # Claims with credible sources are more verifiable
        credible_sources = [s for s in source_indicators if s.startswith('0.8') or s.startswith('0.9')]
        if credible_sources:
            score += 2
        
        # Conspiracy claims are less verifiable
        if claim_type == ClaimType.CONSPIRACY:
            score -= 2
        
        # Testimonial claims are less verifiable
        if claim_type == ClaimType.TESTIMONIAL:
            score -= 1
        
        if score >= 5:
            return VerifiabilityLevel.HIGH
        elif score >= 3:
            return VerifiabilityLevel.MEDIUM
        elif score >= 1:
            return VerifiabilityLevel.LOW
        else:
            return VerifiabilityLevel.NONE
    
    def _calculate_urgency(self, sentence: str, full_context: str, claim_type: ClaimType) -> ClaimUrgency:
        """Calculate the urgency of fact-checking this claim."""
        urgency_score = 1.0
        
        # Check urgency amplifiers
        sentence_lower = sentence.lower()
        for pattern, multiplier in self.urgency_amplifiers:
            if re.search(pattern, sentence_lower, re.IGNORECASE):
                urgency_score *= multiplier
        
        # Check misinformation flags
        for pattern, flag_score in self.misinformation_flags:
            if re.search(pattern, sentence_lower, re.IGNORECASE):
                urgency_score *= (1 + flag_score)
        
        # Certain claim types are more urgent
        if claim_type in [ClaimType.MEDICAL, ClaimType.CONSPIRACY]:
            urgency_score *= 1.5
        
        if claim_type == ClaimType.ELECTORAL:
            urgency_score *= 1.3
        
        # Viral indicators (caps, multiple exclamations)
        if re.search(r'[A-Z]{4,}', sentence):
            urgency_score *= 1.2
        
        if re.search(r'[!]{2,}', sentence):
            urgency_score *= 1.1
        
        if urgency_score >= 4.0:
            return ClaimUrgency.CRITICAL
        elif urgency_score >= 2.5:
            return ClaimUrgency.HIGH
        elif urgency_score >= 1.5:
            return ClaimUrgency.MEDIUM
        else:
            return ClaimUrgency.LOW
    
    def _extract_context(self, full_text: str, sentence: str, window: int = 100) -> str:
        """Extract surrounding context for the claim."""
        try:
            start_pos = full_text.find(sentence)
            if start_pos == -1:
                return sentence
            
            context_start = max(0, start_pos - window)
            context_end = min(len(full_text), start_pos + len(sentence) + window)
            
            return full_text[context_start:context_end].strip()
        except:
            return sentence
    
    def _deduplicate_claims(self, claims: List[DetectedClaim]) -> List[DetectedClaim]:
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
    
    def get_high_priority_claims(self, text: str) -> List[DetectedClaim]:
        """Get only high-priority claims that need immediate attention."""
        all_claims = self.detect_claims(text)
        high_priority = [
            claim for claim in all_claims 
            if (claim.urgency in [ClaimUrgency.CRITICAL, ClaimUrgency.HIGH] or
                claim.verifiability == VerifiabilityLevel.HIGH or
                claim.confidence >= 0.7)
        ]
        return high_priority

# Convenience function
def detect_spanish_claims(text: str) -> List[DetectedClaim]:
    """Quick claim detection function."""
    detector = SpanishClaimDetector()
    return detector.detect_claims(text)

# Test the claim detector
if __name__ == "__main__":
    test_texts = [
        "El 80% de los inmigrantes ilegales cometen delitos según datos oficiales del gobierno",
        "Sánchez ha confirmado que va a subir los impuestos un 15% el próximo año",
        "Los medios ocultan que las vacunas contienen microchips para controlarnos",
        "Hoy hace sol en Madrid y es perfecto para pasear",
        "Según un estudio de la Universidad Complutense, el desempleo bajó un 3% en 2023",
        "¡URGENTE! Descubren que el 5G mata a los pájaros y nos van a hacer lo mismo"
    ]
    
    detector = SpanishClaimDetector()
    
    for text in test_texts:
        print(f"\n--- Texto: {text}")
        claims = detector.detect_claims(text)
        if claims:
            for i, claim in enumerate(claims[:2]):
                print(f"{i+1}. Tipo: {claim.claim_type.value}")
                print(f"   Verificabilidad: {claim.verifiability.value}")
                print(f"   Urgencia: {claim.urgency.value}")
                print(f"   Confianza: {claim.confidence}")
                if claim.numerical_data:
                    print(f"   Datos numéricos: {claim.numerical_data}")
        else:
            print("   Sin afirmaciones verificables detectadas")
