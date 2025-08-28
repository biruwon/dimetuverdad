"""
Enhanced analyzer for journalism: far-right activism detection with comprehensive coverage.
Integrates specialized components for professional journalism workflow.
"""

import json
import os
import sqlite3
import argparse
import time
import random
import re
import warnings
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our enhanced components
from far_right_patterns import FarRightAnalyzer, ThreatLevel
from topic_classifier import SpanishPoliticalTopicClassifier, TopicCategory
from claim_detector import SpanishClaimDetector, ClaimUrgency, VerifiabilityLevel
from retrieval import retrieve_evidence_for_post, format_evidence
from llm_models import LLMModelConfig, EnhancedLLMPipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# DB path (same as other scripts)
DB_PATH = os.path.join(os.path.dirname(__file__), 'accounts.db')
@dataclass
class JournalistAnalysis:
    """Journalism-focused analysis result structure for professional workflows."""
    # Tweet metadata
    tweet_id: str
    tweet_url: str
    username: str
    tweet_content: str
    analysis_timestamp: str
    
    # Journalism categories (standardized)
    category: str  # disinformation, hate_speech, political_bias, conspiracy_theory, call_to_action, general
    subcategory: Optional[str] = None
    confidence: float = 0.0
    
    # Journalism scoring
    far_right_score: float = 0.0
    fact_check_priority: str = "low"  # low, medium, high, critical
    journalistic_impact: str = "low"  # low, medium, high, critical
    
    # Professional analysis
    llm_explanation: str = ""
    evidence_sources: List[Dict] = None
    verification_status: str = "pending"  # pending, verified, debunked, inconclusive
    
    # Enhanced metadata
    targeted_groups: List[str] = None
    calls_to_action: bool = False
    misinformation_risk: str = "low"
    threat_level: str = "low"
    
    # Technical data
    pattern_matches: List[Dict] = None
    topic_classification: Dict = None
    claims_detected: List[Dict] = None
    analysis_json: str = ""
    
    def __post_init__(self):
        if self.evidence_sources is None:
            self.evidence_sources = []
        if self.targeted_groups is None:
            self.targeted_groups = []
        if self.pattern_matches is None:
            self.pattern_matches = []
        if self.claims_detected is None:
            self.claims_detected = []

@dataclass
class AnalysisResult:
    """Comprehensive analysis result structure (backward compatibility)."""
    # Basic info
    post_text: str
    timestamp: str
    tweet_url: Optional[str] = None
    
    # Far-right analysis
    far_right_score: float = 0.0
    threat_level: str = "LOW"
    risk_assessment: str = ""
    pattern_matches: List[Dict] = None
    category_breakdown: Dict = None
    
    # Topic classification
    primary_topic: str = "no_político"
    topic_confidence: float = 0.0
    all_topics: List[Dict] = None
    
    # Claim detection
    verifiable_claims: List[Dict] = None
    high_priority_claims: List[Dict] = None
    total_claims: int = 0
    
    # Content analysis
    misinformation_risk: str = "LOW"
    misinformation_reason: str = ""
    political_bias: str = "unknown"
    bias_confidence: float = 0.0
    calls_to_action: bool = False
    targeted_groups: List[str] = None
    
    # Evidence and verification
    evidence_retrieved: List[Dict] = None
    fact_check_priority: str = "LOW"
    verification_keywords: List[str] = None
    
    # LLM analysis (if available)
    llm_analysis: Optional[Dict] = None
    
    def __post_init__(self):
        if self.pattern_matches is None:
            self.pattern_matches = []
        if self.category_breakdown is None:
            self.category_breakdown = {}
        if self.all_topics is None:
            self.all_topics = []
        if self.verifiable_claims is None:
            self.verifiable_claims = []
        if self.high_priority_claims is None:
            self.high_priority_claims = []
        if self.targeted_groups is None:
            self.targeted_groups = []
        if self.evidence_retrieved is None:
            self.evidence_retrieved = []
        if self.verification_keywords is None:
            self.verification_keywords = []

class EnhancedAnalyzer:
    """
    Enhanced analyzer with improved LLM integration for journalism workflows.
    """
    
    def __init__(self, use_llm: bool = True, journalism_mode: bool = True, model_priority: str = "balanced"):
        self.far_right_analyzer = FarRightAnalyzer()
        self.topic_classifier = SpanishPoliticalTopicClassifier()
        self.claim_detector = SpanishClaimDetector()
        self.use_llm = use_llm
        self.journalism_mode = journalism_mode
        self.llm_pipeline = None
        
        print("🗞️ Iniciando Enhanced Analyzer para Periodismo...")
        print("Componentes cargados:")
        print("- ✓ Analizador de patrones de extrema derecha")
        print("- ✓ Clasificador de temas políticos") 
        print("- ✓ Detector de afirmaciones verificables")
        print("- ✓ Sistema de recuperación de evidencia")
        print("- ✓ Modo periodismo profesional activado")
        
        if use_llm:
            print("- ⏳ Cargando modelo LLM para análisis avanzado...")
            try:
                self.llm_pipeline = EnhancedLLMPipeline(model_priority=model_priority)
                print("- ✓ Modelo LLM cargado correctamente")
            except Exception as e:
                print(f"- ⚠️ Error cargando LLM: {e}")
                self.llm_pipeline = None
                self.use_llm = False
    
    def analyze_for_journalism(self, 
                             tweet_id: str,
                             tweet_url: str, 
                             username: str,
                             content: str,
                             retrieve_evidence: bool = True) -> JournalistAnalysis:
        """
        Professional journalism analysis with standardized categories.
        """
        if not content or len(content.strip()) < 5:
            return JournalistAnalysis(
                tweet_id=tweet_id,
                tweet_url=tweet_url,
                username=username,
                tweet_content=content,
                analysis_timestamp=datetime.now().isoformat(),
                category="general",
                confidence=0.0,
                llm_explanation="Content too short for analysis"
            )
        
        print(f"\n🔍 Análisis periodístico: @{username}")
        print(f"📝 Contenido: {content[:80]}...")
        
        # Step 1: Pattern-based analysis
        far_right_result = self.far_right_analyzer.analyze_text(content)
        topic_results = self.topic_classifier.classify_topic(content)
        claims = self.claim_detector.detect_claims(content)
        
        # Step 2: Determine journalism category
        category, subcategory, confidence = self._determine_journalism_category(
            content, far_right_result, topic_results, claims
        )
        
        # Step 3: Professional scoring
        far_right_score = far_right_result.get('score', 0.0)
        fact_check_priority = self._calculate_fact_check_priority(
            category, far_right_score, claims
        )
        journalistic_impact = self._assess_journalistic_impact(
            category, far_right_score, claims, content
        )
        
        # Step 4: Content analysis for journalism
        targeted_groups = self._extract_targeted_groups(content, far_right_result)
        calls_to_action = self._detect_calls_to_action(content, far_right_result)
        misinformation_risk = self._assess_misinformation_risk_journalism(
            far_right_result, claims, category
        )
        threat_level = far_right_result.get('threat_level', 'LOW')
        
        # Step 5: Evidence retrieval for fact-checking
        evidence_sources = []
        if retrieve_evidence and fact_check_priority in ['high', 'critical']:
            try:
                evidence_sources = retrieve_evidence_for_post(content, max_per_source=3)
            except Exception as e:
                print(f"⚠️ Error retrievando evidencia: {e}")
        
        # Step 6: LLM analysis for professional explanation
        llm_explanation = self._generate_journalism_explanation(
            content, category, far_right_score, claims, evidence_sources
        )
        
        # Step 7: Create comprehensive analysis
        analysis_data = {
            'category': category,
            'subcategory': subcategory,
            'confidence': confidence,
            'far_right_score': far_right_score,
            'pattern_matches': far_right_result.get('pattern_matches', []),
            'topic_classification': {
                'primary_topic': topic_results[0].category.value if topic_results else "no_político",
                'confidence': topic_results[0].confidence if topic_results else 0.0,
                'all_topics': [{
                    'category': t.category.value,
                    'confidence': t.confidence,
                    'subcategory': t.subcategory
                } for t in topic_results[:3]]
            },
            'claims_detected': [{
                'text': claim.text,
                'type': claim.claim_type.value,
                'urgency': claim.urgency.value,
                'verifiability': claim.verifiability.value,
                'confidence': claim.confidence
            } for claim in claims],
            'evidence_analysis': evidence_sources
        }
        
        return JournalistAnalysis(
            tweet_id=tweet_id,
            tweet_url=tweet_url,
            username=username,
            tweet_content=content,
            analysis_timestamp=datetime.now().isoformat(),
            category=category,
            subcategory=subcategory,
            confidence=confidence,
            far_right_score=far_right_score,
            fact_check_priority=fact_check_priority,
            journalistic_impact=journalistic_impact,
            llm_explanation=llm_explanation,
            evidence_sources=evidence_sources,
            targeted_groups=targeted_groups,
            calls_to_action=calls_to_action,
            misinformation_risk=misinformation_risk,
            threat_level=threat_level,
            pattern_matches=far_right_result.get('pattern_matches', []),
            topic_classification=analysis_data['topic_classification'],
            claims_detected=analysis_data['claims_detected'],
            analysis_json=json.dumps(analysis_data, ensure_ascii=False, default=str)
        )
    
    def _determine_journalism_category(self, content: str, far_right_result: Dict, 
                                     topic_results: List, claims: List) -> Tuple[str, str, float]:
        """Determine standardized journalism category with confidence."""
        content_lower = content.lower()
        
        # Check for hate speech indicators (highest priority)
        hate_indicators = [
            'raza inferior', 'sangre pura', 'eliminar', 'deportar', 'virus', 'infectan',
            'supremacía', 'superioridad', 'inferioridad racial', 'genéticamente inferior'
        ]
        if any(indicator in content_lower for indicator in hate_indicators):
            if any(group in content_lower for group in ['musulmán', 'gitano', 'moro', 'negro', 'judío']):
                return "hate_speech", "racism", 0.95
            elif any(group in content_lower for group in ['inmigr', 'extranjero', 'ilegal']):
                return "hate_speech", "xenophobia", 0.92
            else:
                return "hate_speech", "general_hatred", 0.88
        
        # Check for disinformation indicators
        disinfo_indicators = [
            'microchips', 'vacunas', 'controlarnos', 'agenda globalista', 'élite',
            'datos oficiales son mentira', 'gobierno oculta', 'estudio secreto',
            'medios mainstream ocultan'
        ]
        if any(indicator in content_lower for indicator in disinfo_indicators):
            if any(health in content_lower for health in ['vacun', 'covid', 'pandemia']):
                return "disinformation", "health_misinformation", 0.90
            elif any(stat in content_lower for stat in ['85%', '%', 'estadística', 'datos']):
                return "disinformation", "false_statistics", 0.87
            else:
                return "disinformation", "general_misinformation", 0.82
        
        # Check for conspiracy theories
        conspiracy_indicators = [
            'nuevo orden mundial', 'illuminati', 'soros', 'plan kalergi', 'reemplazar',
            'orquestado', 'controlarnos', 'élite globalista'
        ]
        if any(indicator in content_lower for indicator in conspiracy_indicators):
            if 'reemplazar' in content_lower or 'kalergi' in content_lower:
                return "conspiracy_theory", "replacement_theory", 0.91
            elif 'soros' in content_lower:
                return "conspiracy_theory", "antisemitic_conspiracy", 0.89
            else:
                return "conspiracy_theory", "global_conspiracy", 0.84
        
        # Check for calls to action
        action_indicators = [
            'concentración', 'manifestación', 'calles', 'resistencia', 'actuar',
            'organizad', 'difunde', 'comparte', 'despertad', 'todos unidos'
        ]
        if any(indicator in content_lower for indicator in action_indicators):
            if any(urgency in content_lower for urgency in ['hoy', 'ya', 'urgente', 'inmediatamente']):
                return "call_to_action", "urgent_mobilization", 0.93
            elif 'difunde' in content_lower or 'comparte' in content_lower:
                return "call_to_action", "viral_campaign", 0.85
            else:
                return "call_to_action", "general_mobilization", 0.80
        
        # Check for political bias
        political_indicators = [
            'socialistas han destruido', 'dictadura comunista', 'régimen de sánchez',
            'patriotas debemos', 'franco', 'democracia ha fracasado', 'vox puede salvar'
        ]
        if any(indicator in content_lower for indicator in political_indicators):
            if 'franco' in content_lower or 'dictadura' in content_lower:
                return "political_bias", "authoritarianism", 0.89
            elif 'régimen' in content_lower or 'golpe' in content_lower:
                return "political_bias", "antigovernment_propaganda", 0.86
            else:
                return "political_bias", "extreme_partisanship", 0.82
        
        # Use far-right score for general classification
        if far_right_result.get('score', 0) > 0.6:
            return "political_bias", "far_right_content", far_right_result['score']
        elif far_right_result.get('score', 0) > 0.3:
            return "political_bias", "political_content", far_right_result['score']
        
        # Normal content
        return "general", None, 0.95
    
    def _calculate_fact_check_priority(self, category: str, far_right_score: float, claims: List) -> str:
        """Calculate fact-checking priority for journalism workflow."""
        priority_score = 0
        
        # Category-based priority
        if category == "hate_speech":
            priority_score += 4
        elif category == "disinformation":
            priority_score += 3
        elif category == "conspiracy_theory":
            priority_score += 2
        elif category == "call_to_action":
            priority_score += 2
        
        # Far-right score influence
        if far_right_score > 0.8:
            priority_score += 3
        elif far_right_score > 0.6:
            priority_score += 2
        elif far_right_score > 0.4:
            priority_score += 1
        
        # Claims influence
        critical_claims = [c for c in claims if hasattr(c, 'urgency') and c.urgency.value == 'critical']
        high_claims = [c for c in claims if hasattr(c, 'urgency') and c.urgency.value == 'high']
        
        priority_score += len(critical_claims) * 2
        priority_score += len(high_claims)
        
        # Convert to priority level
        if priority_score >= 8:
            return "critical"
        elif priority_score >= 5:
            return "high"
        elif priority_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _assess_journalistic_impact(self, category: str, far_right_score: float, 
                                  claims: List, content: str) -> str:
        """Assess potential journalistic impact of the content."""
        impact_score = 0
        
        # High-impact categories
        if category in ["hate_speech", "disinformation"]:
            impact_score += 3
        elif category in ["conspiracy_theory", "call_to_action"]:
            impact_score += 2
        
        # Viral potential indicators
        viral_indicators = ['urgente', 'exclusivo', 'rompe', 'secreto', 'oculta', 'censurado']
        if any(indicator in content.lower() for indicator in viral_indicators):
            impact_score += 2
        
        # Audience size indicators
        audience_indicators = ['españoles', 'todos', 'ciudadanos', 'patriotas', 'pueblo']
        if any(indicator in content.lower() for indicator in audience_indicators):
            impact_score += 1
        
        # Emotional intensity
        if far_right_score > 0.7:
            impact_score += 2
        
        # Convert to impact level
        if impact_score >= 6:
            return "critical"
        elif impact_score >= 4:
            return "high"
        elif impact_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _assess_misinformation_risk_journalism(self, far_right_result: Dict, 
                                             claims: List, category: str) -> str:
        """Assess misinformation risk for journalism workflow."""
        if category == "disinformation":
            return "high"
        elif category in ["conspiracy_theory", "hate_speech"]:
            return "medium"
        elif far_right_result.get('score', 0) > 0.6:
            return "medium"
        elif len(claims) > 2:
            return "medium"
        else:
            return "low"
    
    def _generate_journalism_explanation(self, content: str, category: str, 
                                       far_right_score: float, claims: List,
                                       evidence_sources: List) -> str:
        """Generate professional explanation for journalists using enhanced prompting."""
        # Create comprehensive analysis context
        analysis_context = {
            'far_right_score': far_right_score,
            'category': category,
            'claims_count': len(claims),
            'evidence_count': len(evidence_sources),
            'threat_level': 'HIGH' if far_right_score > 0.7 else 'MEDIUM' if far_right_score > 0.4 else 'LOW',
            'targeted_groups': [],
            'pattern_matches': [{'category': category}]
        }
        
        # Use enhanced LLM pipeline with specialized journalism focus
        if self.use_llm and self.llm_pipeline:
            try:
                # Use the proper public interface from EnhancedLLMPipeline
                llm_result = self.llm_pipeline.analyze_content(content, analysis_context)
                
                # Extract explanation from the result
                explanation = llm_result.get('llm_explanation', '')
                if explanation and len(explanation) > 20:
                    return explanation
                        
            except Exception as e:
                print(f"⚠️ Error en análisis LLM periodístico avanzado: {e}")
        
        # Enhanced fallback explanations with more detail
        return self._get_enhanced_fallback_explanation(category, far_right_score, len(claims))
    
    def _get_enhanced_fallback_explanation(self, category: str, score: float, claims_count: int) -> str:
        """Enhanced fallback explanations for journalism."""
        explanations = {
            "disinformation": [
                "Contiene afirmaciones falsas que requieren verificación inmediata.",
                "Presenta información no verificada como hechos establecidos.",
                "Incluye datos estadísticos sin fuentes creíbles que los respalden.",
                "Utiliza teorías conspiratorias para justificar afirmaciones falsas."
            ],
            "hate_speech": [
                "Presenta discurso de odio dirigido a minorías específicas.",
                "Utiliza lenguaje deshumanizante hacia grupos vulnerables.",
                "Contiene supremacismo racial o étnico explícito.",
                "Promueve discriminación sistemática contra grupos minoritarios."
            ],
            "conspiracy_theory": [
                "Promueve teorías conspiratorias sin evidencia verificable.",
                "Presenta especulaciones como hechos comprobados.",
                "Utiliza patrones típicos de desinformación conspiratoria.",
                "Carece de fuentes creíbles que respalden las afirmaciones."
            ],
            "call_to_action": [
                "Contiene llamadas directas a la movilización ciudadana.",
                "Promueve organización de protestas o manifestaciones.",
                "Utiliza lenguaje de urgencia para motivar acción inmediata.",
                "Busca viralización de contenido con fines movilizadores."
            ],
            "political_bias": [
                "Presenta sesgo político extremo hacia posiciones de extrema derecha.",
                "Utiliza lenguaje polarizante contra instituciones democráticas.",
                "Promueve narrativas autoritarias o antidemocráticas.",
                "Carece de equilibrio informativo o perspectiva neutral."
            ]
        }
        
        base_explanation = explanations.get(category, ["Contenido analizado sin indicadores específicos."])[0]
        
        # Add enhanced context for journalism
        risk_level = "Alto" if score > 0.7 else "Medio" if score > 0.4 else "Bajo"
        confidence_text = f" Nivel de riesgo: {risk_level} (puntuación: {score:.2f})."
        
        # Add claims information
        if claims_count > 0:
            claims_text = f" Se detectaron {claims_count} afirmaciones verificables que requieren fact-checking."
        else:
            claims_text = " No se detectaron afirmaciones específicas para verificar."
        
        # Add journalism recommendation
        if score > 0.6:
            recommendation = " Se recomienda priorizar para verificación y seguimiento."
        elif score > 0.3:
            recommendation = " Merece atención y monitorización regular."
        else:
            recommendation = " Contenido de bajo riesgo informativo."
        
        return base_explanation + confidence_text + claims_text + recommendation
    
    def analyze_post(self, 
                    text: str, 
                    retrieve_evidence: bool = True,
                    tweet_url: Optional[str] = None) -> AnalysisResult:
        """
        Comprehensive analysis of a single post.
        """
        if not text or len(text.strip()) < 5:
            return AnalysisResult(
                post_text=text,
                timestamp=datetime.now().isoformat(),
                tweet_url=tweet_url
            )
        
        print(f"\n🔍 Analizando: {text[:100]}...")
        
        # Phase 1: Pattern-based analysis (fast)
        print("📊 Fase 1: Análisis de patrones...")
        
        # Far-right pattern analysis
        far_right_result = self.far_right_analyzer.analyze_text(text)
        
        # Topic classification
        topic_results = self.topic_classifier.classify_topic(text)
        primary_topic = topic_results[0] if topic_results else None
        
        # Claim detection
        claims = self.claim_detector.detect_claims(text)
        high_priority_claims = [
            claim for claim in claims 
            if claim.urgency in [ClaimUrgency.CRITICAL, ClaimUrgency.HIGH]
        ]
        
        # Phase 2: Content analysis
        print("📝 Fase 2: Análisis de contenido...")
        
        # Determine if this needs deeper analysis
        needs_deep_analysis = (
            far_right_result['score'] > 0.3 or
            (primary_topic and primary_topic.category in [
                TopicCategory.EXTREMISM, TopicCategory.CONSPIRACY, 
                TopicCategory.VIOLENCE_THREATS, TopicCategory.HATE_SPEECH
            ]) or
            len(high_priority_claims) > 0 or
            any(claim.verifiability == VerifiabilityLevel.HIGH for claim in claims)
        )
        
        # Extract targeted groups and calls to action
        targeted_groups = self._extract_targeted_groups(text, far_right_result)
        calls_to_action = self._detect_calls_to_action(text, far_right_result)
        
        # Phase 3: Evidence retrieval (if needed)
        evidence = []
        if retrieve_evidence and (needs_deep_analysis or len(claims) > 0):
            print("🔍 Fase 3: Recuperación de evidencia...")
            try:
                claim_type = None
                if claims:
                    claim_type = claims[0].claim_type.value
                evidence = retrieve_evidence_for_post(
                    text, 
                    max_per_source=2,
                    claim_type=claim_type
                )
            except Exception as e:
                print(f"⚠️ Error en recuperación de evidencia: {e}")
        
        # Phase 4: LLM analysis (if available and needed)
        llm_analysis = None
        if self.use_llm and self.llm_pipeline and needs_deep_analysis:
            print("🤖 Fase 4: Análisis LLM...")
            llm_analysis = self._run_llm_analysis(text, far_right_result, claims)
        
        # Phase 5: Risk assessment
        print("⚖️ Fase 5: Evaluación de riesgo...")
        misinformation_risk, misinformation_reason = self._assess_misinformation_risk(
            far_right_result, claims, evidence
        )
        
        fact_check_priority = self._determine_fact_check_priority(
            far_right_result, claims, evidence
        )
        
        # Compile final result
        result = AnalysisResult(
            post_text=text,
            timestamp=datetime.now().isoformat(),
            tweet_url=tweet_url,
            
            # Far-right analysis
            far_right_score=far_right_result['score'],
            threat_level=far_right_result['threat_level'],
            risk_assessment=far_right_result['risk_assessment'],
            pattern_matches=far_right_result['pattern_matches'],
            category_breakdown=far_right_result['category_breakdown'],
            
            # Topic classification
            primary_topic=primary_topic.category.value if primary_topic else "no_político",
            topic_confidence=primary_topic.confidence if primary_topic else 0.0,
            all_topics=[{
                'category': t.category.value,
                'confidence': t.confidence,
                'subcategory': t.subcategory
            } for t in topic_results[:3]],
            
            # Claims
            verifiable_claims=[{
                'text': claim.text,
                'type': claim.claim_type.value,
                'verifiability': claim.verifiability.value,
                'urgency': claim.urgency.value,
                'confidence': claim.confidence,
                'numerical_data': claim.numerical_data,
                'entities': claim.key_entities
            } for claim in claims],
            high_priority_claims=[{
                'text': claim.text,
                'type': claim.claim_type.value,
                'urgency': claim.urgency.value,
                'confidence': claim.confidence
            } for claim in high_priority_claims],
            total_claims=len(claims),
            
            # Content analysis
            misinformation_risk=misinformation_risk,
            misinformation_reason=misinformation_reason,
            calls_to_action=calls_to_action,
            targeted_groups=targeted_groups,
            
            # Evidence
            evidence_retrieved=evidence,
            fact_check_priority=fact_check_priority,
            verification_keywords=self._extract_verification_keywords(text, claims),
            
            # LLM analysis
            llm_analysis=llm_analysis
        )
        
        # Print summary
        self._print_analysis_summary(result)
        
        return result
    
    def _extract_targeted_groups(self, text: str, far_right_result: Dict) -> List[str]:
        """Extract groups that are being targeted in the content."""
        targeted_groups = []
        text_lower = text.lower()
        
        # Common targets in Spanish far-right discourse
        target_patterns = {
            'inmigrantes': ['inmigr', 'extranjeros', 'ilegales', 'invasor'],
            'musulmanes': ['islam', 'muslim', 'moro', 'árabe'],
            'izquierda': ['zurd', 'rojo', 'communist', 'marxist'],
            'gobierno': ['sánchez', 'gobierno', 'psoe', 'socialista'],
            'élites': ['élite', 'soros', 'globalist', 'davos'],
            'medios': ['medios', 'prensa', 'periodist', 'televisión'],
            'lgbtq': ['gay', 'lgbt', 'homosexual', 'trans'],
            'feministas': ['feminazi', 'feminista', 'hembrista']
        }
        
        for group, patterns in target_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                targeted_groups.append(group)
        
        return targeted_groups[:5]  # Limit to top 5
    
    def _detect_calls_to_action(self, text: str, far_right_result: Dict) -> bool:
        """Detect calls to action in the content."""
        action_patterns = [
            r'\b(?:vamos|venid|salid|marchad|concentrad)\b',
            r'\b(?:hay que|debemos|tenemos que)\s+(?:luchar|actuar|resistir)\b',
            r'\b(?:todos? a|todas? a)\s+(?:las calles|manifestarse|protestar)\b',
            r'\b(?:es hora de|momento de)\s+(?:actuar|levantarse|despertar)\b',
            r'\b(?:únete|acompáñanos|súmate)\b',
            r'\b(?:no más|basta|stop)\b',
            r'\b(?:¡despertad!|¡reaccionad!|¡luchad!)\b'
        ]
        
        import re
        text_lower = text.lower()
        
        for pattern in action_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        # Check for imperative mood and urgency markers
        if far_right_result['score'] > 0.5:
            urgency_markers = ['!!', 'urgent', 'ya', 'ahora', 'inmediatamente']
            if any(marker in text_lower for marker in urgency_markers):
                return True
        
        return False
    
    def _assess_misinformation_risk(self, 
                                 far_right_result: Dict, 
                                 claims: List, 
                                 evidence: List[Dict]) -> Tuple[str, str]:
        """Assess the risk of misinformation in the content."""
        risk_score = 0
        reasons = []
        
        # High far-right score increases misinformation risk
        if far_right_result['score'] > 0.7:
            risk_score += 3
            reasons.append("Alto contenido de extrema derecha")
        elif far_right_result['score'] > 0.4:
            risk_score += 2
            reasons.append("Contenido moderado de extrema derecha")
        
        # Conspiracy patterns increase risk
        if 'conspiracy_theories' in far_right_result.get('category_breakdown', {}):
            risk_score += 2
            reasons.append("Teorías conspiratorias detectadas")
        
        # High-urgency unverifiable claims increase risk
        high_urgency_unverifiable = [
            claim for claim in claims 
            if claim.urgency in [ClaimUrgency.CRITICAL, ClaimUrgency.HIGH] and
               claim.verifiability in [VerifiabilityLevel.LOW, VerifiabilityLevel.NONE]
        ]
        if high_urgency_unverifiable:
            risk_score += len(high_urgency_unverifiable)
            reasons.append(f"{len(high_urgency_unverifiable)} afirmaciones urgentes no verificables")
        
        # Lack of credible sources increases risk
        if claims and not evidence:
            risk_score += 1
            reasons.append("Sin fuentes verificables encontradas")
        
        # Evidence of debunking reduces risk
        debunked_evidence = [
            ev for ev in evidence 
            if any(res.get('verdict') == 'debunked' for res in ev.get('results', []))
        ]
        if debunked_evidence:
            risk_score = max(0, risk_score - 2)
            reasons.append("Evidencia de desmentidos encontrada")
        
        # Determine final risk level
        if risk_score >= 5:
            return "CRITICAL", "; ".join(reasons)
        elif risk_score >= 3:
            return "HIGH", "; ".join(reasons)
        elif risk_score >= 1:
            return "MEDIUM", "; ".join(reasons)
        else:
            return "LOW", "Sin indicadores significativos de desinformación"
    
    def _determine_fact_check_priority(self, 
                                     far_right_result: Dict, 
                                     claims: List, 
                                     evidence: List[Dict]) -> str:
        """Determine the priority for fact-checking this content."""
        priority_score = 0
        
        # High threat level increases priority
        if far_right_result['threat_level'] in ['CRITICAL', 'HIGH']:
            priority_score += 3
        
        # Verifiable claims increase priority
        high_verifiability_claims = [
            claim for claim in claims 
            if claim.verifiability == VerifiabilityLevel.HIGH
        ]
        priority_score += len(high_verifiability_claims)
        
        # Critical urgency claims get highest priority
        critical_claims = [
            claim for claim in claims 
            if claim.urgency == ClaimUrgency.CRITICAL
        ]
        priority_score += len(critical_claims) * 2
        
        # Statistical and medical claims are high priority
        priority_claim_types = ['estadística', 'médica', 'científica']
        priority_claims = [
            claim for claim in claims 
            if claim.claim_type.value in priority_claim_types
        ]
        priority_score += len(priority_claims)
        
        if priority_score >= 6:
            return "CRITICAL"
        elif priority_score >= 4:
            return "HIGH"
        elif priority_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _extract_verification_keywords(self, text: str, claims: List) -> List[str]:
        """Extract keywords useful for fact-checking verification."""
        keywords = set()
        
        # Add claim-specific keywords
        for claim in claims:
            keywords.update(claim.verification_keywords)
            keywords.update(claim.key_entities[:3])  # Top entities
        
        # Add general verification terms from text
        import re
        verification_patterns = [
            r'\b(?:según|conforme|datos?|estadísticas?|estudio|informe)\b',
            r'\b(?:gobierno|ministerio|ine|oms|oficial)\b',
            r'\b\d+(?:[.,]\d+)*\s*(?:%|millones?|miles?|euros?)\b'
        ]
        
        text_lower = text.lower()
        for pattern in verification_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            keywords.update(matches)
        
        return list(keywords)[:10]  # Limit to top 10
    
    def _run_llm_analysis(self, text: str, far_right_result: Dict, claims: List) -> Optional[Dict]:
        """Run enhanced LLM analysis if available."""
        if not self.llm_pipeline:
            return None
        
        try:
            # Create analysis context
            analysis_context = {
                'far_right_score': far_right_result.get('score', 0.0),
                'threat_level': far_right_result.get('threat_level', 'LOW'),
                'claims_count': len(claims),
                'category_breakdown': far_right_result.get('category_breakdown', {}),
                'category': far_right_result.get('category', 'general')
            }
            
            # Run enhanced LLM analysis
            llm_result = self.llm_pipeline.analyze_content(text, analysis_context)
            
            # Add traditional analysis compatibility
            result = {
                'political_bias': 'right' if far_right_result.get('score', 0) > 0.5 else 'unknown',
                'bias_confidence': far_right_result.get('score', 0.0),
                'misinformation_indicators': self._extract_misinformation_indicators(text, claims),
                'emotional_language': llm_result.get('llm_sentiment', 'neutral'),
                'credibility_assessment': 'low' if far_right_result.get('score', 0) > 0.6 else 'medium',
                'llm_explanation': llm_result.get('llm_explanation', ''),
                'llm_confidence': llm_result.get('llm_confidence', 0.0),
                'llm_categories': llm_result.get('llm_categories', []),
                'processing_time': llm_result.get('processing_time', 0.0),
                'analysis_type': 'enhanced_pipeline'
            }
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'analysis_type': 'failed'
            }
    
    def _extract_misinformation_indicators(self, text: str, claims: List) -> List[str]:
        """Extract indicators of misinformation from text and claims."""
        indicators = []
        text_lower = text.lower()
        
        # Common misinformation patterns in Spanish
        patterns = [
            'datos oficiales son mentira', 'gobierno oculta', 'medios mainstream',
            'élite globalista', 'estudio secreto', 'la verdad que no quieren',
            'despierta', 'solo nosotros', 'censurado', 'prohibido',
            'no te lo cuentan', 'medios mienten', 'conspiración', 'agenda oculta',
            'nadie habla de', 'silenciado', 'manipulación mediática'
        ]
        
        for pattern in patterns:
            if pattern in text_lower:
                indicators.append(pattern)
        
        # Add claim-based indicators
        for claim in claims:
            if hasattr(claim, 'urgency') and claim.urgency.value in ['critical', 'high']:
                indicators.append(f'claim_urgency_{claim.urgency.value}')
            if hasattr(claim, 'verifiability') and claim.verifiability.value == 'low':
                indicators.append('unverifiable_claim')
        
        return indicators[:5]  # Limit to top 5
    
    def get_system_status(self) -> Dict:
        """Get detailed system status for debugging and monitoring."""
        status = {
            "analyzer_status": "ready",
            "components": {
                "far_right_analyzer": bool(self.far_right_analyzer),
                "topic_classifier": bool(self.topic_classifier),
                "claim_detector": bool(self.claim_detector)
            },
            "llm_status": {
                "enabled": self.use_llm,
                "pipeline_loaded": bool(self.llm_pipeline),
                "models": {}
            },
            "journalism_mode": self.journalism_mode,
            "timestamp": datetime.now().isoformat()
        }
        
        # Get LLM model information if available
        if self.llm_pipeline:
            try:
                status["llm_status"]["models"] = self.llm_pipeline.get_model_info()
            except Exception as e:
                status["llm_status"]["error"] = str(e)
        
        return status
    
    def print_system_status(self):
        """Print detailed system status."""
        status = self.get_system_status()
        
        print("\n🔧 ENHANCED ANALYZER SYSTEM STATUS")
        print("=" * 50)
        print(f"📊 Status: {status['analyzer_status']}")
        print(f"🗞️ Journalism Mode: {'✅ Enabled' if status['journalism_mode'] else '❌ Disabled'}")
        
        print("\n📦 Core Components:")
        for component, loaded in status['components'].items():
            icon = "✅" if loaded else "❌"
            print(f"   {icon} {component.replace('_', ' ').title()}")
        
        print(f"\n🤖 LLM Pipeline:")
        llm_status = status['llm_status']
        print(f"   {'✅' if llm_status['enabled'] else '❌'} LLM Enabled")
        print(f"   {'✅' if llm_status['pipeline_loaded'] else '❌'} Pipeline Loaded")
        
        if llm_status.get('models'):
            models_info = llm_status['models']
            print(f"   🖥️ Device: {models_info.get('device', 'unknown')}")
            print(f"   🎯 Generation Model: {models_info.get('generation_model', 'not loaded')}")
            print(f"   🏷️ Classification Model: {models_info.get('classification_model', 'not loaded')}")
            print(f"   ⚡ Quantization: {'✅' if models_info.get('quantization_enabled') else '❌'}")
        
        if llm_status.get('error'):
            print(f"   ⚠️ Error: {llm_status['error']}")
        
        print(f"\n⏰ Status Time: {status['timestamp']}")
        print("=" * 50)
    
    def cleanup_resources(self):
        """Clean up system resources."""
        if self.llm_pipeline:
            try:
                self.llm_pipeline.cleanup_memory()
            except Exception as e:
                print(f"⚠️ Cleanup error: {e}")
        print("🧹 System resources cleaned up")
    
    def _print_analysis_summary(self, result: AnalysisResult):
        """Print a summary of the analysis results."""
        print(f"\n📋 RESUMEN DEL ANÁLISIS")
        print(f"═══════════════════════════════════════")
        print(f"🎯 Tema principal: {result.primary_topic} ({result.topic_confidence:.2f})")
        print(f"⚠️ Puntuación extrema derecha: {result.far_right_score:.3f}")
        print(f"🚨 Nivel de amenaza: {result.threat_level}")
        print(f"📊 Afirmaciones detectadas: {result.total_claims}")
        print(f"🔍 Prioridad verificación: {result.fact_check_priority}")
        print(f"💥 Riesgo desinformación: {result.misinformation_risk}")
        
        if result.targeted_groups:
            print(f"👥 Grupos objetivo: {', '.join(result.targeted_groups)}")
        
        if result.calls_to_action:
            print(f"📢 Contiene llamadas a la acción: SÍ")
        
        if result.high_priority_claims:
            print(f"🚨 Afirmaciones prioritarias: {len(result.high_priority_claims)}")
        
        print()

# Database functions for journalism workflow
def migrate_database_schema():
    """Migrate existing database to add missing columns."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    c = conn.cursor()
    
    # Check if journalist_analyses table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='journalist_analyses'")
    if not c.fetchone():
        print("📋 Creating journalist_analyses table...")
        init_journalist_analysis_table()
        conn.close()
        return
    
    # Get current columns
    c.execute("PRAGMA table_info(journalist_analyses)")
    columns = [col[1] for col in c.fetchall()]
    
    # Add missing columns
    missing_columns = []
    expected_columns = {
        'journalistic_impact': 'TEXT',
        'evidence_sources': 'TEXT',
        'verification_status': 'TEXT DEFAULT "pending"',
        'targeted_groups': 'TEXT',
        'calls_to_action': 'BOOLEAN',
        'misinformation_risk': 'TEXT',
        'threat_level': 'TEXT'
    }
    
    for col_name, col_type in expected_columns.items():
        if col_name not in columns:
            missing_columns.append((col_name, col_type))
    
    if missing_columns:
        print(f"🔧 Adding {len(missing_columns)} missing columns to database...")
        for col_name, col_type in missing_columns:
            try:
                c.execute(f"ALTER TABLE journalist_analyses ADD COLUMN {col_name} {col_type}")
                print(f"  ✓ Added column: {col_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e):
                    print(f"  ⚠️ Error adding {col_name}: {e}")
    else:
        print("✅ Database schema is up to date")
    
    conn.commit()
    conn.close()

def init_journalist_analysis_table():
    """Initialize journalist analyses table with proper schema."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS journalist_analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tweet_id TEXT UNIQUE,
        tweet_url TEXT,
        username TEXT,
        tweet_content TEXT,
        category TEXT,
        subcategory TEXT,
        confidence REAL,
        far_right_score REAL,
        fact_check_priority TEXT,
        journalistic_impact TEXT,
        llm_explanation TEXT,
        evidence_sources TEXT,
        verification_status TEXT DEFAULT 'pending',
        targeted_groups TEXT,
        calls_to_action BOOLEAN,
        misinformation_risk TEXT,
        threat_level TEXT,
        analysis_json TEXT,
        analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS enhanced_analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tweet_url TEXT,
        analysis_json TEXT,
        far_right_score REAL,
        threat_level TEXT,
        primary_topic TEXT,
        total_claims INTEGER,
        misinformation_risk TEXT,
        fact_check_priority TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

def save_journalist_analysis(analysis: JournalistAnalysis):
    """Save journalist analysis to database with retry logic."""
    import time
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            c = conn.cursor()
            
            c.execute('''
            INSERT OR REPLACE INTO journalist_analyses 
            (tweet_id, tweet_url, username, tweet_content, category, subcategory, 
             confidence, far_right_score, fact_check_priority, journalistic_impact,
             llm_explanation, evidence_sources, verification_status, targeted_groups,
             calls_to_action, misinformation_risk, threat_level, analysis_json, analysis_timestamp) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis.tweet_id, analysis.tweet_url, analysis.username, analysis.tweet_content,
                analysis.category, analysis.subcategory, analysis.confidence, analysis.far_right_score,
                analysis.fact_check_priority, analysis.journalistic_impact, analysis.llm_explanation,
                json.dumps(analysis.evidence_sources, ensure_ascii=False), analysis.verification_status,
                json.dumps(analysis.targeted_groups, ensure_ascii=False), analysis.calls_to_action,
                analysis.misinformation_risk, analysis.threat_level, analysis.analysis_json, 
                analysis.analysis_timestamp
            ))
            
            conn.commit()
            conn.close()
            return  # Success
            
        except sqlite3.OperationalError as e:
            if attempt < max_retries - 1:
                print(f"⚠️ Database locked, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"❌ Database remains locked after {max_retries} attempts: {e}")
                raise

def save_enhanced_analysis(result: AnalysisResult):
    """Save enhanced analysis to database (backward compatibility)."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    c = conn.cursor()
    
    analysis_json = json.dumps(asdict(result), ensure_ascii=False, default=str)
    
    c.execute('''
    INSERT INTO enhanced_analyses 
    (tweet_url, analysis_json, far_right_score, threat_level, primary_topic, 
     total_claims, misinformation_risk, fact_check_priority) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        result.tweet_url or result.post_text[:100],
        analysis_json,
        result.far_right_score,
        result.threat_level,
        result.primary_topic,
        result.total_claims,
        result.misinformation_risk,
        result.fact_check_priority
    ))
    
    conn.commit()
    conn.close()

def create_comprehensive_journalism_examples():
    """Create comprehensive test examples for journalism workflow."""
    examples = [
        # DISINFORMATION EXAMPLES
        {
            'tweet_id': 'disinfo_001',
            'username': 'fake_news_es',
            'content': 'EXCLUSIVO: Las vacunas COVID contienen microchips 5G para controlar la población. Los datos oficiales son mentira del gobierno.',
            'expected_category': 'disinformation',
            'expected_subcategory': 'health_misinformation',
            'expected_priority': 'critical'
        },
        {
            'tweet_id': 'disinfo_002',
            'username': 'bulos_madrid',
            'content': 'El 85% de los inmigrantes no trabajan y cobran ayudas del estado. Estudio secreto que Sánchez oculta. #BuloGobierno',
            'expected_category': 'disinformation',
            'expected_subcategory': 'false_statistics',
            'expected_priority': 'high'
        },
        {
            'tweet_id': 'disinfo_003',
            'username': 'desinforma_tv',
            'content': 'Los medios mainstream ocultan la verdad sobre la agenda globalista. Solo nosotros decimos la realidad.',
            'expected_category': 'disinformation',
            'expected_subcategory': 'general_misinformation',
            'expected_priority': 'medium'
        },
        
        # HATE SPEECH EXAMPLES
        {
            'tweet_id': 'hate_001',
            'username': 'supremacista_es',
            'content': 'Los musulmanes son una raza inferior que viene a destruir nuestra cultura cristiana. España para los españoles.',
            'expected_category': 'hate_speech',
            'expected_subcategory': 'racism',
            'expected_priority': 'critical'
        },
        {
            'tweet_id': 'hate_002',
            'username': 'ultra_patriota',
            'content': 'Hay que eliminar a todos los extranjeros de España. Son como virus que infectan nuestra sangre pura.',
            'expected_category': 'hate_speech',
            'expected_subcategory': 'xenophobia',
            'expected_priority': 'critical'
        },
        {
            'tweet_id': 'hate_003',
            'username': 'racista_vox',
            'content': 'Los gitanos y moros son ladrones por naturaleza. La ciencia lo demuestra, son genéticamente inferiores.',
            'expected_category': 'hate_speech',
            'expected_subcategory': 'racism',
            'expected_priority': 'critical'
        },
        
        # POLITICAL BIAS EXAMPLES
        {
            'tweet_id': 'bias_001',
            'username': 'partidista_extremo',
            'content': 'Los socialistas han destruido España con su agenda marxista. Solo VOX puede salvar la patria de esta invasión comunista.',
            'expected_category': 'political_bias',
            'expected_subcategory': 'extreme_partisanship',
            'expected_priority': 'medium'
        },
        {
            'tweet_id': 'bias_002',
            'username': 'propaganda_nacional',
            'content': 'El régimen de Sánchez es una dictadura comunista. Los patriotas debemos resistir este golpe de estado silencioso.',
            'expected_category': 'political_bias',
            'expected_subcategory': 'antigovernment_propaganda',
            'expected_priority': 'medium'
        },
        {
            'tweet_id': 'bias_003',
            'username': 'ultra_derecha_es',
            'content': 'España necesita un líder fuerte como Franco que ponga orden. La democracia ha fracasado.',
            'expected_category': 'political_bias',
            'expected_subcategory': 'authoritarianism',
            'expected_priority': 'high'
        },
        
        # CONSPIRACY THEORY EXAMPLES
        {
            'tweet_id': 'conspiracy_001',
            'username': 'conspiranoia_es',
            'content': 'El nuevo orden mundial controla todo. Los Illuminati están detrás del cambio climático para reducirnos.',
            'expected_category': 'conspiracy_theory',
            'expected_subcategory': 'global_conspiracy',
            'expected_priority': 'medium'
        },
        {
            'tweet_id': 'conspiracy_002',
            'username': 'soros_hunter',
            'content': 'Soros financia la inmigración para reemplazar a los europeos. Es el plan Kalergi en acción.',
            'expected_category': 'conspiracy_theory',
            'expected_subcategory': 'replacement_theory',
            'expected_priority': 'high'
        },
        {
            'tweet_id': 'conspiracy_003',
            'username': 'deep_state_es',
            'content': 'La élite globalista usa las pandemias para controlarnos. Todo está orquestado desde las sombras.',
            'expected_category': 'conspiracy_theory',
            'expected_subcategory': 'global_conspiracy',
            'expected_priority': 'medium'
        },
        
        # CALL TO ACTION EXAMPLES
        {
            'tweet_id': 'action_001',
            'username': 'organizador_patriota',
            'content': '¡CONCENTRACIÓN HOY 18:00 en Cibeles! Hay que salir a las calles a defender España de la invasión. ¡Todos unidos!',
            'expected_category': 'call_to_action',
            'expected_subcategory': 'urgent_mobilization',
            'expected_priority': 'high'
        },
        {
            'tweet_id': 'action_002',
            'username': 'resistencia_nacional',
            'content': 'Es hora de actuar, patriotas. Organizad manifestaciones en vuestras ciudades. ¡Despertad a los españoles!',
            'expected_category': 'call_to_action',
            'expected_subcategory': 'general_mobilization',
            'expected_priority': 'medium'
        },
        {
            'tweet_id': 'action_003',
            'username': 'viral_patriota',
            'content': 'DIFUNDE este mensaje: España está siendo invadida. Comparte para que todos sepan la verdad. #PatriotasUnidos',
            'expected_category': 'call_to_action',
            'expected_subcategory': 'viral_campaign',
            'expected_priority': 'medium'
        },
        
        # GENERAL EXAMPLES (for comparison)
        {
            'tweet_id': 'general_001',
            'username': 'ciudadano_normal',
            'content': 'Qué día tan bonito hace hoy en Madrid. Me voy a dar un paseo por el Retiro con la familia.',
            'expected_category': 'general',
            'expected_subcategory': None,
            'expected_priority': 'low'
        },
        {
            'tweet_id': 'general_002',
            'username': 'lector_prensa',
            'content': 'Interesante artículo sobre la economía española en El País. Habrá que ver cómo evoluciona la situación.',
            'expected_category': 'general',
            'expected_subcategory': None,
            'expected_priority': 'low'
        },
        {
            'tweet_id': 'general_003',
            'username': 'deportista_es',
            'content': 'Gran partido del Real Madrid anoche. Benzema sigue siendo increíble a su edad. ¡Hala Madrid!',
            'expected_category': 'general',
            'expected_subcategory': None,
            'expected_priority': 'low'
        }
    ]
    
    return examples

def run_comprehensive_journalism_test(analyzer: EnhancedAnalyzer, save_to_db: bool = True):
    """Run comprehensive test with all journalism categories."""
    print("🧪 EJECUTANDO TEST COMPRENSIVO PARA PERIODISMO")
    print("=" * 60)
    
    examples = create_comprehensive_journalism_examples()
    
    if save_to_db:
        migrate_database_schema()  # Ensure schema is up to date
        init_journalist_analysis_table()
    
    results = []
    category_stats = {}
    priority_stats = {}
    
    for i, example in enumerate(examples, 1):
        print(f"\n📄 TEST {i}/{len(examples)}: {example['tweet_id']}")
        print(f"👤 Usuario: @{example['username']}")
        print(f"📝 Contenido: {example['content'][:60]}...")
        
        try:
            # Analyze with enhanced analyzer
            analysis = analyzer.analyze_for_journalism(
                tweet_id=example['tweet_id'],
                tweet_url=f"https://twitter.com/{example['username']}/status/{example['tweet_id']}",
                username=example['username'],
                content=example['content'],
                retrieve_evidence=False  # Skip for speed in test
            )
            
            # Check accuracy
            category_match = analysis.category == example['expected_category']
            priority_match = analysis.fact_check_priority == example['expected_priority']
            
            print(f"🎯 Categoría: {analysis.category} ({'✓' if category_match else '❌'})")
            print(f"📊 Prioridad: {analysis.fact_check_priority} ({'✓' if priority_match else '❌'})")
            print(f"🔢 Confianza: {analysis.confidence:.2f}")
            print(f"⚠️ Puntuación extrema derecha: {analysis.far_right_score:.2f}")
            
            # Save to database
            if save_to_db:
                save_journalist_analysis(analysis)
                print("💾 Guardado en BD ✓")
            
            # Update statistics
            category_stats[analysis.category] = category_stats.get(analysis.category, 0) + 1
            priority_stats[analysis.fact_check_priority] = priority_stats.get(analysis.fact_check_priority, 0) + 1
            
            results.append({
                'tweet_id': example['tweet_id'],
                'category': analysis.category,
                'priority': analysis.fact_check_priority,
                'confidence': analysis.confidence,
                'far_right_score': analysis.far_right_score,
                'category_match': category_match,
                'priority_match': priority_match
            })
            
        except Exception as e:
            print(f"❌ Error en análisis: {e}")
            continue
    
    # Final statistics
    print(f"\n📊 ESTADÍSTICAS FINALES DEL TEST")
    print("=" * 40)
    print(f"Tests ejecutados: {len(results)}")
    
    category_accuracy = sum(1 for r in results if r['category_match']) / len(results) * 100
    priority_accuracy = sum(1 for r in results if r['priority_match']) / len(results) * 100
    
    print(f"Precisión categorías: {category_accuracy:.1f}%")
    print(f"Precisión prioridades: {priority_accuracy:.1f}%")
    
    print(f"\n📈 Distribución por categorías:")
    for category, count in category_stats.items():
        print(f"  {category}: {count}")
    
    print(f"\n🎯 Distribución por prioridades:")
    for priority, count in priority_stats.items():
        print(f"  {priority}: {count}")
    
    return results

def main(posts_list: List[str] = None, 
         skip_retrieval: bool = False, 
         skip_save: bool = False,
         use_llm: bool = True,
         journalism_mode: bool = True,
         test_examples: bool = False,
         model_priority: str = "balanced"):
    """
    Main analysis function with enhanced journalism capabilities.
    """
    if test_examples:
        # Run comprehensive journalism test
        analyzer = EnhancedAnalyzer(use_llm=use_llm, journalism_mode=True, model_priority=model_priority)
        results = run_comprehensive_journalism_test(analyzer, save_to_db=not skip_save)
        
        # Save test results
        try:
            output_path = os.path.join(os.path.dirname(__file__), 'journalism_test_results.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n💾 Resultados del test guardados en: {output_path}")
        except Exception as e:
            print(f"❌ Error guardando resultados del test: {e}")
        
        return results
    
    # Regular analysis mode
    if not skip_save:
        if journalism_mode:
            migrate_database_schema()  # Ensure schema is up to date
            init_journalist_analysis_table()
        else:
            migrate_database_schema()  # Ensure schema is up to date
            init_journalist_analysis_table()  # Use same table for compatibility
    
    # Initialize enhanced analyzer
    analyzer = EnhancedAnalyzer(use_llm=use_llm, journalism_mode=journalism_mode, model_priority=model_priority)
    
    # Use default posts if none provided
    if not posts_list:
        posts_list = [
            "Los inmigrantes ilegales están invadiendo España. Hay que defender nuestras fronteras YA! #EspañaPrimero",
            "Sánchez es un traidor que nos está vendiendo a Soros y la élite globalista. ¡Resistencia!",
            "🔴 URGENTE: Descubren que las vacunas contienen microchips para controlarnos. ¡Despierta España!",
            "Manifestación el domingo contra la dictadura sanitaria. Todos a las calles por la libertad!",
            "La ideología de género está destruyendo a nuestros niños. Hay que proteger la familia tradicional.",
            "¡A las armas españoles! Es hora de la reconquista y la revolución nacional. Viva España!"
        ]
    
    results = []
    total_posts = len(posts_list)
    
    print(f"\n🚀 Iniciando análisis de {total_posts} posts...")
    print("=" * 60)
    
    for i, text in enumerate(posts_list, 1):
        print(f"\n📄 POST {i}/{total_posts}")
        print("-" * 40)
        
        try:
            if journalism_mode:
                # Use journalism analysis
                result = analyzer.analyze_for_journalism(
                    tweet_id=f"post_{i:03d}",
                    tweet_url=f"https://twitter.com/example/status/post_{i:03d}",
                    username=f"user_{i}",
                    content=text,
                    retrieve_evidence=not skip_retrieval
                )
                
                # Save if requested
                if not skip_save:
                    save_journalist_analysis(result)
                
                results.append(asdict(result))
                
            else:
                # Use legacy analysis
                result = analyzer.analyze_post(
                    text=text,
                    retrieve_evidence=not skip_retrieval,
                    tweet_url=f"post_{i}"
                )
                
                # Save if requested
                if not skip_save:
                    save_enhanced_analysis(result)
                
                results.append(asdict(result))
            
        except Exception as e:
            print(f"❌ Error analizando post {i}: {e}")
            continue
    
    # Save all results to JSON file
    try:
        output_path = os.path.join(os.path.dirname(__file__), 'enhanced_analysis_results.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 Resultados guardados en: {output_path}")
    except Exception as e:
        print(f"❌ Error guardando resultados: {e}")
    
    # Print final statistics
    print(f"\n📊 ESTADÍSTICAS FINALES")
    print("=" * 40)
    print(f"Posts analizados: {len(results)}")
    
    if results and journalism_mode:
        high_priority = sum(1 for r in results if r.get('fact_check_priority') in ['high', 'critical'])
        hate_speech = sum(1 for r in results if r.get('category') == 'hate_speech')
        disinformation = sum(1 for r in results if r.get('category') == 'disinformation')
        calls_to_action = sum(1 for r in results if r.get('calls_to_action', False))
        
        print(f"Alta prioridad fact-checking: {high_priority}")
        print(f"Discurso de odio: {hate_speech}")
        print(f"Desinformación: {disinformation}")
        print(f"Llamadas a la acción: {calls_to_action}")
    
    elif results:
        high_risk = sum(1 for r in results if r.get('far_right_score', 0) > 0.6)
        critical_threat = sum(1 for r in results if r.get('threat_level') == 'CRITICAL')
        total_claims = sum(r.get('total_claims', 0) for r in results)
        high_priority_fc = sum(1 for r in results if r.get('fact_check_priority') in ['CRITICAL', 'HIGH'])
        
        print(f"Alto riesgo extrema derecha: {high_risk}")
        print(f"Amenaza crítica: {critical_threat}")
        print(f"Total afirmaciones detectadas: {total_claims}")
        print(f"Alta prioridad fact-checking: {high_priority_fc}")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced Journalism Analyzer for Far-Right Activism')
    parser.add_argument('--skip-retrieval', action='store_true', 
                       help='Skip evidence retrieval to speed up analysis')
    parser.add_argument('--skip-save', action='store_true', 
                       help='Skip saving to database')
    parser.add_argument('--no-llm', action='store_true', 
                       help='Disable LLM analysis')
    parser.add_argument('--fast', action='store_true', 
                       help='Fast mode: skip retrieval, saving, and LLM')
    parser.add_argument('--from-db', action='store_true', 
                       help='Load posts from database')
    parser.add_argument('--limit', type=int, default=50,
                       help='Maximum number of posts to analyze from DB')
    parser.add_argument('--test-examples', action='store_true',
                       help='Run comprehensive test with journalism examples')
    parser.add_argument('--legacy-mode', action='store_true',
                       help='Use legacy analysis mode instead of journalism mode')
    parser.add_argument('--status', action='store_true',
                       help='Show detailed system status and exit')
    parser.add_argument('--model-priority', type=str, default='balanced',
                       choices=['speed', 'balanced', 'quality'],
                       help='Model priority: speed (fast), balanced (default), quality (best)')
    parser.add_argument('--test-llm', action='store_true',
                       help='Test LLM pipeline with sample content and exit')
    
    args = parser.parse_args()
    
    if args.fast:
        args.skip_retrieval = True
        args.skip_save = True
        args.no_llm = True
    
    # Handle status check
    if args.status:
        print("🔧 CHECKING SYSTEM STATUS...")
        analyzer = EnhancedAnalyzer(use_llm=not args.no_llm, model_priority=args.model_priority)
        analyzer.print_system_status()
        if analyzer.llm_pipeline:
            analyzer.cleanup_resources()
        exit(0)
    
    # Handle LLM testing
    if args.test_llm:
        print("🧪 TESTING LLM PIPELINE...")
        analyzer = EnhancedAnalyzer(use_llm=True, model_priority=args.model_priority)
        
        if not analyzer.llm_pipeline:
            print("❌ LLM pipeline not available")
            exit(1)
        
        # Test with sample content
        test_content = "Los inmigrantes están destruyendo nuestro país. Es hora de actuar contra la élite globalista."
        test_context = {
            'far_right_score': 0.8,
            'threat_level': 'HIGH',
            'category': 'hate_speech'
        }
        
        print(f"📝 Test content: {test_content}")
        print("⏳ Running LLM analysis...")
        
        try:
            start_time = time.time()
            result = analyzer.llm_pipeline.analyze_content(test_content, test_context)
            end_time = time.time()
            
            print("✅ LLM Analysis Results:")
            print(f"   💭 Explanation: {result.get('llm_explanation', 'N/A')}")
            print(f"   🎯 Confidence: {result.get('llm_confidence', 0.0):.2f}")
            print(f"   📊 Sentiment: {result.get('llm_sentiment', 'N/A')}")
            print(f"   ⏱️ Processing time: {end_time - start_time:.2f}s")
            
            analyzer.cleanup_resources()
            print("🎯 LLM pipeline test completed successfully!")
            
        except Exception as e:
            print(f"❌ LLM test failed: {e}")
            analyzer.cleanup_resources()
            exit(1)
        
        exit(0)
    
    # Load posts
    posts_to_analyze = []
    
    if args.from_db and not args.test_examples:
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('''
            SELECT content FROM tweets 
            WHERE length(content) > 20 
            ORDER BY created_at DESC 
            LIMIT ?
            ''', (args.limit,))
            rows = c.fetchall()
            conn.close()
            
            posts_to_analyze = [r[0] for r in rows if r and r[0].strip()]
            print(f"📥 Cargados {len(posts_to_analyze)} posts de la base de datos")
            
        except Exception as e:
            print(f"❌ Error cargando posts de BD: {e}")
    
    # Run analysis
    if args.test_examples:
        print("🧪 Ejecutando test comprensivo de ejemplos para periodismo...")
        results = main(
            posts_list=None,
            skip_retrieval=args.skip_retrieval,
            skip_save=args.skip_save,
            use_llm=not args.no_llm,
            journalism_mode=True,
            test_examples=True,
            model_priority=args.model_priority
        )
    else:
        # Use default posts if none loaded from DB
        if not posts_to_analyze:
            posts_to_analyze = None  # Will use defaults in main()
            print(f"📥 Usando posts por defecto del analizador")
        
        results = main(
            posts_to_analyze, 
            skip_retrieval=args.skip_retrieval,
            skip_save=args.skip_save,
            use_llm=not args.no_llm,
            journalism_mode=not args.legacy_mode,
            model_priority=args.model_priority
        )
    
    if not results:
        print("❌ No se pudieron analizar posts")
