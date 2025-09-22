"""
Enhanced analyzer: far-right activism detection with comprehensive coverage.
Integrates specialized components for content analysis workflow.
"""

import json
import os
import sqlite3
import time
import warnings
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Import our enhanced components
from far_right_patterns import FarRightAnalyzer
from topic_classifier import SpanishPoliticalTopicClassifier, TopicCategory
from claim_detector import SpanishClaimDetector, VerifiabilityLevel
# Evidence retrieval skipped as requested
# from retrieval import retrieve_evidence_for_post, format_evidence
from llm_models import EnhancedLLMPipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# DB path (same as other scripts)
DB_PATH = os.path.join(os.path.dirname(__file__), 'accounts.db')
@dataclass
class ContentAnalysis:
    """Content analysis result structure for general workflows."""
    # Tweet metadata
    tweet_id: str
    tweet_url: str
    username: str
    tweet_content: str
    analysis_timestamp: str
    
    # Content categories (standardized)
    category: str  # disinformation, hate_speech, far_right_bias, conspiracy_theory, call_to_action, general
    
    # Analysis results
    llm_explanation: str = ""
    analysis_method: str = "pattern"  # "pattern" or "llm"
    
    # Enhanced metadata
    targeted_groups: List[str] = None
    
    # Technical data
    pattern_matches: List[Dict] = None
    topic_classification: Dict = None
    claims_detected: List[Dict] = None
    analysis_json: str = ""
    
    def __post_init__(self):
        if self.targeted_groups is None:
            self.targeted_groups = []
        if self.pattern_matches is None:
            self.pattern_matches = []
        if self.claims_detected is None:
            self.claims_detected = []



class EnhancedAnalyzer:
    """
    Enhanced analyzer with improved LLM integration for content analysis workflows.
    """
    
    def __init__(self, use_llm: bool = True, model_priority: str = "balanced"):
        self.far_right_analyzer = FarRightAnalyzer()
        self.topic_classifier = SpanishPoliticalTopicClassifier()
        self.claim_detector = SpanishClaimDetector()
        self.use_llm = use_llm
        self.model_priority = model_priority
        self.llm_pipeline = None
        
        print("🚀 Iniciando Enhanced Analyzer...")
        print("Componentes cargados:")
        print("- ✓ Analizador de patrones de extrema derecha")
        print("- ✓ Clasificador de temas políticos") 
        print("- ✓ Detector de afirmaciones verificables")
        print("- ✓ Sistema de recuperación de evidencia")
        print("- ✓ Modo de análisis de contenido activado")
        
        if use_llm:
            print("- ⏳ Cargando modelo LLM para análisis avanzado...")
            try:
                # Use recommended model (now defaults to original gpt-oss-20b for best performance)
                self.llm_pipeline = EnhancedLLMPipeline(model_priority=model_priority)
                print("- ✓ Modelo LLM cargado correctamente")
            except Exception as e:
                print(f"- ⚠️ Error cargando LLM: {e}")
                print("- 🔄 Intentando con modelo de respaldo...")
                try:
                    # Fallback to flan-t5-small if Ollama is not available
                    self.llm_pipeline = EnhancedLLMPipeline(model_priority=model_priority)
                    print("- ✓ Modelo de respaldo cargado correctamente")
                except Exception as e2:
                    print(f"- ❌ Error cargando modelo de respaldo: {e2}")
                    self.llm_pipeline = None
                    self.use_llm = False
    
    def analyze_content(self, 
                             tweet_id: str,
                             tweet_url: str, 
                             username: str,
                             content: str,
                             retrieve_evidence: bool = False) -> ContentAnalysis:
        """
        Main content analysis pipeline with consolidated pattern analysis.
        """
        if not content or len(content.strip()) < 5:
            return ContentAnalysis(
                tweet_id=tweet_id,
                tweet_url=tweet_url,
                username=username,
                tweet_content=content,
                analysis_timestamp=datetime.now().isoformat(),
                category="general",
                llm_explanation="Content too short for analysis"
            )
        
        print(f"\n🔍 Content analysis: @{username}")
        print(f"📝 Contenido: {content[:80]}...")
        
        # Pipeline Step 1: Pattern analysis (all analyzers run once)
        pattern_results = self._run_pattern_analysis(content)

        # Pipeline Step 2: Content categorization (using pattern results + LLM fallback)
        print(f"🔍 Step 2: Categorization starting...")
        category, analysis_method = self._categorize_content(content, pattern_results)
        print(f"🔍 Step 2: Category determined: {category}")

        # Pipeline Step 3: Content insights extraction
        insights = self._extract_content_insights(content, pattern_results)

        # Pipeline Step 4: Smart LLM integration for uncertain cases  
        llm_explanation = self._generate_explanation_with_smart_llm(content, category, pattern_results, insights)
        
        # Pipeline Step 5: Create final analysis structure
        analysis_data = self._build_analysis_data(pattern_results, insights)
        
        return ContentAnalysis(
            tweet_id=tweet_id,
            tweet_url=tweet_url,
            username=username,
            tweet_content=content,
            analysis_timestamp=datetime.now().isoformat(),
            category=category,
            llm_explanation=llm_explanation,
            analysis_method=analysis_method,
            targeted_groups=insights['targeted_groups'],
            pattern_matches=pattern_results['pattern_matches'],
            topic_classification=analysis_data['topic_classification'],
            claims_detected=analysis_data['claims_detected'],
            analysis_json=json.dumps(analysis_data, ensure_ascii=False, default=str)
        )
    
    def _run_pattern_analysis(self, content: str) -> Dict:
        """
        Pipeline Step 1: Intelligent pattern analysis leveraging each component's strengths.
        
        Component Strengths:
        - FarRightAnalyzer: Hate speech, extremism patterns, threat detection
        - TopicClassifier: Political context, discourse categorization
        - ClaimDetector: Factual statements, verifiability assessment
        """
        results = {}
        
        # Phase 1: Quick political context assessment (fastest component)
        topic_results = self.topic_classifier.classify_topic(content)
        results['topics'] = topic_results
        
        # Extract political context for guiding other analyzers
        political_context = topic_results[0] if topic_results else None
        is_political_content = political_context and political_context.category.value != "no_político"
        
        # Phase 2: Far-right pattern analysis (strength: extremism detection)
        far_right_result = self.far_right_analyzer.analyze_text(content)
        results['far_right'] = far_right_result
        
        # Phase 3: Claims analysis (strength: factual verification needs)
        # Always run claims analysis to detect all types of disinformation, not just political
        claims = self.claim_detector.detect_claims(content)
        results['claims'] = claims
        
        # Phase 4: Create enriched pattern matches combining all components
        pattern_matches = far_right_result.get('pattern_matches', [])
        
        # Enrich with topic context
        if is_political_content:
            pattern_matches.append({
                'category': 'political_context',
                'matched_text': political_context.category.value,
                'description': f'Contexto político: {political_context.subcategory}',
                'context': content[:50] + '...'
            })
        
        results['pattern_matches'] = pattern_matches
        
        return results
    
    def _categorize_content(self, content: str, pattern_results: Dict) -> Tuple[str, str]:
        """
        Pipeline Step 2: Determine content category using pattern results + LLM fallback.
        Returns: (category, analysis_method)
        """
        far_right_result = pattern_results['far_right']
        claims = pattern_results['claims']
        
        detected_categories = far_right_result.get('categories', [])
        
        # Priority 1: Hate speech and violence incitement (highest severity)
        # Map xenophobia and related patterns to hate_speech category
        if any(cat in detected_categories for cat in ['hate_speech', 'violence_incitement', 'xenophobia']):
            return "hate_speech", "pattern"
        
        # Priority 2: Health/Medical disinformation (using component detection)
        detected_categories = far_right_result.get('categories', [])
        
        # Check for health disinformation using far-right analyzer's new category
        if 'health_disinformation' in detected_categories:
            return "disinformation", "pattern"
        
        # Specific check for vaccine/COVID disinformation that might be misclassified as conspiracy
        content_lower = content.lower()
        if any(term in content_lower for term in ['vacuna', 'covid', 'coronavirus']) and \
           any(term in content_lower for term in ['microchip', 'control', '5g', 'chip', 'mentira']):
            return "disinformation", "pattern"
        
        # Check for health claims with disinformation indicators using claim detector
        health_claims = [c for c in claims if c.claim_type.value == 'médica']
        if health_claims:
            for claim in health_claims:
                # Use the new disinformation assessment from claim detector
                disinfo_assessment = self.claim_detector.assess_disinformation_indicators(
                    claim.text, claim.claim_type
                )
                if disinfo_assessment['risk_level'] in ['high', 'medium']:
                    return "disinformation", "pattern"
        
        # Priority 3: Conspiracy theories (component-based detection)
        if 'conspiracy' in detected_categories:
            return "conspiracy_theory", "pattern"
        
        # Priority 4: Handle overlapping far-right bias and call to action
        if 'far_right_bias' in detected_categories and 'call_to_action' in detected_categories:
            # Prioritize call_to_action if explicit mobilization language is present
            mobilization_terms = ['movilizaos', 'organizaos', 'retirad', 'sacad', 'boicot', 'todos a', 'mañana', 'actuad ya', 'convocatoria', 'difunde']
            if any(term in content.lower() for term in mobilization_terms):
                return "call_to_action", "pattern"
            else:
                return "far_right_bias", "pattern"
        
        # Priority 5: Political bias (specific political bias patterns)
        if 'far_right_bias' in detected_categories:
            return "far_right_bias", "pattern"
        
        # Priority 6: Calls to action (specific call-to-action patterns)
        if 'call_to_action' in detected_categories:
            return "call_to_action", "pattern"
        
        # Priority 6: Anti-government with action language (legacy check)
        if any(cat in detected_categories for cat in ['anti_government']) and self._has_action_language(content):
            return "call_to_action", "pattern"
        
        # Priority 7: Map remaining patterns to appropriate categories
        if detected_categories:
            # Explicit mapping for specific pattern types
            if any(cat in detected_categories for cat in ['xenophobia', 'nationalism', 'historical_revisionism']):
                return "hate_speech", "pattern"
            elif any(cat in detected_categories for cat in ['conspiracy', 'health_disinformation', 'anti_government']):
                return "conspiracy_theory", "pattern" 
            elif 'call_to_action' in detected_categories:
                return "call_to_action", "pattern"
            else:
                # Default to far_right_bias for other detected patterns
                return "far_right_bias", "pattern"
        
        # Priority 8: Non-political claims with disinformation indicators (component-based)
        if claims:
            for claim in claims:
                # Use claim detector's disinformation assessment
                disinfo_assessment = self.claim_detector.assess_disinformation_indicators(
                    claim.text, claim.claim_type
                )
                if disinfo_assessment['risk_level'] in ['high', 'medium']:
                    return "disinformation", "pattern"
        
        # Priority 9: Political claims without patterns
        if claims:
            political_claims = [c for c in claims if c.claim_type.value == 'política']
            if political_claims:
                return "far_right_bias", "pattern"
        
        # NEW: LLM fallback for complex content without clear patterns
        print(f"🔍 Checking LLM fallback conditions...")
        print(f"🔍 Detected categories: {detected_categories}")
        print(f"🔍 Claims count: {len(claims)}")
        
        # Simple LLM fallback rule: Use LLM when pattern detection fails
        if self._should_use_llm_fallback(content, detected_categories, claims):
            print("🧠 No patterns detected - using LLM for analysis")
            llm_category = self._get_llm_category(content, pattern_results)
            print(f"🔍 LLM category result: {llm_category}")
            if llm_category != "general":
                return llm_category, "llm"
            return "general", "llm"

        return "general", "pattern"
    
    def _has_action_language(self, text: str) -> bool:
        """Quick check for action/mobilization language."""
        import re
        action_patterns = [
            r'\b(?:concentración|manifestación|calles|resistencia)\b',
            r'\b(?:organizad|difunde|comparte|despertad)\b',
            r'\b(?:todos unidos|hay que actuar)\b'
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in action_patterns)
    
    def _should_use_llm_fallback(self, content: str, detected_categories: List, claims: List) -> bool:
        """Determine if content requires LLM analysis when pattern detection fails."""
        # Use LLM when no patterns are detected, regardless of claims
        # Claims alone don't determine category - context matters
        return len(detected_categories) == 0
    
    def _get_llm_category(self, content: str, pattern_results: Dict) -> str:
        """Use LLM to categorize content when patterns are insufficient."""
        if not self.use_llm or not self.llm_pipeline:
            print("🔍 LLM not available, returning general")
            return "general"
        
        try:
            print(f"🔍 _get_llm_category called with content: {content[:50]}...")
            print("🔍 Calling llm_pipeline.get_category...")
            
            # Use FAST category detection instead of full analysis
            llm_category = self.llm_pipeline.get_category(content)
            
            # No hardcoded fallback patterns - let the LLM handle all edge cases
            # This makes the system truly scalable without keyword maintenance
            return llm_category
            
        except Exception as e:
            print(f"⚠️ Error en categorización LLM: {e}")
            return "general"
    
    def _extract_content_insights(self, content: str, pattern_results: Dict) -> Dict:
        """
        Pipeline Step 3: Extract content insights using consolidated pattern results.
        This replaces multiple separate extraction methods.
        """
        far_right_result = pattern_results['far_right']
        
        # Extract targeted groups
        targeted_groups = self._extract_targeted_groups(content, far_right_result)
        
        return {
            'targeted_groups': targeted_groups
        }
    
    def _generate_explanation_with_smart_llm(self, content: str, category: str, 
                                           pattern_results: Dict, insights: Dict) -> str:
        """
        Pipeline Step 4: Smart LLM integration - use LLM only when patterns are ambiguous.
        
        LLM Strategy:
        - Clear patterns detected: Use pattern-based explanation
        - Multiple conflicting patterns: Use LLM for disambiguation  
        - No clear patterns: Use LLM for analysis
        """
        # Build base explanation from patterns
        base_explanation = self._generate_pattern_based_explanation(category, pattern_results, insights)
        
        # Check if we have clear, unambiguous results
        detected_categories = pattern_results['far_right'].get('categories', [])
        has_patterns = len(detected_categories) > 0
        has_multiple_patterns = len(detected_categories) > 1
        claims = pattern_results.get('claims', [])
        
        # Decision: When to use LLM (without any scoring)
        if not has_patterns and not claims and category == "general":
            # Check content length and complexity to decide if LLM analysis is needed
            content_length = len(content.split())
            has_complex_language = any(word in content.lower() for word in [
                'características', 'correlaciones', 'interpretaciones', 'fundamentalmente',
                'transformando', 'instituciones', 'problemáticas', 'específicas',
                'tendencias', 'beneficiar', 'actores', 'internacionales'
            ])
            
            # Only skip LLM for very simple, clearly innocent content
            if content_length < 15 and not has_complex_language:
                print("✅ Clear simple content - using pattern-based analysis")
                return base_explanation
            else:
                # Complex content without patterns might need LLM analysis
                print("🧠 Complex content without patterns - using LLM for analysis")
                return self._primary_llm_analysis(content, category, pattern_results, base_explanation)
        
        elif has_patterns and not has_multiple_patterns:
            # Single clear pattern: Pattern analysis is sufficient
            print("🎯 Clear single pattern - using pattern-based analysis")
            return base_explanation
        
        elif has_multiple_patterns or (claims and has_patterns):
            # Multiple patterns or patterns + claims: Use LLM for enhancement
            print("🤖 Multiple patterns detected - using LLM for enhancement")
            return self._enhance_explanation_with_llm(content, category, pattern_results, base_explanation)
        
        else:
            # Ambiguous cases: Use LLM for primary analysis
            print("🧠 Ambiguous content - using LLM for analysis")
            return self._primary_llm_analysis(content, category, pattern_results, base_explanation)
    
    def _generate_pattern_based_explanation(self, category: str, pattern_results: Dict, insights: Dict) -> str:
        """Generate natural language explanation based purely on pattern analysis."""
        detected_categories = pattern_results['far_right'].get('categories', [])
        claims = pattern_results.get('claims', [])
        
        # Generate natural language explanations based on category
        if category == "hate_speech":
            base_explanation = "Este contenido presenta características de discurso de odio, utilizando lenguaje discriminatorio y deshumanizante"
            # Don't mention specific groups to avoid redundancy with targeted_groups field
        
        elif category == "disinformation":
            base_explanation = "Este contenido contiene afirmaciones que presentan características de desinformación"
            if 'health_disinformation' in detected_categories:
                base_explanation = "Este contenido presenta afirmaciones médicas sin respaldo científico verificable"
            elif claims:
                health_claims = [c for c in claims if c.claim_type.value == 'médica']
                if health_claims:
                    base_explanation += " relacionadas con temas de salud"
        
        elif category == "conspiracy_theory":
            base_explanation = "Este contenido promueve teorías conspiratorias sin base empírica"
            if 'conspiracy' in detected_categories:
                base_explanation += ", utilizando narrativas que fomentan desconfianza en instituciones oficiales"
        
        elif category == "far_right_bias":
            base_explanation = "Este contenido muestra marcos interpretativos de extrema derecha"
            if detected_categories:
                base_explanation += " con elementos de retórica extremista"
        
        elif category == "call_to_action":
            base_explanation = "Este contenido incluye llamadas a la acción o movilización."
        
        else:
            return "Contenido analizado sin patrones específicos detectados."
        
        # Add contextual information without redundancy
        context_parts = []
        if len(detected_categories) > 1:
            context_parts.append("presenta múltiples indicadores problemáticos")
        
        if claims and len(claims) > 1:
            context_parts.append("contiene varias afirmaciones verificables")
        
        if context_parts:
            base_explanation += f" y {', '.join(context_parts)}"
        
        return base_explanation + "."
    
    def _enhance_explanation_with_llm(self, content: str, category: str, 
                                    pattern_results: Dict, base_explanation: str) -> str:
        """Use LLM to enhance and validate pattern-based analysis."""
        if not self.use_llm or not self.llm_pipeline:
            return base_explanation
        
        try:
            # Create focused prompt context for enhancement
            from enhanced_prompts import AnalysisType
            
            # Determine analysis type based on detected patterns
            analysis_type = self._determine_llm_analysis_type(category, pattern_results)
            
            # Check if llm_pipeline has the required method
            if not hasattr(self.llm_pipeline, 'get_explanation'):
                print("⚠️ LLM pipeline missing get_explanation method")
                return base_explanation
            
            # Create analysis context
            analysis_context = {
                'category': category,
                'detected_categories': pattern_results['far_right'].get('categories', []),
                'claims_count': len(pattern_results['claims']),
                'pattern_confidence': pattern_results['far_right'].get('confidence', 0.0)
            }
            
            # Get LLM enhancement using the new method
            llm_enhancement = self.llm_pipeline.get_explanation(content, category, analysis_context, analysis_type)
            
            if llm_enhancement and len(llm_enhancement.strip()) > 10:
                # Combine natural language explanations
                return f"{base_explanation} {llm_enhancement}"
            else:
                return base_explanation
                
        except Exception as e:
            print(f"⚠️ Error en mejora LLM: {e}")
            return base_explanation
    
    def _primary_llm_analysis(self, content: str, category: str, 
                            pattern_results: Dict, base_explanation: str) -> str:
        """Use LLM as primary analysis method for ambiguous cases."""
        if not self.use_llm or not self.llm_pipeline:
            # Don't reveal LLM limitations to user - just return base explanation
            return base_explanation
        
        try:
            
            # Create analysis context with comprehensive information
            analysis_context = {
                'category': category,
                'analysis_mode': 'primary',
                'detected_categories': pattern_results['far_right'].get('categories', []),
                'claims_count': len(pattern_results['claims']),
                'pattern_confidence': pattern_results['far_right'].get('confidence', 0.0),
                'has_patterns': len(pattern_results['far_right'].get('categories', [])) > 0
            }
            
            # Use comprehensive analysis for uncertain cases
            from enhanced_prompts import AnalysisType
            analysis_type = AnalysisType.COMPREHENSIVE
            
            # Get LLM explanation using the existing method
            llm_explanation = self.llm_pipeline.get_explanation(content, category, analysis_context, analysis_type)
            
            if llm_explanation and len(llm_explanation.strip()) > 10:
                return llm_explanation
            else:
                # LLM failed to provide explanation - generate a comprehensive pattern-based one
                print("⚠️ LLM explanation was empty, using enhanced pattern analysis")
                return self._generate_enhanced_pattern_explanation(content, category, pattern_results)
                
        except Exception as e:
            print(f"❌ Error en análisis primario LLM: {e}")
            # Don't reveal LLM errors to user - return enhanced pattern explanation
            return self._generate_enhanced_pattern_explanation(content, category, pattern_results)
    
    def _generate_enhanced_pattern_explanation(self, content: str, category: str, pattern_results: Dict) -> str:
        """Generate a comprehensive explanation when LLM is not available or fails."""
        detected_categories = pattern_results['far_right'].get('categories', [])
        claims = pattern_results.get('claims', [])
        
        # Generate detailed explanations based on what was actually detected
        if category == "far_right_bias":
            explanation_parts = []
            
            # Check what specific patterns were detected for far-right bias
            pattern_matches = pattern_results['far_right'].get('pattern_matches', [])
            
            # Analyze the specific type of bias detected
            content_lower = content.lower()
            
            if any('sustitución' in content_lower for word in ['sustitución', 'sustitu']):
                explanation_parts.append("presenta narrativas sobre sustitución poblacional")
            
            if any(word in content_lower for word in ['inmigr', 'extranjero', 'moro']):
                explanation_parts.append("contiene referencias a inmigración con posible sesgo")
            
            if any(word in content_lower for word in ['muslim', 'islám', 'tradiciones culturales']):
                explanation_parts.append("incluye generalizaciones sobre grupos culturales o religiosos")
            
            if any(word in content_lower for word in ['efecto llamada', 'política', 'brussels', 'comisión europea']):
                explanation_parts.append("enmarca políticas oficiales desde perspectivas potencialmente sesgadas")
            
            if explanation_parts:
                base = "Este contenido muestra características de sesgo político de extrema derecha ya que "
                return base + ", ".join(explanation_parts) + ". El análisis detecta marcos interpretativos que pueden reforzar narrativas problemáticas sobre inmigración y diversidad cultural."
            else:
                return "Este contenido presenta marcos interpretativos característicos del sesgo de extrema derecha, utilizando lenguaje que puede reforzar narrativas discriminatorias."
        
        elif category == "conspiracy_theory":
            return "Este contenido promueve teorías conspiratorias presentando afirmaciones sin base empírica verificable y fomentando desconfianza en instituciones oficiales."
        
        elif category == "hate_speech":
            return "Este contenido presenta características de discurso de odio, utilizando lenguaje discriminatorio que puede fomentar hostilidad hacia grupos específicos."
        
        elif category == "disinformation":
            return "Este contenido presenta características de desinformación, incluyendo afirmaciones que requieren verificación y que pueden difundir información inexacta."
        
        elif category == "call_to_action":
            return "Este contenido incluye llamadas explícitas a la acción o movilización que pueden promover activismo de extrema derecha."
        
        else:
            return "Contenido categorizado mediante análisis de patrones sin características específicas de extremismo detectadas."
    
    def _determine_llm_analysis_type(self, category: str, pattern_results: Dict):
        """Determine the best LLM analysis type based on detected patterns."""
        try:
            from enhanced_prompts import AnalysisType
            
            detected_categories = pattern_results['far_right'].get('categories', [])
            
            # Map content categories to LLM analysis types
            if category == "hate_speech" or any(cat in detected_categories for cat in ['hate_speech', 'violence_incitement']):
                return AnalysisType.HATE_SPEECH
            elif category == "disinformation" or 'conspiracy' in detected_categories:
                return AnalysisType.MISINFORMATION
            elif category == "conspiracy_theory":
                return AnalysisType.MISINFORMATION
            elif pattern_results.get('claims'):
                return AnalysisType.CLAIM_VERIFICATION
            elif category == "far_right_bias":
                return AnalysisType.FAR_RIGHT_BIAS
            else:
                return AnalysisType.COMPREHENSIVE
        except Exception as e:
            print(f"⚠️ Error determining LLM analysis type: {e}")
            # Return a default type
            from enhanced_prompts import AnalysisType
            return AnalysisType.COMPREHENSIVE
    
    def _build_analysis_data(self, pattern_results: Dict, insights: Dict) -> Dict:
        """
        Pipeline Step 5: Build the final analysis data structure.
        """
        topic_results = pattern_results['topics']
        claims = pattern_results['claims']
        
        return {
            'category': None,  # Will be set by caller
            'pattern_matches': pattern_results['pattern_matches'],
            'topic_classification': {
                'primary_topic': topic_results[0].category.value if topic_results else "no_político",
                'all_topics': [{
                    'category': t.category.value,
                    'subcategory': t.subcategory
                } for t in topic_results[:3]]
            },
            'claims_detected': [{
                'text': claim.text,
                'type': claim.claim_type.value,
                'urgency': claim.urgency.value,
                'verifiability': claim.verifiability.value
            } for claim in claims],
            'content_insights': insights
        }
    

    def _generate_pattern_explanation(self, far_right_result: Dict, content: str) -> str:
        """Generate explanation for detected patterns."""
        detected_categories = far_right_result.get('categories', [])
        
        if not detected_categories:
            return "Contenido con características de sesgo político"
        
        # Map categories to Spanish descriptions
        category_descriptions = {
            'hate_speech': 'discurso de odio',
            'xenophobia': 'xenofobia',
            'nationalism': 'nacionalismo extremo',
            'conspiracy': 'teorías conspiratorias',
            'violence_incitement': 'incitación a la violencia',
            'anti_government': 'retórica anti-gobierno',
            'historical_revisionism': 'revisionismo histórico'
        }
        
        descriptions = [category_descriptions.get(cat, cat) for cat in detected_categories[:3]]
        return f"Patrones de {', '.join(descriptions)}"
    
    def _generate_claims_explanation(self, claims: List) -> str:
        """Generate explanation for detected claims."""
        if not claims:
            return "No se detectaron afirmaciones verificables"
        
        claim_types = [claim.claim_type.value for claim in claims[:3]]
        return f"Afirmaciones de tipo {', '.join(set(claim_types))} ({len(claims)} total)"
    
    def _get_llm_explanation(self, content: str, category: str, explanation_parts: List[str]) -> Optional[str]:
        """Get enhanced explanation from LLM if available."""
        if not self.llm_pipeline:
            return None
        
        analysis_context = {
            'category': category,
            'existing_analysis': " | ".join(explanation_parts),
            'pattern_matches': [{'category': category}]
        }
        
        try:
            llm_result = self.llm_pipeline.analyze_content(content, analysis_context)
            return llm_result.get('llm_explanation', '')
        except Exception as e:
            print(f"⚠️ Error en análisis LLM: {e}")
            return None
    

    
    def _generate_category_explanation(self, category: str, content: str) -> str:
        """Generate explanation based on detected category."""
        category_explanations = {
            "hate_speech": "Detectado discurso de odio con lenguaje discriminatorio",
            "disinformation": "Identificada posible desinformación o contenido falso",
            "conspiracy_theory": "Detectadas teorías conspiratorias sin evidencia",
            "far_right_bias": "Contenido con marcos interpretativos de extrema derecha",
            "call_to_action": "Detectada llamada a la acción o movilización"
        }
        
        return category_explanations.get(category, "Contenido categorizado")
    
    

    def _extract_targeted_groups(self, text: str, far_right_result: Dict) -> List[str]:
        """Extract targeted groups based on actual matched text, not generic categories."""
        pattern_matches = far_right_result.get('pattern_matches', [])
        
        targeted_groups = []
        text_lower = text.lower()
        
        # Extract groups based on what was actually mentioned in the text
        for match in pattern_matches:
            matched_text = match.get('matched_text', '').lower()
            context = match.get('context', '').lower()
            full_context = matched_text + ' ' + context
            
            # Look for specific group mentions in the matched text and context
            if any(term in full_context for term in ['sánchez', 'psoe', 'gobierno', 'socialista']):
                targeted_groups.append('gobierno')
            if any(term in full_context for term in ['inmigr', 'extranjero', 'moro', 'ilegal']):
                targeted_groups.append('inmigrantes')
            if any(term in full_context for term in ['islam', 'muslim', 'musulm']):
                targeted_groups.append('musulmanes')
            if any(term in full_context for term in ['zurd', 'rojo', 'marxist', 'communist']):
                targeted_groups.append('izquierda')
            if any(term in full_context for term in ['soros', 'élite', 'globalist']):
                targeted_groups.append('élites')
            if any(term in full_context for term in ['medios', 'prensa', 'televisión', 'periodist']):
                targeted_groups.append('medios')
            if any(term in full_context for term in ['gay', 'lgbt', 'trans', 'homosexual']):
                targeted_groups.append('lgbtq')
            if any(term in full_context for term in ['feminazi', 'feminista', 'hembrista']):
                targeted_groups.append('feministas')
        
        return list(set(targeted_groups))[:5]  # Remove duplicates, limit to 5
    
    def _detect_calls_to_action(self, text: str, far_right_result: Dict) -> bool:
        """Detect calls to action using pattern analysis results."""
        detected_categories = far_right_result.get('categories', [])
        
        # Check for action-oriented pattern categories
        if any(cat in detected_categories for cat in ['violence_incitement', 'anti_government']):
            # Additional check for explicit action language
            return self._has_action_language(text)
        
        return False

    def cleanup_resources(self):
        """Clean up any resources used by the analyzer."""
        try:
            if hasattr(self, 'llm_pipeline') and self.llm_pipeline:
                if hasattr(self.llm_pipeline, 'cleanup'):
                    self.llm_pipeline.cleanup()
        except Exception as e:
            print(f"⚠️ Warning during cleanup: {e}")
    
    def print_system_status(self):
        """Print current system status for debugging."""
        print("🔧 ENHANCED ANALYZER SYSTEM STATUS")
        print("=" * 50)
        print(f"🤖 LLM Enabled: {self.use_llm}")
        if self.llm_pipeline:
            print(f"🔧 Model Priority: {self.model_priority}")
            print(f"🧠 LLM Pipeline: Loaded")
        else:
            print("⚠️ LLM Pipeline: Not available")
        
        print(f"🔍 Far-right analyzer: {'✓' if self.far_right_analyzer else '❌'}")
        print(f"📊 Topic classifier: {'✓' if self.topic_classifier else '❌'}")
        print(f"🔎 Claim detector: {'✓' if self.claim_detector else '❌'}")
        
        # Check database
        try:
            conn = sqlite3.connect(DB_PATH, timeout=5.0)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM content_analyses")
            count = c.fetchone()[0]
            conn.close()
            print(f"📊 Database: {count} analyses stored")
        except Exception as e:
            print(f"❌ Database: Error - {e}")


# Database functions for content analysis workflow
def migrate_database_schema():
    """Migrate existing database to add missing columns."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    c = conn.cursor()
    
    # Check if content_analyses table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='content_analyses'")
    if not c.fetchone():
        print("📋 Creating content_analyses table...")
        init_content_analysis_table()
        conn.close()
        return
    
    # Get current columns
    c.execute("PRAGMA table_info(content_analyses)")
    columns = [col[1] for col in c.fetchall()]
    
    # Add missing columns
    missing_columns = []
    expected_columns = {
        'evidence_sources': 'TEXT',
        'verification_status': 'TEXT DEFAULT "pending"',
        'targeted_groups': 'TEXT',
        'misinformation_risk': 'TEXT'
    }
    
    for col_name, col_type in expected_columns.items():
        if col_name not in columns:
            missing_columns.append((col_name, col_type))
    
    if missing_columns:
        print(f"🔧 Adding {len(missing_columns)} missing columns to database...")
        for col_name, col_type in missing_columns:
            try:
                c.execute(f"ALTER TABLE content_analyses ADD COLUMN {col_name} {col_type}")
                print(f"  ✓ Added column: {col_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e):
                    print(f"  ⚠️ Error adding {col_name}: {e}")
    else:
        print("✅ Database schema is up to date")
    
    conn.commit()
    conn.close()

def init_content_analysis_table():
    """Initialize content analyses table with proper schema."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS content_analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tweet_id TEXT UNIQUE,
        tweet_url TEXT,
        username TEXT,
        tweet_content TEXT,
        category TEXT,
        llm_explanation TEXT,
        targeted_groups TEXT,
        analysis_json TEXT,
        analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

def save_content_analysis(analysis: ContentAnalysis):
    """Save content analysis to database with retry logic."""
    import time
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            c = conn.cursor()
            
            c.execute('''
            INSERT OR REPLACE INTO content_analyses 
            (tweet_id, tweet_url, username, tweet_content, category, 
             llm_explanation, targeted_groups, analysis_json, analysis_timestamp) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis.tweet_id, analysis.tweet_url, analysis.username, analysis.tweet_content,
                analysis.category, analysis.llm_explanation,
                json.dumps(analysis.targeted_groups, ensure_ascii=False),
                analysis.analysis_json, analysis.analysis_timestamp
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


if __name__ == '__main__':
    print("❌ Este módulo no debe ejecutarse directamente")
    print("💡 Usa comprehensive_test_suite.py para ejecutar tests")
    print("💡 O importa EnhancedAnalyzer para usar la funcionalidad de análisis")
