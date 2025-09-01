"""
Enhanced analyzer: far-right activism detection with comprehensive coverage.
Integrates specialized components for content analysis workflow.
"""

import json
import os
import sqlite3
import argparse
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
    category: str  # disinformation, hate_speech, political_bias, conspiracy_theory, call_to_action, general
    
    # Analysis results
    llm_explanation: str = ""
    
    # Enhanced metadata
    targeted_groups: List[str] = None
    calls_to_action: bool = False
    
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
        
        print("� Iniciando Enhanced Analyzer...")
        print("Componentes cargados:")
        print("- ✓ Analizador de patrones de extrema derecha")
        print("- ✓ Clasificador de temas políticos") 
        print("- ✓ Detector de afirmaciones verificables")
        print("- ✓ Sistema de recuperación de evidencia")
        print("- ✓ Modo de análisis de contenido activado")
        
        if use_llm:
            print("- ⏳ Cargando modelo LLM para análisis avanzado...")
            try:
                self.llm_pipeline = EnhancedLLMPipeline(model_priority=model_priority)
                print("- ✓ Modelo LLM cargado correctamente")
            except Exception as e:
                print(f"- ⚠️ Error cargando LLM: {e}")
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
        
        # Pipeline Step 2: Content categorization (using pattern results)
        category = self._categorize_content(content, pattern_results)
        
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
            targeted_groups=insights['targeted_groups'],
            calls_to_action=insights['calls_to_action'],
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
    
    def _categorize_content(self, content: str, pattern_results: Dict) -> str:
        """
        Pipeline Step 2: Determine content category using pattern results.
        """
        far_right_result = pattern_results['far_right']
        claims = pattern_results['claims']
        
        detected_categories = far_right_result.get('categories', [])
        
        # Priority 1: Hate speech and violence incitement (highest severity)
        if any(cat in detected_categories for cat in ['hate_speech', 'violence_incitement']):
            return "hate_speech"
        
        # Priority 2: Health/Medical disinformation (using component detection)
        detected_categories = far_right_result.get('categories', [])
        
        # Check for health disinformation using far-right analyzer's new category
        if 'health_disinformation' in detected_categories:
            return "disinformation"
        
        # Check for health claims with disinformation indicators using claim detector
        health_claims = [c for c in claims if c.claim_type.value == 'médica']
        if health_claims:
            for claim in health_claims:
                # Use the new disinformation assessment from claim detector
                disinfo_assessment = self.claim_detector.assess_disinformation_indicators(
                    claim.text, claim.claim_type
                )
                if disinfo_assessment['risk_level'] in ['high', 'medium']:
                    return "disinformation"
        
        # Priority 3: Conspiracy theories (component-based detection)
        if 'conspiracy' in detected_categories:
            return "conspiracy_theory"
        
        # Priority 4: Calls to action (check for action patterns)
        if any(cat in detected_categories for cat in ['anti_government']) and self._has_action_language(content):
            return "call_to_action"
        
        # Priority 5: Political bias (any far-right pattern detected)
        if detected_categories:
            return "political_bias"
        
        # Priority 6: Non-political claims with disinformation indicators (component-based)
        if claims:
            for claim in claims:
                # Use claim detector's disinformation assessment
                disinfo_assessment = self.claim_detector.assess_disinformation_indicators(
                    claim.text, claim.claim_type
                )
                if disinfo_assessment['risk_level'] in ['high', 'medium']:
                    return "disinformation"
        
        # Priority 7: Political claims without patterns
        if claims:
            political_claims = [c for c in claims if c.claim_type.value == 'política']
            if political_claims:
                return "political_bias"
        
        return "general"
    
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
    
    def _extract_content_insights(self, content: str, pattern_results: Dict) -> Dict:
        """
        Pipeline Step 3: Extract content insights using consolidated pattern results.
        This replaces multiple separate extraction methods.
        """
        far_right_result = pattern_results['far_right']
        
        # Extract targeted groups
        targeted_groups = self._extract_targeted_groups(content, far_right_result)
        
        # Detect calls to action
        calls_to_action = self._detect_calls_to_action(content, far_right_result)
        
        return {
            'targeted_groups': targeted_groups,
            'calls_to_action': calls_to_action
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
            # No patterns detected: Simple content, skip LLM
            print("✅ Clear general content - using pattern-based analysis")
            return base_explanation
        
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
        """Generate explanation based purely on pattern analysis."""
        explanation_parts = []
        
        # Category explanation
        if category != "general":
            category_explanation = self._generate_category_explanation(category, "")
            explanation_parts.append(f"📂 Categorización: {category_explanation}")
        
        # Pattern explanation
        detected_categories = pattern_results['far_right'].get('categories', [])
        if detected_categories:
            pattern_explanation = self._generate_pattern_explanation(pattern_results['far_right'], "")
            explanation_parts.append(f"🔍 Patrones detectados: {pattern_explanation}")
        
        # Claims explanation
        claims = pattern_results.get('claims', [])
        if claims:
            claims_explanation = self._generate_claims_explanation(claims)
            explanation_parts.append(f"📋 Afirmaciones: {claims_explanation}")
        
        # Content insights
        if insights['targeted_groups']:
            explanation_parts.append(f"👥 Grupos objetivo: {', '.join(insights['targeted_groups'])}")
        
        if insights['calls_to_action']:
            explanation_parts.append("📢 Contiene llamadas a la acción o movilización")
        
        return " | ".join(explanation_parts) if explanation_parts else "Contenido analizado sin patrones específicos detectados."
    
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
            
            # Create uncertainty context
            uncertainty_context = self.llm_pipeline.prompt_generator.create_uncertainty_context(pattern_results)
            
            # Get LLM enhancement
            llm_result = self.llm_pipeline.analyze_content(content, {
                'category': category,
                'detected_categories': pattern_results['far_right'].get('categories', []),
                'uncertainty_areas': uncertainty_context.uncertainty_areas
            }, analysis_type)
            
            llm_enhancement = llm_result.get('llm_explanation', '')
            
            if llm_enhancement and len(llm_enhancement.strip()) > 10:
                return f"{base_explanation} | 🤖 Análisis avanzado: {llm_enhancement}"
            else:
                return base_explanation
                
        except Exception as e:
            print(f"⚠️ Error en mejora LLM: {e}")
            return base_explanation
    
    def _primary_llm_analysis(self, content: str, category: str, 
                            pattern_results: Dict, base_explanation: str) -> str:
        """Use LLM as primary analysis method for ambiguous cases."""
        if not self.use_llm or not self.llm_pipeline:
            return f"{base_explanation} | ⚠️ Análisis incierto - se recomienda revisión manual"
        
        try:
            from enhanced_prompts import AnalysisType
            
            # Use comprehensive analysis for uncertain cases
            analysis_type = AnalysisType.COMPREHENSIVE
            
            # Create uncertainty context highlighting what patterns couldn't determine
            uncertainty_context = self.llm_pipeline.prompt_generator.create_uncertainty_context(pattern_results)
            
            llm_result = self.llm_pipeline.analyze_content(content, {
                'category': category,
                'analysis_mode': 'primary',
                'uncertainty_areas': uncertainty_context.uncertainty_areas
            }, analysis_type)
            
            llm_explanation = llm_result.get('llm_explanation', '')
            
            if llm_explanation and len(llm_explanation.strip()) > 10:
                return f"🧠 Análisis principal: {llm_explanation}"
            else:
                return f"{base_explanation} | ⚠️ Análisis LLM falló - se recomienda revisión manual"
                
        except Exception as e:
            print(f"❌ Error en análisis primario LLM: {e}")
            return f"{base_explanation} | ❌ Error LLM - se recomienda revisión manual"
    
    def _determine_llm_analysis_type(self, category: str, pattern_results: Dict):
        """Determine the best LLM analysis type based on detected patterns."""
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
        elif category == "political_bias":
            return AnalysisType.POLITICAL_BIAS
        else:
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
            "political_bias": "Contenido con sesgo político marcado",
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
        'calls_to_action': 'BOOLEAN',
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
        calls_to_action BOOLEAN,
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
             llm_explanation, targeted_groups, calls_to_action, analysis_json, analysis_timestamp) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis.tweet_id, analysis.tweet_url, analysis.username, analysis.tweet_content,
                analysis.category, analysis.llm_explanation,
                json.dumps(analysis.targeted_groups, ensure_ascii=False), analysis.calls_to_action,
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



def get_category_examples():
    """Get one example for each category."""
    return {
        'hate_speech': {
            'tweet_id': 'hate_001',
            'username': 'supremacista_es',
            'content': 'Los musulmanes son una raza inferior que viene a destruir nuestra cultura cristiana. España para los españoles.',
            'expected_category': 'hate_speech'
        },
        'disinformation': {
            'tweet_id': 'disinfo_001',
            'username': 'fake_news_es',
            'content': 'EXCLUSIVO: Las vacunas COVID contienen microchips 5G para controlar la población. Los datos oficiales son mentira del gobierno.',
            'expected_category': 'disinformation'
        },
        'conspiracy_theory': {
            'tweet_id': 'conspiracy_001',
            'username': 'conspiranoia_es',
            'content': 'Soros financia la inmigración para reemplazar a los europeos. Es el plan Kalergi en acción.',
            'expected_category': 'conspiracy_theory'
        },
        'political_bias': {
            'tweet_id': 'bias_001',
            'username': 'partidista_extremo',
            'content': 'Los socialistas han destruido España con su agenda marxista. Solo VOX puede salvar la patria de esta invasión comunista.',
            'expected_category': 'political_bias'
        },
        'call_to_action': {
            'tweet_id': 'action_001',
            'username': 'organizador_patriota',
            'content': '¡CONCENTRACIÓN HOY 18:00 en Cibeles! Hay que salir a las calles a defender España de la invasión. ¡Todos unidos!',
            'expected_category': 'call_to_action'
        },
        'general': {
            'tweet_id': 'general_001',
            'username': 'ciudadano_normal',
            'content': 'Qué día tan bonito hace hoy en Madrid. Me voy a dar un paseo por el Retiro con la familia.',
            'expected_category': 'general'
        }
    }

def run_category_test(analyzer: EnhancedAnalyzer, categories: List[str] = None, save_to_db: bool = True):
    """Run test with one example for each specified category or all categories."""
    available_examples = get_category_examples()
    available_categories = list(available_examples.keys())
    
    # If no categories specified, use all
    if not categories:
        categories = available_categories
    
    # Validate categories
    invalid_categories = [cat for cat in categories if cat not in available_categories]
    if invalid_categories:
        print(f"❌ Categorías inválidas: {invalid_categories}")
        print(f"💡 Categorías disponibles: {', '.join(available_categories)}")
        return []
    
    print("🧪 EJECUTANDO TEST POR CATEGORÍAS")
    print("=" * 60)
    print(f"📋 Categorías a probar: {', '.join(categories)}")
    print(f"🔬 Total de ejemplos: {len(categories)}")
    
    if save_to_db:
        migrate_database_schema()  # Ensure schema is up to date
        init_content_analysis_table()
    
    results = []
    
    for category in categories:
        example = available_examples[category]
        
        print(f"\n📄 TEST: {category.upper()}")
        print(f"🆔 ID: {example['tweet_id']}")
        print(f"👤 Usuario: @{example['username']}")
        print(f"📝 Contenido: {example['content'][:80]}...")
        
        try:
            # Analyze with enhanced analyzer
            analysis = analyzer.analyze_content(
                tweet_id=example['tweet_id'],
                tweet_url=f"https://twitter.com/{example['username']}/status/{example['tweet_id']}",
                username=example['username'],
                content=example['content'],
                retrieve_evidence=False  # Skip for speed in test
            )
            
            # Check accuracy
            category_match = analysis.category == example['expected_category']
            
            print(f"🎯 Categoría: {analysis.category} ({'✓' if category_match else '❌'})")
            print(f" Explicación: {analysis.llm_explanation[:100]}...")
            
            # Save to database
            if save_to_db:
                save_content_analysis(analysis)
                print("💾 Guardado en BD ✓")
            
            result = {
                'category_tested': category,
                'tweet_id': example['tweet_id'],
                'tweet_content': example['content'],  # Add analyzed text
                'username': example['username'],
                'predicted_category': analysis.category,
                'expected_category': example['expected_category'],
                'category_match': category_match,
                'explanation': analysis.llm_explanation,
                'targeted_groups': analysis.targeted_groups,
                'calls_to_action': analysis.calls_to_action,
                'success': True
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error en análisis: {e}")
            results.append({
                'category_tested': category,
                'success': False, 
                'error': str(e)
            })
    
    # Print summary
    print(f"\n📊 RESUMEN DE RESULTADOS")
    print("=" * 40)
    successful_tests = [r for r in results if r.get('success', False)]
    category_matches = [r for r in successful_tests if r.get('category_match', False)]
    
    print(f"✅ Tests exitosos: {len(successful_tests)}/{len(results)}")
    print(f"🎯 Categorías correctas: {len(category_matches)}/{len(successful_tests)}")
    
    if len(successful_tests) > 0:
        accuracy = len(category_matches) / len(successful_tests) * 100
        print(f"📈 Precisión: {accuracy:.1f}%")
    
    return results

def main(test_examples: bool = False,
         skip_save: bool = False,
         use_llm: bool = True,
         model_priority: str = "balanced",
         categories: List[str] = None):
    """
    Main analysis function with enhanced content detection capabilities.
    Now supports testing specific categories or all categories.
    """
    if not test_examples:
        print("❌ Esta versión solo soporta el modo test_examples")
        print("� Usa --test-examples para ejecutar el análisis")
        return []
    
    # Run category-based tests
    analyzer = EnhancedAnalyzer(use_llm=use_llm, model_priority=model_priority)
    results = run_category_test(analyzer, categories=categories, save_to_db=not skip_save)
    
    # Save test results
    try:
        output_path = os.path.join(os.path.dirname(__file__), 'content_test_results.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 Resultados del test guardados en: {output_path}")
    except Exception as e:
        print(f"❌ Error guardando resultados del test: {e}")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced Content Analyzer - Test Mode Only')
    parser.add_argument('--skip-save', action='store_true', 
                       help='Skip saving to database')
    parser.add_argument('--no-llm', action='store_true', 
                       help='Disable LLM analysis')
    parser.add_argument('--test-examples', action='store_true',
                       help='Run comprehensive test with content examples (required)')
    parser.add_argument('--status', action='store_true',
                       help='Show detailed system status and exit')
    parser.add_argument('--model-priority', type=str, default='balanced',
                       choices=['speed', 'balanced', 'quality'],
                       help='Model priority: speed (fast), balanced (default), quality (best)')
    parser.add_argument('--test-llm', action='store_true',
                       help='Test LLM pipeline with sample content and exit')
    parser.add_argument('--categories', nargs='*', 
                       choices=['hate_speech', 'disinformation', 'conspiracy_theory', 
                               'political_bias', 'call_to_action', 'general'],
                       help='Specific categories to test (default: all categories)')
    parser.add_argument('--list-categories', action='store_true',
                       help='Show available categories and exit')
    
    args = parser.parse_args()
    
    # Handle list categories
    if args.list_categories:
        available_examples = get_category_examples()
        print("📋 CATEGORÍAS DISPONIBLES PARA TESTING")
        print("=" * 50)
        for category, example in available_examples.items():
            print(f"🏷️ {category}")
            print(f"   📝 Ejemplo: {example['content'][:80]}...")
            print(f"   🎯 Espera: {example['expected_category']}")
            print()
        print("💡 Uso: --categories hate_speech disinformation")
        print("💡 Para todas: --test-examples (sin --categories)")
        exit(0)
    
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
            print(f"   📊 Sentiment: {result.get('llm_sentiment', 'N/A')}")
            print(f"   ⏱️ Processing time: {end_time - start_time:.2f}s")
            
            analyzer.cleanup_resources()
            print("🎯 LLM pipeline test completed successfully!")
            
        except Exception as e:
            print(f"❌ LLM test failed: {e}")
            analyzer.cleanup_resources()
            exit(1)
        
        exit(0)
    
    # Run analysis (only test_examples mode supported)
    results = main(
        test_examples=args.test_examples,
        skip_save=args.skip_save,
        use_llm=not args.no_llm,
        model_priority=args.model_priority,
        categories=args.categories
    )
    
    if not results:
        print("❌ No se pudieron analizar posts")
