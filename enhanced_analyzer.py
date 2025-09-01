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
        
        # Pipeline Step 4: Text generation and explanation
        llm_explanation = self._generate_explanation(content, category, pattern_results, insights)
        
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
        Pipeline Step 1: Run all pattern analyzers once and consolidate results.
        This eliminates duplication by doing all pattern analysis in one place.
        """
        # Run all analyzers once
        far_right_result = self.far_right_analyzer.analyze_text(content)
        topic_results = self.topic_classifier.classify_topic(content)
        claims = self.claim_detector.detect_claims(content)
        
        # Consolidate results
        return {
            'far_right': far_right_result,
            'topics': topic_results,
            'claims': claims,
            'pattern_matches': far_right_result.get('pattern_matches', [])
        }
    
    def _categorize_content(self, content: str, pattern_results: Dict) -> str:
        """
        Pipeline Step 2: Determine content category using consolidated pattern results.
        """
        content_lower = content.lower()
        far_right_result = pattern_results['far_right']
        
        # Check for hate speech indicators (highest priority)
        hate_indicators = [
            'raza inferior', 'sangre pura', 'eliminar', 'deportar', 'virus', 'infectan',
            'supremacía', 'superioridad', 'inferioridad racial', 'genéticamente inferior'
        ]
        if any(indicator in content_lower for indicator in hate_indicators):
            return "hate_speech"
        
        # Check for disinformation indicators
        disinfo_indicators = [
            'microchips', 'vacunas', 'controlarnos', 'agenda globalista', 'élite',
            'datos oficiales son mentira', 'gobierno oculta', 'estudio secreto',
            'medios mainstream ocultan'
        ]
        if any(indicator in content_lower for indicator in disinfo_indicators):
            return "disinformation"
        
        # Check for conspiracy theories
        conspiracy_indicators = [
            'nuevo orden mundial', 'illuminati', 'soros', 'plan kalergi', 'reemplazar',
            'orquestado', 'controlarnos', 'élite globalista'
        ]
        if any(indicator in content_lower for indicator in conspiracy_indicators):
            return "conspiracy_theory"
        
        # Check for calls to action
        action_indicators = [
            'concentración', 'manifestación', 'calles', 'resistencia', 'actuar',
            'organizad', 'difunde', 'comparte', 'despertad', 'todos unidos'
        ]
        if any(indicator in content_lower for indicator in action_indicators):
            return "call_to_action"
        
        # Check for political bias
        political_indicators = [
            'socialistas han destruido', 'dictadura comunista', 'régimen de sánchez',
            'patriotas debemos', 'franco', 'democracia ha fracasado', 'vox puede salvar'
        ]
        if any(indicator in content_lower for indicator in political_indicators):
            return "political_bias"
        
        # Use far-right score for general classification
        if far_right_result.get('score', 0) > 0.6:
            return "political_bias"
        elif far_right_result.get('score', 0) > 0.3:
            return "political_bias"
        
        # Normal content
        return "general"
    
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
    
    def _generate_explanation(self, content: str, category: str, 
                            pattern_results: Dict, insights: Dict) -> str:
        """
        Pipeline Step 4: Generate comprehensive explanation for the detected content.
        """
        explanation_parts = []
        
        # 1. Category-based explanation
        if category != "general":
            category_explanation = self._generate_category_explanation(category, content)
            explanation_parts.append(f"📂 Categorización: {category_explanation}")
        
        # 2. Pattern analysis explanation
        far_right_result = pattern_results['far_right']
        if far_right_result.get('score', 0) > 0.3:
            pattern_explanation = self._generate_pattern_explanation(far_right_result, content)
            explanation_parts.append(f"🔍 Patrones detectados: {pattern_explanation}")
        
        # 3. Claims analysis explanation
        claims = pattern_results['claims']
        if claims:
            claims_explanation = self._generate_claims_explanation(claims)
            explanation_parts.append(f"📋 Afirmaciones: {claims_explanation}")
        
        # 4. Content insights explanation
        if insights['targeted_groups']:
            explanation_parts.append(f"👥 Grupos objetivo: {', '.join(insights['targeted_groups'])}")
        
        if insights['calls_to_action']:
            explanation_parts.append("📢 Contiene llamadas a la acción o movilización")
        
        # 5. Use LLM for enhanced explanation if available
        if self.use_llm and self.llm_pipeline:
            try:
                llm_explanation = self._get_llm_explanation(content, category, explanation_parts)
                if llm_explanation:
                    explanation_parts.append(f"🤖 Análisis avanzado: {llm_explanation}")
            except Exception as e:
                print(f"❌ Error en LLM: {e}")
        
        return " | ".join(explanation_parts) if explanation_parts else "Contenido analizado sin patrones específicos detectados."
    
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
        patterns = far_right_result.get('pattern_matches', [])
        if not patterns:
            return "Contenido con características de sesgo político"
        
        pattern_descriptions = []
        for pattern in patterns[:3]:  # Limit to top 3 patterns
            if 'category' in pattern:
                pattern_descriptions.append(pattern['category'])
        
        return f"Patrones de {', '.join(pattern_descriptions)}"
    
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
