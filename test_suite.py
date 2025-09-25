#!/usr/bin/env python3
"""
            'hate_speech_1': {
                'content': 'Los moros nos están invadiendo 🤬 Fuera de España!! No queremos más mezquitas aquí',
                'expected_category': Categories.HATE_SPEECH,
                'description': 'Direct hate speech with racial slurs'
            },
            'hate_speech_2': {
                'content': 'Ya estoy harto de estos menas que no hacen nada más que robar y traficar 😡 Devolvédlos a su país YA!',
                'expected_category': Categories.HATE_SPEECH,
                'description': 'Hate speech targeting minors - tests consolidated xenophobia patterns'
            },
            
            'disinformation_1': {
                'content': '🚨 URGENTE!! Las vacunas COVID tienen grafeno para controlarnos con 5G!! Compartid antes de que lo censuren!!',
                'expected_category': Categories.DISINFORMATION,
                'description': 'Health disinformation with conspiracy elements'
            }, Suite for Enhanced Analyzer
Optimized version with essential test cases only - much faster execution.
"""

import argparse
import json
import time
from typing import Dict, Any
from enhanced_analyzer import EnhancedAnalyzer
from categories import Categories

class TestSuite:
    """Test suite focusing on core consolidated categories."""
    
    def __init__(self, save_to_db: bool = False):
        print("🚀 Initializing Enhanced Analyzer...")
        self.analyzer = EnhancedAnalyzer(model_priority="fast")
        self.save_to_db = save_to_db
    
    def get_essential_pattern_tests(self) -> Dict[str, Dict]:
        """Essential pattern tests - 2 per consolidated category for maximum coverage with minimal time."""
        return {
            # Core consolidated categories
            'hate_speech_1': {
                'content': 'Los moros nos están invadiendo 🤬 Fuera de España!! No queremos más mezquitas aquí',
                'expected_category': Categories.HATE_SPEECH,
                'description': 'Direct hate speech with racial slurs'
            },
            'hate_speech_2': {
                'content': 'Ya estoy harto de estos menas que no hacen nada más que robar y traficar 😡 Devolvédlos a su país YA!',
                'expected_category': Categories.HATE_SPEECH,
                'description': 'Hate speech targeting minors - tests consolidated xenophobia patterns'
            },
            
            'disinformation_1': {
                'content': '🚨 URGENTE!! Las vacunas COVID tienen grafeno para controlarnos con 5G!! Compartid antes de que lo censuren!!',
                'expected_category': Categories.DISINFORMATION,
                'description': 'COVID vaccine misinformation - tests consolidated health_disinformation'
            },
            'disinformation_2': {
                'content': 'Los medios OCULTAN la verdad: 9 de cada 10 casos de COVID son INVENTADOS para meter miedo!!',
                'expected_category': [Categories.DISINFORMATION, Categories.CONSPIRACY_THEORY],
                'description': 'False statistics - multi-category test'
            },
            
            'conspiracy_theory_1': {
                'content': 'Soros financia la inmigración masiva para reemplazar a los europeos blancos. Es el plan Kalergi en marcha!! 🚨🚨',
                'expected_category': Categories.CONSPIRACY_THEORY,
                'description': 'Great replacement conspiracy theory'
            },
            'conspiracy_theory_2': {
                'content': 'Bill Gates, Soros y el Foro de Davos se reunen en secreto para decidir nuestro futuro 👁️ Despertad ovejas!!',
                'expected_category': Categories.CONSPIRACY_THEORY,
                'description': 'Elite coordination conspiracy theory'
            },
            
            'far_right_bias_1': {
                'content': 'Los ROJOS han convertido España en Venezuela 🇻🇪 Solo Vox puede salvarnos de estos COMUNISTAS!',
                'expected_category': Categories.FAR_RIGHT_BIAS,
                'description': 'Extreme partisan rhetoric against socialists'
            },
            'far_right_bias_2': {
                'content': 'La agenda WOKE está destruyendo nuestros valores cristianos!! 😡 Nos quieren convertir en maricas y feminazis',
                'expected_category': [Categories.FAR_RIGHT_BIAS, Categories.HATE_SPEECH],
                'description': 'Anti-woke bias with slurs - multi-category test'
            },
            
            'call_to_action_1': {
                'content': '🚨 CONVOCATORIA URGENTE!! Todos a Cibeles HOY 18:00h!! A defender España de la INVASIÓN!! RT MASIVO!!',
                'expected_category': Categories.CALL_TO_ACTION,
                'description': 'Direct mobilization call with location and time'
            },
            'call_to_action_2': {
                'content': 'BASTA YA!! La gente tiene que unirse y salir a las calles antes de que sea demasiado tarde!! #Resistencia',
                'expected_category': Categories.CALL_TO_ACTION,
                'description': 'Urgent street mobilization call'
            },
            
            'general_1': {
                'content': 'Hoy hace un día muy bonito en Madrid. Perfecto para pasear por el Retiro con la familia ☀️',
                'expected_category': Categories.GENERAL,
                'description': 'Normal everyday content that should not match any patterns'
            }
        }
    
    def get_essential_llm_tests(self) -> Dict[str, Dict]:
        """Essential LLM tests - subtle cases that require LLM analysis."""
        return {
            'hate_speech_llm_1': {
                'content': 'Ya sabéis cómo son esa gente... no cambian nunca. Siempre igual, da igual donde vayan',
                'expected_category': Categories.HATE_SPEECH,
                'description': 'Subtle hate speech with implicit derogatory generalizations'
            },
            
            'disinformation_llm_1': {
                'content': 'Mi primo que trabaja en el hospital me dice que todos los casos COVID son gripe común reclasificada',
                'expected_category': Categories.DISINFORMATION,
                'description': 'Subtle medical misinformation with fake authority'
            },
            
            'conspiracy_theory_llm_1': {
                'content': 'Es casualidad que todos los líderes mundiales hablen igual sobre el cambio climático? Alguien les está diciendo qué decir...',
                'expected_category': Categories.CONSPIRACY_THEORY,
                'description': 'Subtle conspiracy theory about global coordination'
            },
            
            'general_llm_1': {
                'content': 'Me parece que el gobierno debería invertir más en educación pública y menos en otras cosas',
                'expected_category': Categories.GENERAL,
                'description': 'Normal political opinion without extremist elements'
            }
        }
    
    def run_pattern_tests(self, quick_mode: bool = False) -> Dict[str, Any]:
        """Run essential pattern tests."""
        test_cases = self.get_essential_pattern_tests()
        
        if quick_mode:
            # In quick mode, take only 2 tests total for fastest execution
            test_cases = {k: v for i, (k, v) in enumerate(test_cases.items()) if i < 2}  # 2 tests total
        
        print("🔍 TESTING PATTERN-BASED DETECTION")
        print("=" * 60)
        print(f"⚡ Running {len(test_cases)} pattern tests... ({'quick mode' if quick_mode else 'full mode'})")
        
        passed = 0
        failed = 0
        results = []
        
        for test_id, test_case in test_cases.items():
            print(f"⚡ {test_id}... ", end="", flush=True)
            
            try:
                analysis = self.analyzer.analyze_content(
                    tweet_id=test_id,
                    tweet_url=f"https://example.com/{test_id}",
                    username="test_user",
                    content=test_case['content']
                )
                
                expected = test_case['expected_category']
                actual = analysis.category
                
                # Handle multi-category expectations
                if isinstance(expected, list):
                    is_success = actual in expected
                    expected_str = f"[{', '.join(expected)}]"
                else:
                    is_success = actual == expected
                    expected_str = expected
                
                if is_success:
                    print("✅")
                    print(f"   ✅ PASSED: {test_id}")
                    print(f"   📝 Content: {test_case['content'][:80]}...")
                    print(f"   🎯 Category: {actual}")
                    print(f"   💡 Explanation: {analysis.llm_explanation[:100]}...")
                    passed += 1
                else:
                    print("❌")
                    print(f"   ❌ FAILED: {test_id}")
                    print(f"   📝 Content: {test_case['content'][:80]}...")
                    print(f"   🎯 Expected: {expected_str}, Got: {actual}")
                    print(f"   💡 Explanation: {analysis.llm_explanation[:100]}...")
                    failed += 1
                
                results.append({
                    'test_id': test_id,
                    'expected': expected,
                    'actual': actual,
                    'success': is_success,
                    'description': test_case['description'],
                    'content': test_case['content'],
                    'llm_explanation': analysis.llm_explanation,
                    'analysis_method': analysis.analysis_method
                })
                print()
                
            except Exception as e:
                print("💥")
                print(f"   💥 ERROR: {test_id} - {str(e)}")
                failed += 1
                results.append({
                    'test_id': test_id,
                    'error': str(e),
                    'success': False
                })
        
        print(f"📊 Pattern Tests: {passed}/{passed+failed} passed ({passed/(passed+failed)*100:.1f}%)")
        return {'passed': passed, 'failed': failed, 'results': results}
    
    def run_llm_tests(self, quick_mode: bool = False) -> Dict[str, Any]:
        """Run essential LLM tests."""
        test_cases = self.get_essential_llm_tests()
        
        if quick_mode:
            test_cases = {k: v for i, (k, v) in enumerate(test_cases.items()) if i < 2}  # 2 tests in quick mode
        
        print("🧠 TESTING LLM-BASED CLASSIFICATION")
        print("=" * 60)
        print(f"⚡ Running {len(test_cases)} LLM tests... ({'quick mode' if quick_mode else 'full mode'})")
        
        passed = 0
        failed = 0
        results = []
        
        for test_id, test_case in test_cases.items():
            print(f"⚡ {test_id}... ", end="", flush=True)
            
            try:
                analysis = self.analyzer.analyze_content(
                    tweet_id=test_id,
                    tweet_url=f"https://example.com/{test_id}",
                    username="test_user_llm",
                    content=test_case['content']
                )
                
                expected = test_case['expected_category']
                actual = analysis.category
                is_success = actual == expected
                
                if is_success:
                    print("✅")
                    print(f"   ✅ PASSED: {test_id}")
                    passed += 1
                else:
                    print("❌")
                    print(f"   ❌ FAILED: {test_id} - Expected: {expected}, Got: {actual}")
                    failed += 1
                
                results.append({
                    'test_id': test_id,
                    'expected': expected,
                    'actual': actual,
                    'success': is_success,
                    'description': test_case['description'],
                    'content': test_case['content'],
                    'llm_explanation': analysis.llm_explanation,
                    'analysis_method': analysis.analysis_method
                })
                
            except Exception as e:
                print("💥")
                print(f"   💥 ERROR: {test_id} - {str(e)}")
                failed += 1
                results.append({
                    'test_id': test_id,
                    'error': str(e),
                    'success': False,
                    'description': test_case['description'],
                    'content': test_case['content']
                })
        
        print(f"📊 LLM Tests: {passed}/{passed+failed} passed ({passed/(passed+failed)*100:.1f}%)")
        return {'passed': passed, 'failed': failed, 'results': results}
    
    def run_comprehensive_suite(self, quick_mode: bool = False) -> Dict[str, Any]:
        """Run the complete test suite."""
        print(f"🚀  TEST SUITE")
        if quick_mode:
            print("⚡ Quick mode: Running 2 tests only...")
        print("=" * 70)
        
        start_time = time.time()
        
        # Run pattern tests
        pattern_results = self.run_pattern_tests(quick_mode=quick_mode)
        
        # Run LLM tests  
        llm_results = self.run_llm_tests(quick_mode=quick_mode)
        
        # Calculate totals
        total_passed = pattern_results['passed'] + llm_results['passed']
        total_failed = pattern_results['failed'] + llm_results['failed']
        total_tests = total_passed + total_failed
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        elapsed_time = time.time() - start_time
        
        # Print summary
        print("📊 TEST SUMMARY")
        print("=" * 50)
        print(f"✅ Pattern Tests: {pattern_results['passed']}/{pattern_results['passed'] + pattern_results['failed']} passed")
        print(f"🧠 LLM Tests: {llm_results['passed']}/{llm_results['passed'] + llm_results['failed']} passed")
        print(f"📈 Overall Success Rate: {success_rate:.1f}% ({total_passed}/{total_tests})")
        print(f"⏱️  Total Execution Time: {elapsed_time:.2f} seconds")
        
        if total_failed > 0:
            print(f"⚠️  {total_failed} tests failed - check individual results above")
        else:
            print("🎉 All tests passed! Multi-category system working perfectly!")
        
        return {
            'pattern_results': pattern_results,
            'llm_results': llm_results,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'success_rate': success_rate,
            'execution_time': elapsed_time
        }

def main():
    parser = argparse.ArgumentParser(description='Test Suite for Enhanced Analyzer')
    parser.add_argument('--quick', action='store_true', help='Run quick mode (2 tests total for speed)')
    parser.add_argument('--patterns-only', action='store_true', help='Run only pattern tests')
    parser.add_argument('--llm-only', action='store_true', help='Run only LLM tests')
    parser.add_argument('--save-results', action='store_true', help='Save results to JSON')
    
    args = parser.parse_args()
    
    test_suite = TestSuite()
    
    if args.patterns_only:
        results = test_suite.run_pattern_tests(quick_mode=args.quick)
    elif args.llm_only:
        results = test_suite.run_llm_tests(quick_mode=args.quick)
    else:
        results = test_suite.run_comprehensive_suite(quick_mode=args.quick)
    
    if args.save_results:
        with open('test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to test_results.json")

if __name__ == "__main__":
    main()