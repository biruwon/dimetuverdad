#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Analyzer
Extracted from enhanced_analyzer.py and merged with custom test cases
"""

import json
import os
import argparse
from typing import List, Dict, Any
from enhanced_analyzer import EnhancedAnalyzer

class ComprehensiveTestSuite:
    """Complete test suite combining pattern and LLM test cases."""
    
    def __init__(self, save_to_db: bool = False):
        self.analyzer = EnhancedAnalyzer()
        self.save_to_db = save_to_db
        
    def get_pattern_based_tests(self) -> Dict[str, Dict]:
        """Get test cases that should be caught by pattern detection."""
        return {
            'hate_speech_pattern_1': {
                'tweet_id': 'hate_001',
                'username': 'supremacista_es',
                'content': 'Los musulmanes son una raza inferior que viene a destruir nuestra cultura cristiana. España para los españoles.',
                'expected_category': 'hate_speech',
                'analysis_method': 'pattern',
                'description': 'Direct hate speech with racial slurs'
            },
            'hate_speech_pattern_2': {
                'tweet_id': 'hate_002',
                'username': 'test_user',
                'content': 'Los inmigrantes ilegales son fundamentalmente incompatibles con nuestra sociedad y representan una amenaza directa a nuestros valores.',
                'expected_category': 'hate_speech',
                'analysis_method': 'pattern',
                'description': 'Incompatibility language targeting immigrants'
            },
            'hate_speech_pattern_3': {
                'tweet_id': 'hate_003',
                'username': 'test_user',
                'content': 'Estas personas no merecen los mismos derechos que los ciudadanos reales porque no comparten nuestros principios fundamentales.',
                'expected_category': 'hate_speech',
                'analysis_method': 'pattern',
                'description': 'Dehumanization and rights denial'
            },
            
            'disinformation_pattern_1': {
                'tweet_id': 'disinfo_001',
                'username': 'fake_news_es',
                'content': 'EXCLUSIVO: Las vacunas COVID contienen microchips 5G para controlar la población. Los datos oficiales son mentira del gobierno.',
                'expected_category': 'disinformation',
                'analysis_method': 'pattern',
                'description': 'COVID vaccine misinformation'
            },
            'disinformation_pattern_2': {
                'tweet_id': 'disinfo_002',
                'username': 'test_user',
                'content': 'Estudios recientes demuestran que el 85% de los casos reportados son completamente fabricados por los medios de comunicación.',
                'expected_category': 'disinformation',
                'analysis_method': 'pattern',
                'description': 'False statistics without sources'
            },
            'disinformation_pattern_3': {
                'tweet_id': 'disinfo_003',
                'username': 'test_user',
                'content': 'Los datos oficiales ocultan la verdadera magnitud del problema, que según investigaciones privadas es tres veces mayor.',
                'expected_category': 'disinformation',
                'analysis_method': 'pattern',
                'description': 'Unsupported factual claims'
            },
            
            'conspiracy_theory_pattern_1': {
                'tweet_id': 'conspiracy_001',
                'username': 'conspiranoia_es',
                'content': 'Soros financia la inmigración para reemplazar a los europeos. Es el plan Kalergi en acción.',
                'expected_category': 'conspiracy_theory',
                'analysis_method': 'pattern',
                'description': 'Great Replacement theory'
            },
            'conspiracy_theory_pattern_2': {
                'tweet_id': 'conspiracy_002',
                'username': 'test_user',
                'content': 'Las grandes corporaciones coordinan secretamente sus políticas para mantener el control sobre la opinión pública.',
                'expected_category': 'conspiracy_theory',
                'analysis_method': 'pattern',
                'description': 'Corporate coordination theory'
            },
            
            'far_right_bias_pattern_1': {
                'tweet_id': 'bias_001',
                'username': 'partidista_extremo',
                'content': 'Los socialistas han destruido España con su agenda marxista. Solo VOX puede salvar la patria de esta invasión comunista.',
                'expected_category': 'far_right_bias',
                'analysis_method': 'pattern',
                'description': 'Partisan extremist rhetoric'
            },
            'far_right_bias_pattern_2': {
                'tweet_id': 'bias_002',
                'username': 'test_user',
                'content': 'La nueva agenda cultural está sistemáticamente transformando los pilares tradicionales de nuestra identidad nacional.',
                'expected_category': 'far_right_bias',
                'analysis_method': 'pattern',
                'description': 'Cultural transformation threat narrative'
            },
            'far_right_bias_pattern_3': {
                'tweet_id': 'bias_003',
                'username': 'test_user',
                'content': 'La inmigración masiva está destruyendo sistemáticamente el tejido social y cultural que nuestros antepasados construyeron durante siglos.',
                'expected_category': ['far_right_bias', 'hate_speech'],  # Both reasonable
                'analysis_method': 'pattern',
                'description': 'Immigration cultural threat narrative'
            },
            
            'call_to_action_pattern_1': {
                'tweet_id': 'action_001',
                'username': 'organizador_patriota',
                'content': '¡CONCENTRACIÓN HOY 18:00 en Cibeles! Hay que salir a las calles a defender España de la invasión. ¡Todos unidos!',
                'expected_category': 'call_to_action',
                'analysis_method': 'pattern',
                'description': 'Direct mobilization call'
            },
            'call_to_action_pattern_2': {
                'tweet_id': 'action_002',
                'username': 'test_user',
                'content': 'Los ciudadanos conscientes de esta realidad deberían organizarse para defender los valores de nuestra comunidad.',
                'expected_category': 'call_to_action',
                'analysis_method': 'pattern',
                'description': 'Community defense call'
            },
            'call_to_action_pattern_3': {
                'tweet_id': 'action_003',
                'username': 'test_user',
                'content': 'Es fundamental que los padres de familia se movilicen para proteger la educación de sus hijos ante estas amenazas.',
                'expected_category': 'call_to_action',
                'analysis_method': 'pattern',
                'description': 'Parental mobilization call'
            },
            'call_to_action_pattern_4': {
                'tweet_id': 'action_004',
                'username': 'test_user',
                'content': 'Es hora de que la gente se una y tome medidas concretas para frenar esta situación antes de que sea demasiado tarde.',
                'expected_category': 'call_to_action',
                'analysis_method': 'pattern',
                'description': 'Urgent mobilization call'
            },
            'call_to_action_pattern_5': {
                'tweet_id': 'action_005',
                'username': 'test_user',
                'content': 'Los verdaderos patriotas deben actuar ahora para recuperar el control de nuestras instituciones.',
                'expected_category': 'call_to_action',
                'analysis_method': 'pattern',
                'description': 'Patriotic action call'
            }
        }
    
    def get_llm_fallback_tests(self) -> Dict[str, Dict]:
        """Get test cases requiring LLM analysis (subtle content)."""
        return {
            'hate_speech_llm_1': {
                'tweet_id': 'hate_llm_001',
                'username': 'test_user_llm',
                'content': 'Algunos grupos culturales tienen características que los hacen incompatibles con los valores occidentales modernos.',
                'expected_category': 'hate_speech',
                'analysis_method': 'llm',
                'description': 'Subtle hate speech using neutral language'
            },
            
            'disinformation_llm_1': {
                'tweet_id': 'disinfo_llm_001',
                'username': 'test_user_llm',
                'content': 'He leído varios estudios que sugieren correlaciones preocupantes entre ciertos tratamientos médicos y efectos adversos no reportados.',
                'expected_category': 'disinformation',
                'analysis_method': 'llm',
                'description': 'Implicit medical disinformation without explicit claims'
            },
            
            'conspiracy_theory_llm_1': {
                'tweet_id': 'conspiracy_llm_001',
                'username': 'test_user_llm',
                'content': 'Es interesante cómo ciertos eventos mundiales parecen beneficiar siempre a los mismos actores económicos internacionales.',
                'expected_category': ['conspiracy_theory', 'disinformation'],  # Both reasonable
                'analysis_method': 'llm',
                'description': 'Coded conspiracy language without explicit patterns'
            },
            'conspiracy_theory_llm_2': {
                'tweet_id': 'conspiracy_llm_002',
                'username': 'test_user_llm',
                'content': 'Resulta interesante observar cómo determinadas empresas farmacéuticas obtienen contratos millonarios cada vez que surgen nuevas crisis sanitarias globales.',
                'expected_category': ['conspiracy_theory', 'disinformation'],  # Both reasonable
                'analysis_method': 'llm',
                'description': 'Pharmaceutical conspiracy suggestion'
            },
            'conspiracy_theory_llm_3': {
                'tweet_id': 'conspiracy_llm_003',
                'username': 'test_user_llm',
                'content': 'Diversos informes independientes sugieren que ciertos sectores tecnológicos manipulan la información que recibimos diariamente.',
                'expected_category': ['conspiracy_theory', 'disinformation'],  # Both reasonable
                'analysis_method': 'llm',
                'description': 'Information manipulation theory'
            },
            
            'far_right_bias_llm_1': {
                'tweet_id': 'bias_llm_001',
                'username': 'test_user_llm',
                'content': 'La agenda progresista está transformando fundamentalmente las instituciones tradicionales de nuestra sociedad de maneras que muchos ciudadanos encuentran problemáticas.',
                'expected_category': 'far_right_bias',
                'analysis_method': 'llm',
                'description': 'Veiled far-right bias without explicit partisan language'
            },
            'far_right_bias_llm_2': {
                'tweet_id': 'bias_llm_002',
                'username': 'test_user_llm',
                'content': 'El sistema educativo actual impone una perspectiva ideológica específica que marginaliza las creencias tradicionales.',
                'expected_category': ['far_right_bias', 'disinformation'],  # Both reasonable
                'analysis_method': 'llm',
                'description': 'Educational bias concern'
            },
            'far_right_bias_llm_3': {
                'tweet_id': 'bias_llm_003',
                'username': 'test_user_llm',
                'content': 'Los valores occidentales tradicionales están siendo erosionados por una agenda globalista que busca homogeneizar todas las culturas.',
                'expected_category': ['far_right_bias', 'conspiracy_theory'],  # Both reasonable
                'analysis_method': 'llm',
                'description': 'Globalist agenda narrative'
            },
            
            'call_to_action_llm_1': {
                'tweet_id': 'action_llm_001',
                'username': 'test_user_llm',
                'content': 'Es momento de que los ciudadanos responsables tomen medidas para proteger sus comunidades y valores fundamentales.',
                'expected_category': 'call_to_action',
                'analysis_method': 'llm',
                'description': 'Indirect call to action without specific mobilization'
            }
        }
    
    def get_neutral_tests(self) -> Dict[str, Dict]:
        """Get genuinely neutral content tests."""
        return {
            'general_1': {
                'tweet_id': 'general_001',
                'username': 'ciudadano_normal',
                'content': 'Qué día tan bonito hace hoy en Madrid. Me voy a dar un paseo por el Retiro con la familia.',
                'expected_category': 'general',
                'analysis_method': 'pattern',
                'description': 'Weather and family activities'
            },
            'general_2': {
                'tweet_id': 'general_002',
                'username': 'test_user',
                'content': 'El nuevo restaurante en el barrio tiene una comida excelente. Recomiendo especialmente la paella de mariscos.',
                'expected_category': 'general',
                'analysis_method': 'pattern',
                'description': 'Restaurant recommendation'
            },
            'general_3': {
                'tweet_id': 'general_003',
                'username': 'test_user',
                'content': 'Mañana empieza la nueva temporada de mi serie favorita. He estado esperando este momento durante meses.',
                'expected_category': 'general',
                'analysis_method': 'pattern',
                'description': 'Entertainment content'
            }
        }
    
    def run_pattern_tests(self) -> Dict[str, Any]:
        """Run tests that should be caught by pattern detection."""
        print("🔍 TESTING PATTERN-BASED DETECTION")
        print("=" * 60)
        
        test_cases = self.get_pattern_based_tests()
        results = {
            'category': 'pattern_based',
            'total_tests': len(test_cases),
            'passed': 0,
            'failed': 0,
            'test_results': []
        }
        
        for test_name, test_case in test_cases.items():
            print(f"\n📄 TEST: {test_name}")
            print(f"📝 Content: {test_case['content'][:80]}...")
            print(f"🎯 Expected: {test_case['expected_category']}")
            
            try:
                # Analyze content
                analysis = self.analyzer.analyze_content(
                    tweet_id=test_case['tweet_id'],
                    tweet_url=f"https://twitter.com/{test_case['username']}/status/{test_case['tweet_id']}",
                    username=test_case['username'],
                    content=test_case['content'],
                    retrieve_evidence=False  # Skip for speed
                )
                
                actual_category = analysis.category
                expected = test_case['expected_category']
                
                # Handle both single expected value and list of expected values
                if isinstance(expected, list):
                    is_correct = actual_category in expected
                    expected_str = f"one of {expected}"
                else:
                    is_correct = actual_category == expected
                    expected_str = expected
                
                if is_correct:
                    print(f"✅ PASS - Got: {actual_category}")
                    results['passed'] += 1
                    status = 'PASS'
                else:
                    print(f"❌ FAIL - Expected: {expected_str}, Got: {actual_category}")
                    results['failed'] += 1
                    status = 'FAIL'
                
                results['test_results'].append({
                    'test_name': test_name,
                    'expected': expected,
                    'actual': actual_category,
                    'status': status,
                    'method_used': analysis.analysis_method if hasattr(analysis, 'analysis_method') else 'unknown'
                })
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
                results['failed'] += 1
                results['test_results'].append({
                    'test_name': test_name,
                    'expected': test_case['expected_category'],
                    'actual': 'ERROR',
                    'status': 'ERROR',
                    'error': str(e)
                })
        
        print(f"\n📊 Pattern Tests: {results['passed']}/{results['total_tests']} passed ({(results['passed']/results['total_tests'])*100:.1f}%)")
        return results
    
    def run_llm_tests(self) -> Dict[str, Any]:
        """Run tests requiring LLM analysis."""
        print("\n🧠 TESTING LLM-BASED CLASSIFICATION")
        print("=" * 60)
        
        test_cases = self.get_llm_fallback_tests()
        results = {
            'category': 'llm_based',
            'total_tests': len(test_cases),
            'passed': 0,
            'failed': 0,
            'test_results': []
        }
        
        for test_name, test_case in test_cases.items():
            print(f"\n📄 TEST: {test_name}")
            print(f"📝 Content: {test_case['content'][:80]}...")
            print(f"🎯 Expected: {test_case['expected_category']}")
            
            try:
                # Analyze content
                analysis = self.analyzer.analyze_content(
                    tweet_id=test_case['tweet_id'],
                    tweet_url=f"https://twitter.com/{test_case['username']}/status/{test_case['tweet_id']}",
                    username=test_case['username'],
                    content=test_case['content'],
                    retrieve_evidence=False  # Skip for speed
                )
                
                actual_category = analysis.category
                expected = test_case['expected_category']
                
                # Handle both single expected value and list of expected values
                if isinstance(expected, list):
                    is_correct = actual_category in expected
                    expected_str = f"one of {expected}"
                else:
                    is_correct = actual_category == expected
                    expected_str = expected
                
                if is_correct:
                    print(f"✅ PASS - Got: {actual_category}")
                    results['passed'] += 1
                    status = 'PASS'
                else:
                    print(f"❌ FAIL - Expected: {expected_str}, Got: {actual_category}")
                    results['failed'] += 1
                    status = 'FAIL'
                
                results['test_results'].append({
                    'test_name': test_name,
                    'expected': expected,
                    'actual': actual_category,
                    'status': status,
                    'method_used': analysis.analysis_method if hasattr(analysis, 'analysis_method') else 'unknown'
                })
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
                results['failed'] += 1
                results['test_results'].append({
                    'test_name': test_name,
                    'expected': test_case['expected_category'],
                    'actual': 'ERROR',
                    'status': 'ERROR',
                    'error': str(e)
                })
        
        print(f"\n📊 LLM Tests: {results['passed']}/{results['total_tests']} passed ({(results['passed']/results['total_tests'])*100:.1f}%)")
        return results
    
    def run_neutral_tests(self) -> Dict[str, Any]:
        """Run tests for genuinely neutral content."""
        print("\n🌍 TESTING NEUTRAL CONTENT CLASSIFICATION")
        print("=" * 60)
        
        test_cases = self.get_neutral_tests()
        results = {
            'category': 'neutral_content',
            'total_tests': len(test_cases),
            'passed': 0,
            'failed': 0,
            'test_results': []
        }
        
        for test_name, test_case in test_cases.items():
            print(f"\n📄 TEST: {test_name}")
            print(f"📝 Content: {test_case['content'][:80]}...")
            print(f"🎯 Expected: {test_case['expected_category']}")
            
            try:
                # Analyze content
                analysis = self.analyzer.analyze_content(
                    tweet_id=test_case['tweet_id'],
                    tweet_url=f"https://twitter.com/{test_case['username']}/status/{test_case['tweet_id']}",
                    username=test_case['username'],
                    content=test_case['content'],
                    retrieve_evidence=False  # Skip for speed
                )
                
                actual_category = analysis.category
                expected = test_case['expected_category']
                
                # For general content, we should accept 'general' classification
                # But if LLM incorrectly classifies as specific category, it's a failure
                if expected == 'general':
                    is_correct = actual_category == 'general'
                else:
                    is_correct = actual_category == expected
                
                if is_correct:
                    print(f"✅ PASS - Got: {actual_category}")
                    results['passed'] += 1
                    status = 'PASS'
                else:
                    print(f"❌ FAIL - Expected: {expected}, Got: {actual_category}")
                    results['failed'] += 1
                    status = 'FAIL'
                
                results['test_results'].append({
                    'test_name': test_name,
                    'expected': expected,
                    'actual': actual_category,
                    'status': status,
                    'method_used': analysis.analysis_method if hasattr(analysis, 'analysis_method') else 'unknown'
                })
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
                results['failed'] += 1
                results['test_results'].append({
                    'test_name': test_name,
                    'expected': test_case['expected_category'],
                    'actual': 'ERROR',
                    'status': 'ERROR',
                    'error': str(e)
                })
        
        print(f"\n📊 Neutral Tests: {results['passed']}/{results['total_tests']} passed ({(results['passed']/results['total_tests'])*100:.1f}%)")
        return results
    
    def run_comprehensive_suite(self) -> Dict[str, Any]:
        """Run all test categories."""
        print("🧪 COMPREHENSIVE TEST SUITE")
        print("=" * 70)
        print("Testing pattern detection, LLM classification, and neutral content handling")
        print()
        
        # Run all test categories
        pattern_results = self.run_pattern_tests()
        llm_results = self.run_llm_tests()
        neutral_results = self.run_neutral_tests()
        
        # Compile comprehensive results
        total_tests = pattern_results['total_tests'] + llm_results['total_tests'] + neutral_results['total_tests']
        total_passed = pattern_results['passed'] + llm_results['passed'] + neutral_results['passed']
        total_failed = pattern_results['failed'] + llm_results['failed'] + neutral_results['failed']
        
        comprehensive_results = {
            'suite_name': 'comprehensive_analyzer_test',
            'timestamp': str(os.path.getmtime(__file__)),
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'success_rate': (total_passed / total_tests) * 100 if total_tests > 0 else 0,
            'results_by_category': {
                'pattern_based': pattern_results,
                'llm_based': llm_results,
                'neutral_content': neutral_results
            }
        }
        
        # Print comprehensive summary
        print("\n" + "=" * 70)
        print("🎯 COMPREHENSIVE TEST RESULTS")
        print("=" * 70)
        print(f"📊 Pattern-based Tests: {pattern_results['passed']}/{pattern_results['total_tests']} ({(pattern_results['passed']/pattern_results['total_tests'])*100:.1f}%)")
        print(f"🧠 LLM-based Tests: {llm_results['passed']}/{llm_results['total_tests']} ({(llm_results['passed']/llm_results['total_tests'])*100:.1f}%)")
        print(f"🌍 Neutral Content Tests: {neutral_results['passed']}/{neutral_results['total_tests']} ({(neutral_results['passed']/neutral_results['total_tests'])*100:.1f}%)")
        print(f"🎯 OVERALL SUCCESS RATE: {total_passed}/{total_tests} ({comprehensive_results['success_rate']:.1f}%)")
        
        if comprehensive_results['success_rate'] == 100.0:
            print("🎉 PERFECT SCORE! All tests passed!")
        elif comprehensive_results['success_rate'] >= 90.0:
            print("🚀 Excellent performance!")
        elif comprehensive_results['success_rate'] >= 75.0:
            print("✅ Good performance with room for improvement")
        else:
            print("⚠️  Performance needs improvement")
        
        # Save results
        try:
            output_path = os.path.join(os.path.dirname(__file__), 'comprehensive_test_results.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_results, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n💾 Results saved to: {output_path}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
        
        return comprehensive_results

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Comprehensive Test Suite for Enhanced Analyzer')
    parser.add_argument('--save-to-db', action='store_true', 
                       help='Save test results to database')
    parser.add_argument('--patterns-only', action='store_true',
                       help='Run only pattern-based tests')
    parser.add_argument('--llm-only', action='store_true',
                       help='Run only LLM-based tests')
    parser.add_argument('--neutral-only', action='store_true',
                       help='Run only neutral content tests')
    
    args = parser.parse_args()
    
    # Initialize test suite
    test_suite = ComprehensiveTestSuite(save_to_db=args.save_to_db)
    
    # Run specified tests
    if args.patterns_only:
        test_suite.run_pattern_tests()
    elif args.llm_only:
        test_suite.run_llm_tests()
    elif args.neutral_only:
        test_suite.run_neutral_tests()
    else:
        # Run comprehensive suite
        test_suite.run_comprehensive_suite()

if __name__ == "__main__":
    main()
