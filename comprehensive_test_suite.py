#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Analyzer
Extracted from enhanced_analyzer.py and merged with custom test cases
"""

import json
import os
import argparse
import time
from typing import List, Dict, Any
from enhanced_analyzer import EnhancedAnalyzer

class ComprehensiveTestSuite:
    """Complete test suite combining pattern and LLM test cases."""
    
    def __init__(self, save_to_db: bool = False):
        self.analyzer = EnhancedAnalyzer()
        self.save_to_db = save_to_db
    
    def get_category_examples(self) -> Dict[str, Dict]:
        """Get examples for each category for testing specific categories."""
        return {
            'hate_speech': {
                'content': 'Los moros nos están invadiendo 🤬 Fuera de España!! No queremos más mezquitas aquí',
                'expected_category': 'hate_speech'
            },
            'disinformation': {
                'content': 'Las vacunas COVID contienen microchips 5G para controlarnos. El gobierno oculta la verdad',
                'expected_category': 'disinformation'
            },
            'conspiracy_theory': {
                'content': 'El 11M fue un montaje del gobierno socialista para ganar las elecciones. Soros está detrás',
                'expected_category': 'conspiracy_theory'
            },
            'far_right_bias': {
                'content': 'Los ROJOS han convertido España en Venezuela 🇻🇪 Solo Vox puede salvarnos de estos COMUNISTAS!',
                'expected_category': 'far_right_bias'
            },
            'call_to_action': {
                'content': 'Mañana todos a la calle! Organizaos y difunde este mensaje. Es hora de actuar contra la tiranía!',
                'expected_category': 'call_to_action'
            },
            'general': {
                'content': 'Hoy hace un día muy bonito en Madrid. Perfecto para pasear por el Retiro',
                'expected_category': 'general'
            }
        }
    
    def run_category_test(self, categories: List[str] = None, save_to_db: bool = True) -> List[Dict]:
        """
        Run tests for specific categories or all categories.
        
        Args:
            categories: List of categories to test. If None, tests all categories.
            save_to_db: Whether to save results to database
            
        Returns:
            List of test results
        """
        available_examples = self.get_category_examples()
        
        # If no specific categories requested, test all
        if categories is None:
            categories = list(available_examples.keys())
        
        print(f"🧪 TESTING CATEGORIES: {', '.join(categories)}")
        print("=" * 60)
        
        results = []
        success_count = 0
        
        for category in categories:
            if category not in available_examples:
                print(f"❌ Category '{category}' not available. Available: {list(available_examples.keys())}")
                continue
                
            example = available_examples[category]
            content = example['content']
            expected_category = example['expected_category']
            
            print(f"\n🔍 Testing category: {category}")
            print(f"📝 Content: {content[:80]}...")
            
            try:
                # Analyze content
                start_time = time.time()
                analysis = self.analyzer.analyze_content(
                    tweet_id=f"{category}_test",
                    tweet_url=f"https://example.com/{category}",
                    username="test_user",
                    content=content
                )
                end_time = time.time()
                
                # Check result
                actual_category = analysis.category
                is_success = actual_category == expected_category
                
                if is_success:
                    success_count += 1
                    print(f"✅ SUCCESS: {expected_category} → {actual_category}")
                else:
                    print(f"❌ FAILED: Expected {expected_category}, got {actual_category}")
                
                # Save to database if requested
                if save_to_db:
                    from enhanced_analyzer import save_content_analysis
                    save_content_analysis(analysis)
                
                # Store result
                result = {
                    'category': category,
                    'content': content,
                    'expected_category': expected_category,
                    'actual_category': actual_category,
                    'success': is_success,
                    'analysis_method': analysis.analysis_method,
                    'processing_time': round(end_time - start_time, 2),
                    'explanation': analysis.llm_explanation
                }
                results.append(result)
                
            except Exception as e:
                print(f"❌ ERROR testing {category}: {e}")
                results.append({
                    'category': category,
                    'content': content,
                    'expected_category': expected_category,
                    'actual_category': 'error',
                    'success': False,
                    'error': str(e)
                })
        
        # Print summary
        print(f"\n📊 CATEGORY TEST RESULTS")
        print("=" * 40)
        print(f"✅ Success: {success_count}/{len(categories)}")
        print(f"📈 Success Rate: {(success_count/len(categories)*100):.1f}%")
        
        if success_count < len(categories):
            failed_tests = [r for r in results if not r.get('success', False)]
            print(f"\n❌ Failed Tests: {len(failed_tests)}")
            for test in failed_tests:
                print(f"  - {test['category']}: Expected {test['expected_category']}, got {test['actual_category']}")
        
        return results
        
    def get_pattern_based_tests(self) -> Dict[str, Dict]:
        """Get test cases that should be caught by pattern detection."""
        return {
            'hate_speech_pattern_1': {
                'tweet_id': 'hate_001',
                'username': 'supremacista_es',
                'content': 'Los moros nos están invadiendo 🤬 Fuera de España!! No queremos más mezquitas aquí',
                'expected_category': 'hate_speech',
                'analysis_method': 'pattern',
                'description': 'Direct hate speech with racial slurs'
            },
            'hate_speech_pattern_2': {
                'tweet_id': 'hate_002',
                'username': 'test_user',
                'content': 'Ya estoy harto de estos menas que no hacen nada más que robar y traficar 😡 Devolvédlos a su país YA!',
                'expected_category': 'hate_speech',
                'analysis_method': 'pattern',
                'description': 'Hate speech targeting minors using derogatory terms'
            },
            'hate_speech_pattern_3': {
                'tweet_id': 'hate_003',
                'username': 'test_user',
                'content': 'Las feminazis han destruido este país. No son mujeres, son una plaga que hay que exterminar',
                'expected_category': 'hate_speech',
                'analysis_method': 'pattern',
                'description': 'Anti-feminist hate speech with violent language'
            },
            
            'disinformation_pattern_1': {
                'tweet_id': 'disinfo_001',
                'username': 'fake_news_es',
                'content': '🚨 URGENTE!! Mi cuñado médico me confirma: las vacunas COVID tienen grafeno para controlarnos con 5G!! Compartid antes de que lo censuren!!',
                'expected_category': 'disinformation',
                'analysis_method': 'pattern',
                'description': 'COVID vaccine misinformation with false authority'
            },
            'disinformation_pattern_2': {
                'tweet_id': 'disinfo_002',
                'username': 'test_user',
                'content': 'Los medios OCULTAN la verdad: 9 de cada 10 casos de COVID son INVENTADOS para meter miedo!! Lo dice un estudio de Harvard que después censuraron',
                'expected_category': ['disinformation', 'conspiracy_theory'],  # Both reasonable - media conspiracy elements
                'analysis_method': 'pattern',
                'description': 'False statistics with fake academic authority'
            },
            'disinformation_pattern_3': {
                'tweet_id': 'disinfo_003',
                'username': 'test_user',
                'content': 'El cambio climático es el mayor ENGAÑO de la historia! Los datos están manipulados por los globalistas para robarnos más dinero',
                'expected_category': ['disinformation', 'conspiracy_theory'],  # Both reasonable - conspiracy elements about globalists
                'analysis_method': 'pattern',
                'description': 'Climate change denial disinformation'
            },
            
            'conspiracy_theory_pattern_1': {
                'tweet_id': 'conspiracy_001',
                'username': 'conspiranoia_es',
                'content': 'Soros está detrás de todo! 💰 Financia las pateras para destruir Europa #PlanKalergi',
                'expected_category': 'conspiracy_theory',
                'analysis_method': 'pattern',
                'description': 'Great Replacement conspiracy with Soros'
            },
            'conspiracy_theory_pattern_2': {
                'tweet_id': 'conspiracy_002',
                'username': 'test_user',
                'content': 'Bill Gates, Soros y el Foro de Davos se reunen en secreto para decidir nuestro futuro 👁️ Despertad ovejas!!',
                'expected_category': 'conspiracy_theory',
                'analysis_method': 'pattern',
                'description': 'Elite coordination conspiracy theory'
            },
            
            'far_right_bias_pattern_1': {
                'tweet_id': 'bias_001',
                'username': 'partidista_extremo',
                'content': 'Los ROJOS han convertido España en Venezuela 🇻🇪 Solo Vox puede salvarnos de estos COMUNISTAS!',
                'expected_category': 'far_right_bias',
                'analysis_method': 'pattern',
                'description': 'Extreme partisan rhetoric against socialists'
            },
            'far_right_bias_pattern_2': {
                'tweet_id': 'bias_002',
                'username': 'test_user',
                'content': 'La agenda WOKE está destruyendo nuestros valores cristianos!! 😡 Nos quieren convertir en maricas y feminazis',
                'expected_category': ['far_right_bias', 'hate_speech'],  # Both reasonable due to slurs
                'analysis_method': 'pattern',
                'description': 'Anti-woke far-right bias with homophobic language'
            },
            'far_right_bias_pattern_3': {
                'tweet_id': 'bias_003',
                'username': 'test_user',
                'content': 'La inmigración masiva es una invasión planificada para destruir nuestra raza y nuestra cultura católica',
                'expected_category': ['far_right_bias', 'hate_speech'],  # Both reasonable
                'analysis_method': 'pattern',
                'description': 'Immigration threat narrative with racial elements'
            },
            
            'call_to_action_pattern_1': {
                'tweet_id': 'action_001',
                'username': 'organizador_patriota',
                'content': '🚨 CONVOCATORIA URGENTE!! Todos a Cibeles HOY 18:00h!! A defender España de la INVASIÓN!! RT MASIVO!!',
                'expected_category': 'call_to_action',
                'analysis_method': 'pattern',
                'description': 'Direct mobilization call with location and time'
            },
            'call_to_action_pattern_2': {
                'tweet_id': 'action_002',
                'username': 'test_user',
                'content': 'Patriotas! Es hora de ORGANIZARSE!! 💪 Uníos a nosotros para salvar España!',
                'expected_category': 'call_to_action',
                'analysis_method': 'pattern',
                'description': 'Patriotic organization call'
            },
            'call_to_action_pattern_3': {
                'tweet_id': 'action_003',
                'username': 'test_user',
                'content': 'Padres de familia!! RETIRAD a vuestros hijos de los colegios que enseñan ideología de género!! Actuad YA!',
                'expected_category': 'call_to_action',
                'analysis_method': 'pattern',
                'description': 'Parental action call against LGBTQ education'
            },
            'call_to_action_pattern_4': {
                'tweet_id': 'action_004',
                'username': 'test_user',
                'content': 'BASTA YA!! La gente tiene que unirse y salir a las calles antes de que sea demasiado tarde!! #Resistencia',
                'expected_category': 'call_to_action',
                'analysis_method': 'pattern',
                'description': 'Urgent street mobilization call'
            },
            'call_to_action_pattern_5': {
                'tweet_id': 'action_005',
                'username': 'test_user',
                'content': 'Los verdaderos españoles debemos recuperar nuestras instituciones. MOVILIZAOS ahora o las perderemos para siempre!',
                'expected_category': 'call_to_action',
                'analysis_method': 'pattern',
                'description': 'Institutional takeover mobilization call'
            },
            
            # Additional challenging pattern-based tests
            'hate_speech_pattern_4': {
                'tweet_id': 'hate_004',
                'username': 'test_user',
                'content': 'Los gitanos son genéticamente ladrones y mentirosos. No pueden cambiar, está en su ADN de parásitos',
                'expected_category': 'hate_speech',
                'analysis_method': 'pattern',
                'description': 'Genetic hate speech targeting Roma people'
            },
            'hate_speech_pattern_5': {
                'tweet_id': 'hate_005',
                'username': 'test_user',
                'content': 'Las tías están para fregar, parir y callarse. Eso del feminismo son tonterías de brujas histéricas 🙄',
                'expected_category': ['hate_speech', 'far_right_bias'],  # Both reasonable - misogynistic bias content
                'analysis_method': 'pattern',
                'description': 'Misogynistic hate speech with gender roles'
            },
            
            'disinformation_pattern_4': {
                'tweet_id': 'disinfo_004',
                'username': 'test_user',
                'content': 'El calentamiento global es una MENTIRA!! Los datos están trucados!! En los 70 decían que venía una era glacial 🧊❄️',
                'expected_category': ['disinformation', 'conspiracy_theory'],  # Both reasonable - conspiracy about data manipulation
                'analysis_method': 'pattern',
                'description': 'Climate denial disinformation with false historical claims'
            },
            'disinformation_pattern_5': {
                'tweet_id': 'disinfo_005',
                'username': 'test_user',
                'content': 'Big Pharma lleva 50 años ocultando la cura del cáncer!! Tienen la medicina pero prefieren que muramos para seguir vendiendo quimio!!',
                'expected_category': ['disinformation', 'conspiracy_theory'],  # Both reasonable
                'analysis_method': 'pattern',
                'description': 'Medical conspiracy disinformation'
            },
            
            'conspiracy_theory_pattern_3': {
                'tweet_id': 'conspiracy_003',
                'username': 'test_user',
                'content': 'Todos los medios están comprados!! CNN, BBC, El País... todos dicen lo mismo porque tienen el mismo dueño: los Rothschild 💰',
                'expected_category': 'conspiracy_theory',
                'analysis_method': 'pattern',
                'description': 'Media ownership conspiracy with antisemitic elements'
            },
            'conspiracy_theory_pattern_4': {
                'tweet_id': 'conspiracy_004',
                'username': 'test_user',
                'content': 'La pandemia fue PLANIFICADA!! 🦠 Las élites ya tenían todo preparado para controlarnos y quitarnos la libertad!! #Plandemia',
                'expected_category': 'conspiracy_theory',
                'analysis_method': 'pattern',
                'description': 'Pandemic planning conspiracy theory'
            },
            
            'far_right_bias_pattern_4': {
                'tweet_id': 'bias_004',
                'username': 'test_user',
                'content': 'El feminismo es cáncer puro!! Han convertido a las mujeres en locas histéricas que odian a los hombres y la familia 😡',
                'expected_category': ['far_right_bias', 'hate_speech'],  # Both reasonable due to aggressive language
                'analysis_method': 'pattern',
                'description': 'Anti-feminist far-right bias with aggressive language'
            },
            'far_right_bias_pattern_5': {
                'tweet_id': 'bias_005',
                'username': 'test_user',
                'content': 'En los colegios están ADOCTRINANDO a nuestros niños!! Les dicen que pueden ser niñas si quieren!! Esto es pedofilia LGBTI!!',
                'expected_category': ['far_right_bias', 'hate_speech', 'call_to_action'],  # All reasonable - bias, hate speech, and implicit call to action
                'analysis_method': 'pattern',
                'description': 'Anti-LGBTQ far-right bias targeting schools'
            },
            
            'call_to_action_pattern_6': {
                'tweet_id': 'action_006',
                'username': 'test_user',
                'content': 'BOICOT TOTAL a Coca-Cola, Nike y todas las empresas WOKE!! 🚫 Que se arruinen por apoyar la agenda globalista!! #BoicotWoke',
                'expected_category': ['call_to_action', 'conspiracy_theory'],  # Both reasonable - boycott call + conspiracy elements
                'analysis_method': 'pattern',
                'description': 'Economic boycott call against woke companies'
            },
            'call_to_action_pattern_7': {
                'tweet_id': 'action_007',
                'username': 'test_user',
                'content': 'Padres!! SACAD a vuestros hijos de esos colegios progres!! Homeschooling o colegios católicos!! Proteged su inocencia!!',
                'expected_category': 'call_to_action',
                'analysis_method': 'pattern',
                'description': 'Educational withdrawal call for religious alternatives'
            }
        }
    
    def get_llm_fallback_tests(self) -> Dict[str, Dict]:
        """Get test cases requiring LLM analysis (subtle content)."""
        return {
            'hate_speech_llm_1': {
                'tweet_id': 'hate_llm_001',
                'username': 'test_user_llm',
                'content': 'Los sudacas que vienen aquí son todos unos vagos. Vienen a quitarnos el trabajo y las ayudas que nos corresponden a nosotros',
                'expected_category': 'hate_speech',
                'analysis_method': 'llm',
                'description': 'Hate speech with derogatory terms for Latin Americans'
            },
            
            'disinformation_llm_1': {
                'tweet_id': 'disinfo_llm_001',
                'username': 'test_user_llm',
                'content': 'Mi vecino que trabaja en Pfizer me ha dicho que las vacunas tienen efectos que no salen en los medios. Muy fuerte lo que me ha contado...',
                'expected_category': 'disinformation',
                'analysis_method': 'llm',
                'description': 'Vaccine disinformation using anecdotal false authority'
            },
            
            'conspiracy_theory_llm_1': {
                'tweet_id': 'conspiracy_llm_001',
                'username': 'test_user_llm',
                'content': 'Qué casualidad que todas las crisis benefician siempre a los mismos bancos y multinacionales... 🤔 Ya sabéis quién mueve los hilos',
                'expected_category': ['conspiracy_theory', 'disinformation'],  # Both reasonable
                'analysis_method': 'llm',
                'description': 'Economic elite conspiracy theory'
            },
            'conspiracy_theory_llm_2': {
                'tweet_id': 'conspiracy_llm_002',
                'username': 'test_user_llm',
                'content': 'Casualmente cada vez que hay "pandemia" las farmacéuticas hacen el negocio del siglo 💰 Qué raro todo, ¿verdad?',
                'expected_category': ['conspiracy_theory', 'disinformation'],  # Both reasonable
                'analysis_method': 'llm',
                'description': 'Pharmaceutical conspiracy with sarcasm'
            },
            'conspiracy_theory_llm_3': {
                'tweet_id': 'conspiracy_llm_003',
                'username': 'test_user_llm',
                'content': 'Google, Facebook y Twitter censuran todo lo que no les conviene 🤐 Controlan la información para mantenernos borregos',
                'expected_category': ['conspiracy_theory', 'disinformation'],  # Both reasonable
                'analysis_method': 'llm',
                'description': 'Tech censorship conspiracy theory'
            },
            
            'far_right_bias_llm_1': {
                'tweet_id': 'bias_llm_001',
                'username': 'test_user_llm',
                'content': 'Los progres están cambiando España y no para bien. Cada día hay más locuras woke en las instituciones',
                'expected_category': 'far_right_bias',
                'analysis_method': 'llm',
                'description': 'Anti-progressive bias with woke criticism'
            },
            'far_right_bias_llm_2': {
                'tweet_id': 'bias_llm_002',
                'username': 'test_user_llm',
                'content': 'En los colegios ya no enseñan historia de España, solo propaganda progresista. Están lavando el cerebro a nuestros hijos',
                'expected_category': ['far_right_bias', 'disinformation'],  # Both reasonable
                'analysis_method': 'llm',
                'description': 'Educational indoctrination bias'
            },
            'far_right_bias_llm_3': {
                'tweet_id': 'bias_llm_003',
                'username': 'test_user_llm',
                'content': 'La cultura occidental se está perdiendo por culpa del multiculturalismo forzado. Nos están borrando del mapa 😔',
                'expected_category': ['far_right_bias', 'conspiracy_theory'],  # Both reasonable
                'analysis_method': 'llm',
                'description': 'Cultural replacement conspiracy theory'
            },
            
            'call_to_action_llm_1': {
                'tweet_id': 'action_llm_001',
                'username': 'test_user_llm',
                'content': 'Españoles de bien!! Es hora de despertar y hacer algo antes de que sea tarde! Organizaos en vuestras ciudades!!',
                'expected_category': 'call_to_action',
                'analysis_method': 'llm',
                'description': 'General mobilization call to action'
            },
            
            # More realistic LLM-based tests (less subtle, more Twitter-like)
            'hate_speech_llm_2': {
                'tweet_id': 'hate_llm_002',
                'username': 'test_user_llm',
                'content': 'Los menas son todos delincuentes que vienen aquí a vivir del cuento. Están destruyendo nuestros barrios.',
                'expected_category': 'hate_speech',
                'analysis_method': 'llm',
                'description': 'Direct hate speech using derogatory terms'
            },
            'hate_speech_llm_3': {
                'tweet_id': 'hate_llm_003',
                'username': 'test_user_llm',
                'content': 'Las feminazis han convertido a los hombres en ciudadanos de segunda clase. Ya no podemos ni opinar.',
                'expected_category': 'hate_speech',
                'analysis_method': 'llm',
                'description': 'Anti-feminist hate speech with gendered slurs'
            },
            
            'disinformation_llm_2': {
                'tweet_id': 'disinfo_llm_002',
                'username': 'test_user_llm',
                'content': 'El gobierno está inflando los números de la inflación para justificar sus políticas desastrosas. Los datos reales son el doble.',
                'expected_category': 'disinformation',
                'analysis_method': 'llm',
                'description': 'Economic disinformation with false claims about statistics'
            },
            'disinformation_llm_3': {
                'tweet_id': 'disinfo_llm_003',
                'username': 'test_user_llm',
                'content': 'Mi primo trabaja en el hospital y me dice que la mayoría de casos COVID son inventados para cobrar más dinero.',
                'expected_category': 'disinformation',
                'analysis_method': 'llm',
                'description': 'Healthcare disinformation using anecdotal false evidence'
            },
            
            'conspiracy_theory_llm_4': {
                'tweet_id': 'conspiracy_llm_004',
                'username': 'test_user_llm',
                'content': 'Las farmacéuticas y Bill Gates están detrás de la pandemia para vendernos las vacunas y controlarnos.',
                'expected_category': ['conspiracy_theory', 'disinformation'],  # Both reasonable
                'analysis_method': 'llm',
                'description': 'COVID conspiracy theory with specific actors - can be classified as either conspiracy or disinformation'
            },
            'conspiracy_theory_llm_5': {
                'tweet_id': 'conspiracy_llm_005',
                'username': 'test_user_llm',
                'content': 'Los chemtrails son reales. Solo hay que mirar al cielo y ver esas líneas que nos están fumigando a todos.',
                'expected_category': 'conspiracy_theory',
                'analysis_method': 'llm',
                'description': 'Chemtrails conspiracy theory'
            },
            
            'far_right_bias_llm_4': {
                'tweet_id': 'bias_llm_004',
                'username': 'test_user_llm',
                'content': 'Vox es el único partido que defiende a los españoles de verdad. El resto son traidores vendidos al globalismo.',
                'expected_category': 'far_right_bias',
                'analysis_method': 'llm',
                'description': 'Direct far-right bias with partisan extremism'
            },
            'far_right_bias_llm_5': {
                'tweet_id': 'bias_llm_005',
                'username': 'test_user_llm',
                'content': 'Los inmigrantes ilegales reciben más ayudas que los españoles. Esto es una invasión financiada por los rojos.',
                'expected_category': ['far_right_bias', 'hate_speech'],  # Both valid - contains "invasión" hate speech + far-right bias elements
                'analysis_method': 'llm',
                'description': 'Immigration-focused far-right bias with hate speech language'
            },
            
            'call_to_action_llm_2': {
                'tweet_id': 'action_llm_002',
                'username': 'test_user_llm',
                'content': 'Mañana todos los patriotas en Sol a las 19h. No podemos permitir que nos sigan robando el país.',
                'expected_category': 'call_to_action',
                'analysis_method': 'llm',
                'description': 'Direct mobilization call with patriotic rhetoric'
            },
            'call_to_action_llm_3': {
                'tweet_id': 'action_llm_003',
                'username': 'test_user_llm',
                'content': 'Hay que votar en masa a la única opción que nos queda. Difunde este mensaje que el futuro de España depende de nosotros.',
                'expected_category': 'call_to_action',
                'analysis_method': 'llm',
                'description': 'Electoral mobilization call with sharing request'
            }
        }
    
    def get_neutral_tests(self) -> Dict[str, Dict]:
        """Get genuinely neutral content tests."""
        return {
            'general_1': {
                'tweet_id': 'general_001',
                'username': 'ciudadano_normal',
                'content': 'Qué día más bueno!! ☀️ Me voy al Retiro con la familia a disfrutar del domingo',
                'expected_category': 'general',
                'analysis_method': 'pattern',
                'description': 'Weather and family activities with emoji'
            },
            'general_2': {
                'tweet_id': 'general_002',
                'username': 'test_user',
                'content': 'Acabo de probar el nuevo restaurante japonés del barrio 🍜 La comida buenísima!! Lo recomiendo',
                'expected_category': 'general',
                'analysis_method': 'pattern',
                'description': 'Restaurant recommendation with emoji'
            },
            'general_3': {
                'tweet_id': 'general_003',
                'username': 'test_user',
                'content': 'Por fin empieza mi serie favorita!! 📺 Llevaba meses esperando este momento #Netflix',
                'expected_category': 'general',
                'analysis_method': 'pattern',
                'description': 'Entertainment content with emoji and hashtag'
            },
            
            # Additional neutral tests with Twitter-like format
            'general_4': {
                'tweet_id': 'general_004',
                'username': 'test_user',
                'content': 'Interesante el debate de anoche en la tele. Cada partido tiene su visión del futuro de España 🇪🇸',
                'expected_category': 'general',
                'analysis_method': 'pattern',
                'description': 'Neutral political observation'
            },
            'general_5': {
                'tweet_id': 'general_005',
                'username': 'test_user',
                'content': 'Mi carrera de economía me está ayudando mucho a entender cómo funcionan los mercados 📈',
                'expected_category': 'general',
                'analysis_method': 'pattern',
                'description': 'Personal educational experience'
            },
            'general_6': {
                'tweet_id': 'general_006',
                'username': 'test_user',
                'content': 'Las ciudades multiculturales tienen cosas buenas y malas, como todo en la vida. Hay que encontrar el equilibrio',
                'expected_category': 'general',
                'analysis_method': 'pattern',
                'description': 'Balanced view on cultural diversity'
            },
            'general_7': {
                'tweet_id': 'general_007',
                'username': 'test_user',
                'content': 'Cada familia celebra las fiestas de manera diferente. En Andalucía hacemos unas paellas que... 🥘😋',
                'expected_category': 'general',
                'analysis_method': 'pattern',
                'description': 'Personal family traditions'
            },
            'general_8': {
                'tweet_id': 'general_008',
                'username': 'test_user',
                'content': 'La educación tiene que evolucionar, pero sin perder lo bueno de antes. Todo es encontrar el equilibrio 🎓',
                'expected_category': 'general',
                'analysis_method': 'pattern',
                'description': 'Balanced educational opinion'
            },
            
            # Edge cases that could be tricky to classify
            'edge_case_1': {
                'tweet_id': 'edge_001',
                'username': 'test_user',
                'content': 'He leído algunos estudios que dicen que cada método de enseñanza funciona diferente según el niño 🤔',
                'expected_category': 'general',
                'analysis_method': 'llm',
                'description': 'Neutral educational observation'
            },
            'edge_case_2': {
                'tweet_id': 'edge_002',
                'username': 'test_user',
                'content': 'Es curioso cómo las subidas de impuestos afectan más a unos sectores que a otros. La economía es complicada 📊',
                'expected_category': 'general',
                'analysis_method': 'llm',
                'description': 'Neutral economic observation'
            },
            'edge_case_3': {
                'tweet_id': 'edge_003',
                'username': 'test_user',
                'content': 'Los ciudadanos deberían informarse bien antes de votar. Es importante para la democracia 🗳️',
                'expected_category': 'general',
                'analysis_method': 'llm',
                'description': 'Civic responsibility message'
            }
        }
    
    def _analyze_test_batch(self, test_cases: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze a batch of test cases efficiently."""
        results = []
        
        # Process tests in batches to reduce LLM loading overhead
        for test_name, test_case in test_cases.items():
            print(f"⚡ {test_name[:20]}...", end=" ", flush=True)
            
            try:
                # Analyze content with speed optimizations
                analysis = self.analyzer.analyze_content(
                    tweet_id=test_case['tweet_id'],
                    tweet_url=f"https://twitter.com/{test_case['username']}/status/{test_case['tweet_id']}",
                    username=test_case['username'],
                    content=test_case['content'],
                    retrieve_evidence=False  # Skip evidence retrieval for speed
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
                
                status = 'PASS' if is_correct else 'FAIL'
                
                print("✅" if is_correct else "❌", end=" ", flush=True)
                
                results.append({
                    'test_name': test_name,
                    'content': test_case['content'],
                    'expected': expected,
                    'actual': actual_category,
                    'status': status,
                    'method_used': analysis.analysis_method if hasattr(analysis, 'analysis_method') else 'unknown'
                })
                
            except Exception as e:
                print("💥", end=" ", flush=True)
                
                results.append({
                    'test_name': test_name,
                    'content': test_case['content'],
                    'expected': test_case['expected_category'],
                    'actual': 'ERROR',
                    'status': 'ERROR',
                    'error': str(e)
                })
        
        print()  # New line after batch
        
        return results
    
    def run_pattern_tests(self) -> Dict[str, Any]:
        """Run tests that should be caught by pattern detection."""
        print("🔍 TESTING PATTERN-BASED DETECTION")
        print("=" * 60)
        
        test_cases = self.get_pattern_based_tests()
        print(f"⚡ Running {len(test_cases)} pattern tests...")
        
        test_results = self._analyze_test_batch(test_cases)
        
        # Calculate results
        passed = len([r for r in test_results if r['status'] == 'PASS'])
        failed = len([r for r in test_results if r['status'] in ['FAIL', 'ERROR']])
        
        results = {
            'category': 'pattern_based',
            'total_tests': len(test_cases),
            'passed': passed,
            'failed': failed,
            'test_results': test_results
        }
        
        print(f"\n📊 Pattern Tests: {results['passed']}/{results['total_tests']} passed ({(results['passed']/results['total_tests'])*100:.1f}%)")
        return results
    
    def run_llm_tests(self) -> Dict[str, Any]:
        """Run tests requiring LLM analysis."""
        print("\n🧠 TESTING LLM-BASED CLASSIFICATION")
        print("=" * 60)
        
        test_cases = self.get_llm_fallback_tests()
        print(f"⚡ Running {len(test_cases)} LLM tests...")
        
        test_results = self._analyze_test_batch(test_cases)
        
        # Calculate results
        passed = len([r for r in test_results if r['status'] == 'PASS'])
        failed = len([r for r in test_results if r['status'] in ['FAIL', 'ERROR']])
        
        results = {
            'category': 'llm_based',
            'total_tests': len(test_cases),
            'passed': passed,
            'failed': failed,
            'test_results': test_results
        }
        
        print(f"📊 LLM Tests: {results['passed']}/{results['total_tests']} passed ({(results['passed']/results['total_tests'])*100:.1f}%)")
        return results
    
    def run_neutral_tests(self) -> Dict[str, Any]:
        """Run tests for genuinely neutral content."""
        print("\n🌍 TESTING NEUTRAL CONTENT CLASSIFICATION")
        print("=" * 60)
        
        test_cases = self.get_neutral_tests()
        print(f"⚡ Running {len(test_cases)} neutral tests...")
        
        test_results = self._analyze_test_batch(test_cases)
        
        # Calculate results
        passed = len([r for r in test_results if r['status'] == 'PASS'])
        failed = len([r for r in test_results if r['status'] in ['FAIL', 'ERROR']])
        
        results = {
            'category': 'neutral_content',
            'total_tests': len(test_cases),
            'passed': passed,
            'failed': failed,
            'test_results': test_results
        }
        
        print(f"📊 Neutral Tests: {results['passed']}/{results['total_tests']} passed ({(results['passed']/results['total_tests'])*100:.1f}%)")
        return results
    
    def run_comprehensive_suite(self) -> Dict[str, Any]:
        """Run all test categories."""
        start_time = time.time()
        
        print("🧪 COMPREHENSIVE TEST SUITE")
        print("=" * 70)
        print("Testing pattern detection, LLM classification, and neutral content handling")
        print()
        
        # Run all test categories
        pattern_start = time.time()
        pattern_results = self.run_pattern_tests()
        pattern_time = time.time() - pattern_start
        
        llm_start = time.time()
        llm_results = self.run_llm_tests()
        llm_time = time.time() - llm_start
        
        neutral_start = time.time()
        neutral_results = self.run_neutral_tests()
        neutral_time = time.time() - neutral_start
        
        total_time = time.time() - start_time
        
        # Compile comprehensive results
        total_tests = pattern_results['total_tests'] + llm_results['total_tests'] + neutral_results['total_tests']
        total_passed = pattern_results['passed'] + llm_results['passed'] + neutral_results['passed']
        total_failed = pattern_results['failed'] + llm_results['failed'] + neutral_results['failed']
        
        comprehensive_results = {
            'suite_name': 'comprehensive_analyzer_test',
            'timestamp': time.time(),
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'success_rate': (total_passed / total_tests) * 100 if total_tests > 0 else 0,
            'execution_time': {
                'total_seconds': round(total_time, 2),
                'pattern_tests_seconds': round(pattern_time, 2),
                'llm_tests_seconds': round(llm_time, 2),
                'neutral_tests_seconds': round(neutral_time, 2),
                'tests_per_second': round(total_tests / total_time, 2) if total_time > 0 else 0
            },
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
        print(f"📊 Pattern-based Tests: {pattern_results['passed']}/{pattern_results['total_tests']} ({(pattern_results['passed']/pattern_results['total_tests'])*100:.1f}%) - {pattern_time:.2f}s")
        print(f"🧠 LLM-based Tests: {llm_results['passed']}/{llm_results['total_tests']} ({(llm_results['passed']/llm_results['total_tests'])*100:.1f}%) - {llm_time:.2f}s")
        print(f"🌍 Neutral Content Tests: {neutral_results['passed']}/{neutral_results['total_tests']} ({(neutral_results['passed']/neutral_results['total_tests'])*100:.1f}%) - {neutral_time:.2f}s")
        print(f"🎯 OVERALL SUCCESS RATE: {total_passed}/{total_tests} ({comprehensive_results['success_rate']:.1f}%)")
        print(f"⏱️  TOTAL EXECUTION TIME: {total_time:.2f} seconds ({comprehensive_results['execution_time']['tests_per_second']:.1f} tests/sec)")
        
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
    
    def run_failed_tests_only(self, results_file_path: str = None) -> Dict[str, Any]:
        """Re-run only the previously failed test cases for faster iteration."""
        start_time = time.time()
        
        print("🔄 RE-RUNNING FAILED TESTS ONLY")
        print("=" * 60)
        print()
        
        # Load previous results
        if results_file_path is None:
            results_file_path = os.path.join(os.path.dirname(__file__), 'comprehensive_test_results.json')
        
        try:
            with open(results_file_path, 'r', encoding='utf-8') as f:
                previous_results = json.load(f)
        except FileNotFoundError:
            print(f"❌ Results file not found: {results_file_path}")
            print("🔍 Please run the full comprehensive suite first.")
            return {}
        except Exception as e:
            print(f"❌ Error loading results file: {e}")
            return {}
        
        # Extract failed test names from all categories
        failed_tests = []
        for category_name, category_data in previous_results['results_by_category'].items():
            for test_result in category_data['test_results']:
                if test_result['status'] == 'FAIL':
                    failed_tests.append({
                        'name': test_result['test_name'],
                        'category': category_name,
                        'expected': test_result['expected'],
                        'actual': test_result['actual']
                    })
        
        if not failed_tests:
            print("🎉 No failed tests found in previous results!")
            return {}
        
        print(f"🔍 Found {len(failed_tests)} failed tests:")
        for test in failed_tests:
            print(f"   • {test['name']} ({test['category']}) - Expected: {test['expected']}, Got: {test['actual']}")
        print()
        
        # Get all test cases
        all_pattern_tests = self.get_pattern_based_tests()
        all_llm_tests = self.get_llm_fallback_tests()
        all_neutral_tests = self.get_neutral_tests()
        
        # Combine all tests
        all_tests = {**all_pattern_tests, **all_llm_tests, **all_neutral_tests}
        
        # Run only the failed tests
        rerun_results = {
            'suite_name': 'failed_tests_rerun',
            'timestamp': time.time(),
            'original_failed_count': len(failed_tests),
            'now_passed': 0,
            'still_failed': 0,
            'test_results': []
        }
        
        tests_start_time = time.time()
        
        for failed_test in failed_tests:
            test_name = failed_test['name']
            if test_name not in all_tests:
                print(f"⚠️  Test case not found: {test_name}")
                continue
            
            test_case = all_tests[test_name]
            
            print(f"\n🔄 RE-TESTING: {test_name}")
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
                    print(f"✅ NOW PASS - Got: {actual_category}")
                    rerun_results['now_passed'] += 1
                    status = 'NOW_PASS'
                else:
                    print(f"❌ STILL FAIL - Expected: {expected_str}, Got: {actual_category}")
                    rerun_results['still_failed'] += 1
                    status = 'STILL_FAIL'
                
                rerun_results['test_results'].append({
                    'test_name': test_name,
                    'content': test_case['content'],
                    'expected': expected,
                    'actual': actual_category,
                    'status': status,
                    'method_used': analysis.analysis_method if hasattr(analysis, 'analysis_method') else 'unknown',
                    'original_result': failed_test['actual']
                })
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
                rerun_results['still_failed'] += 1
                rerun_results['test_results'].append({
                    'test_name': test_name,
                    'content': test_case['content'],
                    'expected': test_case['expected_category'],
                    'actual': 'ERROR',
                    'status': 'ERROR',
                    'error': str(e),
                    'original_result': failed_test['actual']
                })
        
        tests_time = time.time() - tests_start_time
        total_time = time.time() - start_time
        
        # Add timing information
        rerun_results['execution_time'] = {
            'total_seconds': round(total_time, 2),
            'tests_seconds': round(tests_time, 2),
            'tests_per_second': round(len(failed_tests) / tests_time, 2) if tests_time > 0 else 0
        }
        
        # Summary
        print(f"\n📊 FAILED TESTS RE-RUN RESULTS")
        print("=" * 50)
        print(f"🔄 Originally failed: {rerun_results['original_failed_count']}")
        print(f"✅ Now passing: {rerun_results['now_passed']}")
        print(f"❌ Still failing: {rerun_results['still_failed']}")
        print(f"⏱️  Execution time: {total_time:.2f} seconds ({rerun_results['execution_time']['tests_per_second']:.1f} tests/sec)")
        
        if rerun_results['now_passed'] > 0:
            print(f"🎉 Improvement! {rerun_results['now_passed']} tests now pass!")
        
        if rerun_results['still_failed'] == 0:
            print("🏆 All previously failed tests now pass!")
        
        # Update the original comprehensive results file with new results
        try:
            # Update the original results with the new test results
            for rerun_test in rerun_results['test_results']:
                test_name = rerun_test['test_name']
                # Find the test in the original results and update it
                for category_name, category_data in previous_results['results_by_category'].items():
                    for i, original_test in enumerate(category_data['test_results']):
                        if original_test['test_name'] == test_name:
                            # Update the test result
                            previous_results['results_by_category'][category_name]['test_results'][i].update({
                                'status': 'PASS' if rerun_test['status'] == 'NOW_PASS' else rerun_test['status'],
                                'actual': rerun_test['actual'],
                                'method_used': rerun_test['method_used']
                            })
                            # Update category counters
                            if rerun_test['status'] == 'NOW_PASS' and original_test['status'] == 'FAIL':
                                previous_results['results_by_category'][category_name]['passed'] += 1
                                previous_results['results_by_category'][category_name]['failed'] -= 1
                            break
            
            # Update overall counters
            previous_results['total_passed'] = sum(cat['passed'] for cat in previous_results['results_by_category'].values())
            previous_results['total_failed'] = sum(cat['failed'] for cat in previous_results['results_by_category'].values())
            previous_results['success_rate'] = (previous_results['total_passed'] / previous_results['total_tests']) * 100
            previous_results['timestamp'] = time.time()
            
            # Save updated results to original file
            with open(results_file_path, 'w', encoding='utf-8') as f:
                json.dump(previous_results, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n💾 Updated results saved to: {results_file_path}")
            
            # Also save detailed rerun results for reference
            rerun_output_path = os.path.join(os.path.dirname(__file__), 'failed_tests_rerun_details.json')
            with open(rerun_output_path, 'w', encoding='utf-8') as f:
                json.dump(rerun_results, f, ensure_ascii=False, indent=2, default=str)
            print(f"💾 Detailed rerun results saved to: {rerun_output_path}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
        
        return rerun_results

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
    parser.add_argument('--failed-only', action='store_true',
                       help='Re-run only previously failed tests')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose mode with detailed output (disables fast mode)')
    parser.add_argument('--categories', nargs='*', 
                       choices=['hate_speech', 'disinformation', 'conspiracy_theory', 
                               'far_right_bias', 'call_to_action', 'general'],
                       help='Test specific categories (default: run full test suite)')
    parser.add_argument('--list-categories', action='store_true',
                       help='Show available categories and exit')
    
    args = parser.parse_args()
    
    # Initialize test suite
    test_suite = ComprehensiveTestSuite(save_to_db=args.save_to_db)
    
    # Handle list categories
    if args.list_categories:
        available_examples = test_suite.get_category_examples()
        print("📋 CATEGORÍAS DISPONIBLES PARA TESTING")
        print("=" * 50)
        for category, example in available_examples.items():
            print(f"🏷️ {category}")
            print(f"   📝 Ejemplo: {example['content'][:80]}...")
            print(f"   🎯 Espera: {example['expected_category']}")
            print()
        print("💡 Uso: --categories hate_speech disinformation")
        print("💡 Para todas: sin --categories (ejecuta suite completa)")
        return
    
    # Run category-specific tests if requested
    if args.categories:
        test_suite.run_category_test(categories=args.categories, save_to_db=args.save_to_db)
        return
    
    # Run specified tests
    if args.failed_only:
        test_suite.run_failed_tests_only()
    elif args.patterns_only:
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
