"""
Model Comparison Script for Spanish Far-Right Analysis
Co            # === DISINFORMATION ===
            TestExample(
                id="disinfo_001",
                text="El 85% de los inmigrantes no trabajan y cobran ayudas del estado. Estudio secreto del gobierno lo confirma.",
                category=Categories.DISINFORMATION,
                description="False statistics without credible sources",
                far_right_score=0.45,
            ),fferent LLM models with test examples to evaluate performance.
"""

import json
import time
import argparse
from typing import List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime

from llm_models import LLMModelConfig, EnhancedLLMPipeline
from categories import Categories

@dataclass
class TestExample:
    """Test example for model comparison."""
    text: str
    expected_category: str
    category: str
    description: str
    far_right_score: float = 0.5

@dataclass 
class ModelResult:
    """Result from a single model test."""
    model_name: str
    model_config: Dict
    example_id: str
    processing_time: float
    success: bool
    error: str = ""
    
    # LLM outputs
    llm_explanation: str = ""
    llm_categories: List[str] = None
    
    def __post_init__(self):
        if self.llm_categories is None:
            self.llm_categories = []

class ModelComparator:
    """Compare different LLM models on test examples."""
    
    def __init__(self):
        self.test_examples = self._create_test_examples()
        self.results = []
    
    def _create_test_examples(self) -> List[TestExample]:
        """Create diverse test examples for comparison - one from each major category."""
        return [
            # === HATE SPEECH ===
            TestExample(
                id="hate_001",
                text="Los musulmanes son una raza inferior que viene a destruir nuestra cultura cristiana.",
                category=Categories.HATE_SPEECH,
                description="Explicit racial hate speech against Muslims",
                far_right_score=0.85,
            ),
            
            # === DISINFORMATION ===
            TestExample(
                id="disinfo_001",
                text="El 85% de los inmigrantes no trabajan y cobran ayudas del estado. Estudio secreto del gobierno lo confirma.",
                category=Categories.DISINFORMATION,
                description="False statistics about immigration",
                far_right_score=0.70,
            ),
            
            # === CONSPIRACY THEORY ===
            TestExample(
                id="conspiracy_001",
                text="Soros financia la inmigración para reemplazar a los europeos. Es el plan Kalergi en acción.",
                category=Categories.CONSPIRACY_THEORY,
                description="Great replacement conspiracy theory",
                far_right_score=0.78,
            ),
            
            # === CALL TO ACTION ===
            TestExample(
                id="action_001",
                text="¡CONCENTRACIÓN HOY 18:00 en Cibeles! Hay que salir a las calles a defender España de la invasión.",
                category=Categories.CALL_TO_ACTION,
                description="Urgent mobilization with location and time",
                far_right_score=0.75,
            ),
            
            # === FAR-RIGHT BIAS ===
            TestExample(
                id="bias_001",
                text="España necesita un líder fuerte como Franco que ponga orden. La democracia ha fracasado.",
                category=Categories.FAR_RIGHT_BIAS,
                description="Authoritarian nostalgia",
                far_right_score=0.80,
            ),
            
            # === NORMAL CONTENT (Control) ===
            TestExample(
                id="normal_001",
                text="Qué día tan bonito hace hoy en Madrid. Me voy a dar un paseo por el Retiro con la familia.",
                category=Categories.GENERAL,
                description="Normal everyday content",
                far_right_score=0.0,
            )
        ]
    
    def run_comparison(self, models_to_test: List[str] = None, max_examples: int = None) -> List[ModelResult]:
        """Run comparison across specified models."""
        if models_to_test is None:
            # Test fast models by default
            models_to_test = LLMModelConfig.get_fast_models()
        
        examples_to_test = self.test_examples[:max_examples] if max_examples else self.test_examples
        
        print(f"🔬 INICIANDO COMPARACIÓN DE MODELOS")
        print(f"📊 Modelos a probar: {len(models_to_test)}")
        print(f"📝 Ejemplos de prueba: {len(examples_to_test)}")
        print("=" * 60)
        
        all_results = []
        
        for i, model_name in enumerate(models_to_test, 1):
            print(f"\n🤖 PROBANDO MODELO {i}/{len(models_to_test)}: {model_name}")
            print("-" * 40)
            
            model_results = self._test_single_model(model_name, examples_to_test)
            all_results.extend(model_results)
        
        self.results = all_results
        return all_results
    
    def _test_single_model(self, model_name: str, examples: List[TestExample]) -> List[ModelResult]:
        """Test a single model against all examples."""
        model_results = []
        
        try:
            # Get model configuration
            if model_name not in LLMModelConfig.MODELS:
                print(f"❌ Model {model_name} not found in configuration")
                return model_results
            
            model_config = LLMModelConfig.MODELS[model_name]
            print(f"📦 {model_config['description']}")
            print(f"   Size: {model_config['size_gb']}GB | Speed: {model_config['speed']} | Quality: {model_config['quality']}")
            
            # Initialize pipeline with specific model
            specific_models = {}
            if model_config["task_type"] == "generation":
                specific_models["generation"] = model_name
            else:
                specific_models["classification"] = model_name
                # Add a fast generation model for the prompt system, avoiding incompatible ones
                fast_generation_models = [name for name, config in LLMModelConfig.MODELS.items() 
                                        if config["task_type"] == "generation" 
                                        and config["speed"] in ["ultra_fast", "very_fast"]
                                        and not config.get("compatibility_issues")]  # Exclude incompatible models
                
                if fast_generation_models:
                    specific_models["generation"] = fast_generation_models[0]
                else:
                    # Fallback to any safe generation model
                    specific_models["generation"] = LLMModelConfig.get_fastest_model_for_task("generation")
            
            pipeline = EnhancedLLMPipeline(
                model_priority="speed",
                enable_quantization=False,  # Disable for comparison consistency
                specific_models=specific_models
            )
            
            # Simple verification - just check if models loaded
            if not pipeline.generation_model and not pipeline.classification_model:
                print(f"❌ Failed to load {model_name}")
                return model_results
            
            # Test each example
            for j, example in enumerate(examples, 1):
                print(f"   📝 Ejemplo {j}/{len(examples)}: {example.id} ({example.category})")
                
                result = self._test_example_with_model(pipeline, model_name, model_config, example)
                model_results.append(result)
                
                # Show quick result
                if result.success:
                    print(f"      ✅ {result.processing_time:.2f}s")
                else:
                    print(f"      ❌ Failed: {result.error}")
            
            # Cleanup
            pipeline.cleanup_memory()
            del pipeline
            
        except Exception as e:
            print(f"❌ Error testing model {model_name}: {e}")
        
        return model_results
    
    def _test_example_with_model(self, pipeline: EnhancedLLMPipeline, 
                               model_name: str, model_config: Dict, 
                               example: TestExample) -> ModelResult:
        """Test a single example with a model."""
        start_time = time.time()
        
        try:
            # Create analysis context
            analysis_context = {
                'far_right_score': example.far_right_score,
                'category': example.category
            }
            
            # Run analysis
            result = pipeline.analyze_content(example.text, analysis_context)
            processing_time = time.time() - start_time
            
            return ModelResult(
                model_name=model_name,
                model_config=model_config,
                example_id=example.id,
                processing_time=processing_time,
                success=True,
                llm_explanation=result.get('llm_explanation', ''),
                llm_categories=result.get('llm_categories', [])
            )
        except Exception as e:
            processing_time = time.time() - start_time
            return ModelResult(
                model_name=model_name,
                model_config=model_config,
                example_id=example.id,
                processing_time=processing_time,
                success=False,
                error=str(e)
            )
    
    def generate_comparison_report(self) -> Dict:
        """Generate focused comparison report with essential information."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Create example text lookup
        example_texts = {ex.id: ex.text for ex in self.test_examples}
        
        # Group results by model
        by_model = {}
        for result in self.results:
            if result.model_name not in by_model:
                by_model[result.model_name] = []
            by_model[result.model_name].append(result)
        
        # Calculate basic statistics for each model
        model_stats = {}
        for model_name, results in by_model.items():
            successful_results = [r for r in results if r.success]
            
            if successful_results:
                avg_time = sum(r.processing_time for r in successful_results) / len(successful_results)
                
                model_stats[model_name] = {
                    "success_rate": len(successful_results) / len(results),
                    "avg_processing_time": round(avg_time, 3),
                    "total_tests": len(results),
                    "failed_tests": len(results) - len(successful_results)
                }
            else:
                model_stats[model_name] = {
                    "success_rate": 0.0,
                    "avg_processing_time": 0.0,
                    "total_tests": len(results),
                    "failed_tests": len(results)
                }
        
        # Create focused detailed results
        focused_results = []
        for result in self.results:
            focused_result = {
                "model_name": result.model_name,
                "example_id": result.example_id,
                "input_text": example_texts.get(result.example_id, "Text not found"),
                "processing_time_seconds": round(result.processing_time, 3),
                "success": result.success
            }
            
            if result.success:
                focused_result.update({
                    "llm_categories": result.llm_categories,
                    "llm_explanation": result.llm_explanation
                })
            else:
                focused_result["error"] = result.error
            
            focused_results.append(focused_result)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary_statistics": model_stats,
            "detailed_analysis": focused_results
        }
    
    def print_summary_report(self):
        """Print a summary of the comparison results."""
        report = self.generate_comparison_report()
        
        print(f"\n📊 RESUMEN DE COMPARACIÓN DE MODELOS")
        print("=" * 60)
        print(f"⏰ Timestamp: {report['timestamp']}")
        print(f"🤖 Modelos probados: {len(report['summary_statistics'])}")
        
        print(f"\n🏆 RANKING POR RENDIMIENTO")
        print("-" * 40)
        
        # Sort models by success rate, then by speed
        sorted_models = sorted(
            report['summary_statistics'].items(),
            key=lambda x: (x[1]['success_rate'], -x[1]['avg_processing_time']),
            reverse=True
        )
        
        for i, (model_name, stats) in enumerate(sorted_models, 1):
            print(f"{i}. {model_name}")
            print(f"   ✅ Éxito: {stats['success_rate']:.1%} ({stats['total_tests'] - stats['failed_tests']}/{stats['total_tests']})")
            print(f"   ⏱️ Tiempo promedio: {stats['avg_processing_time']:.3f}s")
            
            if stats['failed_tests'] > 0:
                print(f"   ❌ Fallos: {stats['failed_tests']}")
            print()
        
        # Speed comparison
        print(f"\n⚡ COMPARACIÓN DE VELOCIDAD")
        print("-" * 30)
        speed_sorted = sorted(
            [(name, stats) for name, stats in report['summary_statistics'].items() if stats['success_rate'] > 0],
            key=lambda x: x[1]['avg_processing_time']
        )
        
        for name, stats in speed_sorted:
            print(f"{name}: {stats['avg_processing_time']:.3f}s (éxito: {stats['success_rate']:.1%})")

def main():
    parser = argparse.ArgumentParser(description="Compare LLM models for Spanish far-right analysis")
    parser.add_argument('--models', nargs='+', help='Specific models to test')
    parser.add_argument('--fast-only', action='store_true', help='Test only fast models (< 1GB)')
    parser.add_argument('--spanish-only', action='store_true', help='Test only Spanish-optimized models')
    parser.add_argument('--generation-only', action='store_true', help='Test only text generation models')
    parser.add_argument('--ollama', action='store_true', help='Test only Ollama models (requires Ollama server)')
    parser.add_argument('--all', action='store_true', help='Test ALL available models (slow!)')
    parser.add_argument('--quick', action='store_true', help='Quick test with 2 fastest models only')
    parser.add_argument('--max-examples', type=int, default=1, help='Maximum number of examples to test (default: 1 for speed)')
    parser.add_argument('--save-results', action='store_true', help='Save detailed results to JSON')
    
    args = parser.parse_args()
    
    # Determine which models to test
    models_to_test = None
    if args.models:
        models_to_test = args.models
    elif args.all:
        models_to_test = list(LLMModelConfig.MODELS.keys())
    elif args.fast_only:
        models_to_test = LLMModelConfig.get_models_by_size(1.0)
    elif args.spanish_only:
        models_to_test = LLMModelConfig.get_spanish_models()
    elif args.generation_only:
        # Get all generation models
        models_to_test = LLMModelConfig.get_models_by_task("generation")
    elif args.ollama:
        # Get only Ollama models
        models_to_test = LLMModelConfig.get_ollama_models()
        if not models_to_test:
            print("❌ No Ollama models found. Make sure Ollama server is running: ollama serve")
            return
    else:
        # Default: test only 1-2 fastest models for quick code validation
        # Get one super-fast model of each type dynamically
        fastest_generation = LLMModelConfig.get_fastest_model_for_task("generation")
        fastest_classification = LLMModelConfig.get_fastest_model_for_task("classification")
        models_to_test = [fastest_generation, fastest_classification]
    
    print(f"🎯 Modelos seleccionados para prueba: {models_to_test}")
    
    # Run comparison
    comparator = ModelComparator()
    comparator.run_comparison(models_to_test, args.max_examples)
    
    # Show summary
    comparator.print_summary_report()
    
    # Save detailed results if requested
    if args.save_results:
        report = comparator.generate_comparison_report()
        output_file = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Resultados detallados guardados en: {output_file}")
    else:
        # Always save a basic comparison report
        report = comparator.generate_comparison_report()
        output_file = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Resultados guardados automáticamente en: {output_file}")

if __name__ == "__main__":
    main()
