"""
Advanced Prompt Optimization System for AI-driven prompt improvement.

This module implements an AdvancedPromptOptimizer class that uses systematic,
batch-based optimization with proper cross-validation and targeted improvements.

⚠️  RESEARCH AND DETECTION PURPOSES ONLY  ⚠️

This system is designed to systematically improve prompts for content analysis
without human intervention, using data-driven refinement loops.
"""

import time
import json
import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .categories import Categories
from .llm_models import EnhancedLLMPipeline, ResponseParser
from .prompts import EnhancedPromptGenerator


@dataclass
class ValidationItem:
    """Represents a single validation data point."""
    content: str
    true_category: str
    source: str = "unknown"
    metadata: Dict[str, Any] = None


@dataclass
class EvaluationMetrics:
    """Metrics from prompt evaluation."""
    accuracy: float
    precision_per_category: Dict[str, float]
    recall_per_category: Dict[str, float]
    f1_per_category: Dict[str, float]
    confusion_matrix: Dict[str, Dict[str, int]]
    error_patterns: List[Dict[str, Any]]
    total_samples: int
    timestamp: datetime


@dataclass
class ErrorAnalysis:
    """Comprehensive analysis of classification errors."""
    common_misclassifications: Dict[str, List[Dict[str, Any]]]
    category_performance: Dict[str, Dict[str, float]]
    problematic_patterns: List[str]
    improvement_suggestions: List[str]


@dataclass
class OptimizationResult:
    """Result of an optimization iteration."""
    iteration: int
    train_metrics: EvaluationMetrics
    val_metrics: EvaluationMetrics
    error_analysis: ErrorAnalysis
    improved_prompts: Dict[str, str]
    improvement_score: float
    converged: bool


class AdvancedPromptOptimizer:
    """
    Advanced prompt optimization system using systematic batch-based approach.

    This class implements a complete optimization loop that:
    1. Uses proper train/validation splits for evaluation
    2. Analyzes error patterns comprehensively across batches
    3. Generates targeted improvements for specific issues
    4. Tests improvements on held-out validation data
    5. Only keeps improvements that generalize well

    The system operates without human intervention, using quantitative metrics
    to guide prompt improvements with proper statistical validation.
    """

    def __init__(self,
                 validation_dataset: List[ValidationItem],
                 current_prompts: Dict[str, str],
                 llm_pipeline: Optional[EnhancedLLMPipeline] = None,
                 max_iterations: int = 5,
                 convergence_threshold: float = 0.005,
                 convergence_window: int = 3,
                 min_improvement_threshold: float = 0.01):
        """
        Initialize the advanced prompt optimizer.

        Args:
            validation_dataset: List of validation items with ground truth
            current_prompts: Current prompt versions to optimize
            llm_pipeline: LLM pipeline for AI-driven improvements (auto-created if None)
            max_iterations: Maximum optimization iterations
            convergence_threshold: Minimum improvement threshold for convergence
            convergence_window: Number of recent iterations to check for convergence
            min_improvement_threshold: Minimum improvement on validation set to keep changes
        """
        self.validation_dataset = validation_dataset
        self.current_prompts = current_prompts.copy()
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.convergence_window = convergence_window
        self.min_improvement_threshold = min_improvement_threshold

        # Initialize LLM pipeline for AI-driven improvements
        self.llm_pipeline = llm_pipeline or EnhancedLLMPipeline()

        # Optimization history
        self.optimization_history: List[OptimizationResult] = []
        self.metrics_history: List[EvaluationMetrics] = []

        # Create prompt generator for baseline comparisons
        self.prompt_generator = EnhancedPromptGenerator()

    async def optimize_prompts(self) -> Dict[str, str]:
        """
        Run systematic batch optimization with proper cross-validation.

        Strategy:
        1. Split data into train/validation sets
        2. Analyze error patterns across training batches
        3. Generate targeted improvements for specific issues
        4. Test improvements on held-out validation set
        5. Only keep improvements that generalize

        Returns:
            Optimized prompts after convergence or max iterations
        """
        print("🚀 Starting ADVANCED systematic prompt optimization...")
        print(f"📊 Validation dataset: {len(self.validation_dataset)} samples")
        print(f"🎯 Max iterations: {self.max_iterations}")
        print(f"📈 Convergence threshold: {self.convergence_threshold}")
        print(f"🎯 Min improvement threshold: {self.min_improvement_threshold}")
        print()

        # Split dataset for proper evaluation
        train_size = int(len(self.validation_dataset) * 0.8)
        train_data = self.validation_dataset[:train_size]
        val_data = self.validation_dataset[train_size:]

        print(f"📊 Train set: {len(train_data)} samples")
        print(f"📊 Validation set: {len(val_data)} samples")
        print()

        for iteration in range(self.max_iterations):
            print(f"🔄 Iteration {iteration + 1}/{self.max_iterations}")
            iteration_start = time.time()

            # Step 1: Evaluate current prompts on training data
            print("📊 Evaluating current prompts on training data...")
            train_metrics = await self.evaluate_on_dataset(train_data)

            # Step 2: Analyze error patterns comprehensively
            print("🔍 Analyzing error patterns comprehensively...")
            error_analysis = self.analyze_error_patterns_comprehensive(train_metrics.error_patterns)

            # Step 3: Generate targeted improvements
            print("🤖 Generating targeted improvements...")
            improved_prompts = await self.generate_targeted_improvements(
                train_metrics, error_analysis
            )

            # Step 4: Test improvements on validation set
            print("🧪 Testing improvements on validation set...")
            val_metrics = await self.evaluate_prompts_on_dataset(improved_prompts, val_data)

            # Step 5: Calculate improvement score (validation accuracy - training accuracy)
            improvement = val_metrics.accuracy - train_metrics.accuracy

            print(f"📊 Train accuracy: {train_metrics.accuracy:.3f}")
            print(f"📊 Validation accuracy: {val_metrics.accuracy:.3f}")
            # Step 6: Decide whether to keep improvements
            if improvement > self.min_improvement_threshold:
                print("✅ Keeping improvements (generalize well)")
                self.current_prompts = improved_prompts.copy()
            else:
                print("⚠️ Discarding improvements (don't generalize)")
                # Keep current prompts

            # Record this iteration
            result = OptimizationResult(
                iteration=iteration + 1,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                error_analysis=error_analysis,
                improved_prompts=self.current_prompts.copy(),
                improvement_score=improvement,
                converged=False
            )
            self.optimization_history.append(result)
            self.metrics_history.append(val_metrics)

            # Check for convergence
            if self.has_converged():
                print("🎯 Convergence achieved!")
                result.converged = True
                break

            iteration_time = time.time() - iteration_start
            print(f"⏱️  Iteration {iteration + 1} completed in {iteration_time:.2f}s")
            print()

        print("🏁 Optimization complete!")
        final_metrics = self.metrics_history[-1] if self.metrics_history else None
        if final_metrics:
            print(f"📊 Final accuracy: {final_metrics.accuracy:.3f}")
        return self.current_prompts

    async def evaluate_on_dataset(self, dataset: List[ValidationItem]) -> EvaluationMetrics:
        """
        Evaluate prompts on a specific dataset.

        Args:
            dataset: Dataset to evaluate on

        Returns:
            Comprehensive evaluation metrics
        """
        total_samples = len(dataset)
        correct_predictions = 0
        predictions = []
        actuals = []

        # Confusion matrix: predicted -> actual -> count
        confusion_matrix = {cat: {cat2: 0 for cat2 in Categories.get_all_categories()}
                          for cat in Categories.get_all_categories()}

        error_patterns = []

        for i, item in enumerate(dataset, 1):
            print(f"  📝 Processing item {i}/{total_samples}...")

            # Get prediction using current prompts
            predicted_category = await self.classify_with_current_prompts(item.content)

            predictions.append(predicted_category)
            actuals.append(item.true_category)

            # Track accuracy
            if predicted_category == item.true_category:
                correct_predictions += 1
            else:
                # Record error pattern
                error_patterns.append({
                    'content': item.content,
                    'predicted': predicted_category,
                    'actual': item.true_category,
                    'source': item.source,
                    'metadata': item.metadata
                })

            # Update confusion matrix
            confusion_matrix[predicted_category][item.true_category] += 1

        # Calculate per-category metrics
        precision_per_category = {}
        recall_per_category = {}
        f1_per_category = {}

        all_categories = Categories.get_all_categories()
        for category in all_categories:
            # Precision: TP / (TP + FP)
            tp = confusion_matrix[category][category]
            fp = sum(confusion_matrix[category][other] for other in all_categories if other != category)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0

            # Recall: TP / (TP + FN)
            fn = sum(confusion_matrix[other][category] for other in all_categories if other != category)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            # F1 Score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            precision_per_category[category] = precision
            recall_per_category[category] = recall
            f1_per_category[category] = f1

        return EvaluationMetrics(
            accuracy=correct_predictions / total_samples,
            precision_per_category=precision_per_category,
            recall_per_category=recall_per_category,
            f1_per_category=f1_per_category,
            confusion_matrix=confusion_matrix,
            error_patterns=error_patterns,
            total_samples=total_samples,
            timestamp=datetime.now()
        )

    async def evaluate_prompts_on_dataset(self, prompts: Dict[str, str], dataset: List[ValidationItem]) -> EvaluationMetrics:
        """
        Evaluate specific prompts on a dataset.

        Args:
            prompts: Prompts to evaluate
            dataset: Dataset to evaluate on

        Returns:
            Evaluation metrics for these prompts
        """
        # Temporarily swap prompts
        original_prompts = self.current_prompts.copy()
        self.current_prompts = prompts.copy()

        try:
            return await self.evaluate_on_dataset(dataset)
        finally:
            # Always restore original prompts
            self.current_prompts = original_prompts

    def analyze_error_patterns_comprehensive(self, error_patterns: List[Dict[str, Any]]) -> ErrorAnalysis:
        """
        Perform comprehensive analysis of error patterns.

        Args:
            error_patterns: List of classification errors

        Returns:
            Detailed error analysis
        """
        if not error_patterns:
            return ErrorAnalysis(
                common_misclassifications={},
                category_performance={},
                problematic_patterns=[],
                improvement_suggestions=[]
            )

        # Group errors by misclassification type
        common_misclassifications = {}
        for error in error_patterns:
            error_type = f"{error['predicted']} → {error['actual']}"
            if error_type not in common_misclassifications:
                common_misclassifications[error_type] = []
            common_misclassifications[error_type].append(error)

        # Analyze category performance
        category_performance = {}
        all_categories = Categories.get_all_categories()

        for category in all_categories:
            # Errors where this category was predicted but wrong
            false_positives = [e for e in error_patterns if e['predicted'] == category]
            # Errors where this category should have been predicted but wasn't
            false_negatives = [e for e in error_patterns if e['actual'] == category]

            category_performance[category] = {
                'false_positives': len(false_positives),
                'false_negatives': len(false_negatives),
                'total_errors': len(false_positives) + len(false_negatives),
                'fp_examples': [e['content'][:100] for e in false_positives[:3]],
                'fn_examples': [e['content'][:100] for e in false_negatives[:3]]
            }

        # Identify problematic patterns
        problematic_patterns = []

        # Find categories with high error rates
        high_error_categories = [
            cat for cat, perf in category_performance.items()
            if perf['total_errors'] > len(error_patterns) * 0.1  # >10% of all errors
        ]
        if high_error_categories:
            problematic_patterns.append(f"High error categories: {', '.join(high_error_categories)}")

        # Find common misclassification pairs
        frequent_errors = sorted(common_misclassifications.items(), key=lambda x: len(x[1]), reverse=True)
        if frequent_errors:
            top_error = frequent_errors[0]
            problematic_patterns.append(f"Most common error: {top_error[0]} ({len(top_error[1])} cases)")

        # Generate improvement suggestions
        improvement_suggestions = []

        for cat, perf in category_performance.items():
            if perf['false_positives'] > 0:
                improvement_suggestions.append(
                    f"Reduce false positives for {cat}: {len(perf['fp_examples'])} examples found"
                )
            if perf['false_negatives'] > 0:
                improvement_suggestions.append(
                    f"Improve detection of {cat}: {len(perf['fn_examples'])} missed cases"
                )

        # Add specific suggestions based on error patterns
        if len(frequent_errors) > 0:
            top_error_pair = frequent_errors[0][0]
            pred_cat, true_cat = top_error_pair.split(' → ')
            improvement_suggestions.append(
                f"Improve distinction between {pred_cat} and {true_cat} (most common confusion)"
            )

        return ErrorAnalysis(
            common_misclassifications=common_misclassifications,
            category_performance=category_performance,
            problematic_patterns=problematic_patterns,
            improvement_suggestions=improvement_suggestions
        )

    async def generate_targeted_improvements(
        self,
        metrics: EvaluationMetrics,
        error_analysis: ErrorAnalysis
    ) -> Dict[str, str]:
        """
        Generate targeted improvements based on comprehensive error analysis.

        Args:
            metrics: Current evaluation metrics
            error_analysis: Detailed error analysis

        Returns:
            Dictionary of improved prompts
        """
        # Create a sophisticated improvement prompt
        improvement_prompt = self._build_improvement_prompt(metrics, error_analysis)

        try:
            # Use LLM to generate improvements
            improved_prompts_json = await self._generate_with_llm(improvement_prompt)

            if improved_prompts_json:
                improved_prompts = json.loads(improved_prompts_json)
                if isinstance(improved_prompts, dict) and 'categorization' in improved_prompts:
                    print("✅ Successfully generated improved prompts")
                    return improved_prompts
                else:
                    print("⚠️ LLM returned invalid structure")
            else:
                print("⚠️ LLM returned empty response")

        except Exception as e:
            print(f"⚠️ Error generating improvements: {e}")

        # Return current prompts if improvement generation fails
        return self.current_prompts.copy()

    def _build_improvement_prompt(self, metrics: EvaluationMetrics, error_analysis: ErrorAnalysis) -> str:
        """Build a sophisticated prompt for LLM-based improvements."""

        # Format error analysis for LLM
        error_summary_parts = []
        error_summary_parts.append("=== PATRONES DE ERROR ===")

        for error_type, errors in list(error_analysis.common_misclassifications.items())[:5]:  # Top 5
            error_summary_parts.append(f"• {error_type}: {len(errors)} casos")
            for error in errors[:2]:  # 2 examples per type
                error_summary_parts.append(f"  - Contenido: '{error['content'][:80]}...'")

        error_summary_parts.append("\n=== ANÁLISIS POR CATEGORÍA ===")
        for cat, perf in error_analysis.category_performance.items():
            if perf['total_errors'] > 0:
                error_summary_parts.append(f"• {cat}: {perf['total_errors']} errores totales")
                if perf['false_positives'] > 0:
                    error_summary_parts.append(f"  - Falsos positivos: {perf['false_positives']}")
                if perf['false_negatives'] > 0:
                    error_summary_parts.append(f"  - Falsos negativos: {perf['false_negatives']}")

        error_summary_parts.append("\n=== SUGERENCIAS DE MEJORA ===")
        for suggestion in error_analysis.improvement_suggestions[:5]:  # Top 5
            error_summary_parts.append(f"• {suggestion}")

        error_summary = "\n".join(error_summary_parts)

        # Build the complete improvement prompt
        prompt = f"""
Eres un experto en optimización de prompts para sistemas de clasificación de contenido problemático en español.

DATOS DE EVALUACIÓN ACTUAL:
- Precisión global: {metrics.accuracy:.3f}
- Total errores: {len(metrics.error_patterns)}
- Total muestras: {metrics.total_samples}

{error_summary}

PROMPTS ACTUALES:
{json.dumps(self.current_prompts, indent=2, ensure_ascii=False)}

INSTRUCCIONES PARA OPTIMIZACIÓN:

1. **ANÁLISIS DE ERRORES**: Identifica los patrones específicos que causan confusiones entre categorías.

2. **MEJORAS DIRIGIDAS**:
   - Para categorías con muchos falsos positivos: Añade criterios más específicos para evitar clasificaciones erróneas
   - Para categorías con muchos falsos negativos: Mejora la detección de patrones característicos
   - Para confusiones comunes: Añade ejemplos específicos y reglas de distinción

3. **REFINAMIENTO DE PROMPTS**:
   - Mejora las reglas de clasificación basándote en los errores observados
   - Añade ejemplos específicos de casos confusos
   - Clarifica criterios de distinción entre categorías problemáticas
   - Asegura que las explicaciones sean específicas y útiles

4. **CONSISTENCIA**: Mantén el formato y estructura de los prompts originales.

FORMATO DE RESPUESTA REQUERIDO:
Responde ÚNICAMENTE con un objeto JSON válido que tenga exactamente estas claves:
- "categorization": prompt de categorización mejorado
- "local_explanation": prompt de explicación local mejorado
- "external_explanation": prompt de explicación externa mejorado
- "gemini_analysis": prompt de análisis Gemini mejorado

NO incluyas texto adicional, explicaciones, o formato markdown. Solo el JSON puro.
"""

        return prompt

    async def classify_with_current_prompts(self, content: str) -> str:
        """
        Classify content using current prompts and actual LLM pipeline.

        Args:
            content: Text content to classify

        Returns:
            Predicted category
        """
        try:
            result = "general"  # Default fallback

            # Prioritize Ollama classification if available
            if self.llm_pipeline.ollama_client:
                ollama_result = self._classify_with_ollama_custom_prompt(content)
                ollama_categories = ollama_result.get("llm_categories", [])
                if ollama_categories:
                    result = ollama_categories[0]

            # Use generation model with classification-specific prompts
            elif self.llm_pipeline.generation_model:
                result = self._classify_with_generation_model_custom_prompt(content)

            return result

        except Exception as e:
            print(f"⚠️ LLM classification error: {e}")
            return "general"

    def _classify_with_ollama_custom_prompt(self, text: str) -> Dict:
        """Perform classification using Ollama model with CUSTOM prompts from optimizer."""
        try:
            # Use the CUSTOM categorization prompt from current_prompts
            custom_prompt = self.current_prompts.get('categorization',
                                                   self.prompt_generator.build_categorization_prompt(text))

            # Add the content to classify
            full_prompt = f"{custom_prompt}\n\nCONTENIDO A ANALIZAR:\n{text}\n\nFORMATO OBLIGATORIO:\nCATEGORÍA: [nombre_categoría]\nEXPLICACIÓN: [2-3 frases explicando por qué pertenece a esa categoría]"

            response = self.llm_pipeline.ollama_client.chat.completions.create(
                model=self.llm_pipeline.ollama_model_name,
                messages=[
                    {"role": "system", "content": "Eres un experto analista de contenido especializado en clasificar contenido problemático."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.1
            )

            result = response.choices[0].message.content.strip().lower()

            # Extract category from response
            valid_categories = Categories.get_all_categories()
            detected_category = Categories.GENERAL

            for category in valid_categories:
                if category in result:
                    detected_category = category
                    break

            return {"llm_categories": [detected_category]}

        except Exception as e:
            return {"llm_categories": []}

    def _classify_with_generation_model_custom_prompt(self, text: str) -> str:
        """Perform classification using generation model with CUSTOM prompts from optimizer."""
        try:
            # Use the CUSTOM categorization prompt from current_prompts
            custom_prompt = self.current_prompts.get('categorization',
                                                   self.prompt_generator.build_categorization_prompt(text))

            full_prompt = f"{custom_prompt}\n\nCONTENIDO A ANALIZAR:\n{text}\n\nFORMATO OBLIGATORIO:\nCATEGORÍA: [nombre_categoría]\nEXPLICACIÓN: [2-3 frases explicando por qué pertenece a esa categoría]"

            # Get generation parameters
            gen_config = self.llm_pipeline.model_info.get("generation", {})
            generation_params = gen_config.get("generation_params", {}).copy()
            generation_params.update({"temperature": 0.1, "max_new_tokens": 100})

            # Generate response
            if self.llm_pipeline.generation_model == "ollama":
                response = self.llm_pipeline.ollama_client.chat.completions.create(
                    model=self.llm_pipeline.ollama_model_name,
                    messages=[
                        {"role": "system", "content": "Eres un experto analista de contenido especializado en clasificar contenido problemático."},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=100
                )
                result = response.choices[0].message.content.strip().lower()
            else:
                response = self.llm_pipeline.generation_model(full_prompt, **generation_params)
                parser_type = gen_config.get("response_parser", "text_generation")
                result = ResponseParser.parse_response(response, parser_type, full_prompt, gen_config)
                result = result.strip().lower() if result else ""

            # Extract category
            valid_categories = Categories.get_all_categories()
            for category in valid_categories:
                if category in result:
                    return category

            return Categories.GENERAL

        except Exception as e:
            return Categories.GENERAL

    async def _generate_with_llm(self, prompt: str) -> str:
        """Generate response using LLM pipeline for JSON generation."""
        try:
            if self.llm_pipeline.ollama_client:
                response = self.llm_pipeline.ollama_client.chat.completions.create(
                    model=self.llm_pipeline.ollama_model_name,
                    messages=[
                        {"role": "system", "content": "Eres un experto en optimización de prompts. Siempre respondes ÚNICAMENTE con JSON válido. No incluyas texto adicional, explicaciones, o formato markdown. Solo JSON puro."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                return response.choices[0].message.content.strip()

            elif self.llm_pipeline.generation_model and self.llm_pipeline.generation_model != "ollama":
                json_prompt = f"{prompt}\n\nINSTRUCCIONES CRÍTICAS:\n- Responde ÚNICAMENTE con JSON válido\n- No incluyas texto adicional\n- El JSON debe tener exactamente las claves requeridas"

                gen_config = self.llm_pipeline.model_info.get("generation", {})
                generation_params = gen_config.get("generation_params", {}).copy()
                generation_params.update({"temperature": 0.2, "max_new_tokens": 2000})

                response = self.llm_pipeline.generation_model(json_prompt, **generation_params)
                parser_type = gen_config.get("response_parser", "text_generation")
                result = ResponseParser.parse_response(response, parser_type, json_prompt, gen_config)
                return result.strip() if result else ""

            else:
                return ""

        except Exception as e:
            print(f"⚠️ LLM generation error: {e}")
            return ""

    def has_converged(self) -> bool:
        """Check if optimization has converged."""
        if len(self.metrics_history) < self.convergence_window:
            return False

        recent_accuracies = [m.accuracy for m in self.metrics_history[-self.convergence_window:]]
        improvement = max(recent_accuracies) - min(recent_accuracies)
        return improvement < self.convergence_threshold

    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        if not self.optimization_history:
            return {"error": "No optimization history available"}

        final_metrics = self.metrics_history[-1] if self.metrics_history else None

        return {
            "total_iterations": len(self.optimization_history),
            "converged": self.has_converged(),
            "final_accuracy": final_metrics.accuracy if final_metrics else 0,
            "final_prompts": self.current_prompts,
            "optimization_history": [
                {
                    "iteration": result.iteration,
                    "train_accuracy": result.train_metrics.accuracy,
                    "val_accuracy": result.val_metrics.accuracy,
                    "improvement_score": result.improvement_score,
                    "converged": result.converged
                }
                for result in self.optimization_history
            ]
        }

    def save_optimization_results(self, filepath: str):
        """Save optimization results to file."""
        results = self.get_optimization_report()
        results["timestamp"] = datetime.now().isoformat()
        results["validation_dataset_size"] = len(self.validation_dataset)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"💾 Advanced optimization results saved to {filepath}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_validation_dataset_from_posts(posts: List[Dict[str, Any]]) -> List[ValidationItem]:
    """Create validation dataset from existing analyzed posts."""
    validation_items = []

    for post in posts:
        if 'content' in post and 'category' in post:
            item = ValidationItem(
                content=post['content'],
                true_category=post['category'],
                source=post.get('username', 'unknown'),
                metadata={
                    'tweet_id': post.get('tweet_id'),
                    'timestamp': post.get('timestamp')
                }
            )
            validation_items.append(item)

    return validation_items


def load_validation_dataset(filepath: str) -> List[ValidationItem]:
    """Load validation dataset from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    validation_items = []
    for item_data in data:
        item = ValidationItem(
            content=item_data['content'],
            true_category=item_data['true_category'],
            source=item_data.get('source', 'unknown'),
            metadata=item_data.get('metadata', {})
        )
        validation_items.append(item)

    return validation_items