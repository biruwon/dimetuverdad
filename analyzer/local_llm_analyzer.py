"""
Local LLM Analyzer using gpt-oss:20b for both category detection and explanation generation.
This replaces the model priority system with a single, consistent local model.
"""

import json
from typing import Tuple, Optional
from openai import OpenAI

from .categories import Categories
from .prompts import EnhancedPromptGenerator


class LocalLLMAnalyzer:
    """
    Unified local LLM analysis using gpt-oss:20b for:
    - Category detection (when patterns fail)
    - Explanation generation (always)
    """
    
    def __init__(self, model: str = "gpt-oss:20b", verbose: bool = False):
        """
        Initialize local LLM analyzer with Ollama.
        
        Args:
            model: Ollama model to use (default: gpt-oss:20b)
            verbose: Enable detailed logging
        """
        self.model = model
        self.verbose = verbose
        self.ollama_client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"  # Required but unused for Ollama
        )
        self.prompt_generator = EnhancedPromptGenerator()
        
        if self.verbose:
            print(f"🤖 LocalLLMAnalyzer initialized with model: {self.model}")
    
    async def categorize_and_explain(self, content: str) -> Tuple[str, str]:
        """
        Single LLM call for both category detection and explanation.
        Used when pattern detection fails to find a category.
        
        Args:
            content: Content to analyze
        
        Returns:
            Tuple of (category, explanation)
        """
        if self.verbose:
            print(f"🔍 Running local LLM categorization + explanation")
            print(f"📝 Content: {content[:100]}...")
        
        # Build unified prompt for category + explanation
        prompt = self._build_categorization_prompt(content)
        
        try:
            # Single LLM call for both tasks
            response = self._generate_with_ollama(prompt)
            
            # Parse structured response
            category, explanation = self._parse_category_and_explanation(response)
            
            if self.verbose:
                print(f"✅ Category detected: {category}")
                print(f"💭 Explanation: {explanation[:100]}...")
            
            return category, explanation
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error in categorize_and_explain: {e}")
            # Fallback to general category with error explanation
            return Categories.GENERAL, f"Error en análisis local: {str(e)}"
    
    async def explain_only(self, content: str, category: str) -> str:
        """
        Generate explanation for known category (from pattern detection).
        
        Args:
            content: Content to explain
            category: Already-detected category
        
        Returns:
            Explanation (Spanish, 2-3 sentences)
        """
        if self.verbose:
            print(f"🔍 Generating local explanation for category: {category}")
            print(f"📝 Content: {content[:100]}...")
        
        # Build category-specific explanation prompt
        prompt = self._build_explanation_prompt(content, category)
        
        try:
            explanation = self._generate_with_ollama(prompt)
            
            if self.verbose:
                print(f"💭 Explanation generated: {explanation[:100]}...")
            
            return explanation
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error in explain_only: {e}")
            return f"Error generando explicación local: {str(e)}"
    
    def _build_categorization_prompt(self, content: str) -> str:
        """
        Build prompt for combined category detection + explanation.
        
        Returns structured output format:
        CATEGORÍA: [category_name]
        EXPLICACIÓN: [2-3 sentences in Spanish]
        """
        all_categories = Categories.get_all_categories()
        category_descriptions = {
            Categories.HATE_SPEECH: "Discurso de odio, xenofobia, amenazas violentas",
            Categories.DISINFORMATION: "Información falsa, datos fabricados, afirmaciones sin evidencia",
            Categories.CONSPIRACY_THEORY: "Teorías conspirativas, narrativas de agenda oculta",
            Categories.FAR_RIGHT_BIAS: "Retórica extremista, nacionalismo radical, anti-establishment",
            Categories.CALL_TO_ACTION: "Llamadas a movilización, organización de protestas",
            Categories.NATIONALISM: "Énfasis en identidad nacional, retórica patriótica",
            Categories.ANTI_GOVERNMENT: "Crítica institucional, desconfianza en el gobierno",
            Categories.HISTORICAL_REVISIONISM: "Reinterpretación histórica, minimización de atrocidades",
            Categories.POLITICAL_GENERAL: "Discurso político neutral o general",
            Categories.GENERAL: "Contenido neutral sin patrones problemáticos"
        }
        
        category_list = "\n".join([
            f"  - {cat}: {category_descriptions.get(cat, '')}"
            for cat in all_categories
        ])
        
        prompt = f"""Eres un experto en análisis de contenido político español, especializado en detectar discurso extremista y desinformación.

Analiza el siguiente contenido y:
1. Identifica la categoría más apropiada
2. Proporciona una explicación clara en español (2-3 frases)

CATEGORÍAS DISPONIBLES:
{category_list}

CONTENIDO A ANALIZAR:
{content}

IMPORTANTE:
- Prioriza hate_speech si hay xenofobia o amenazas
- Usa disinformation para afirmaciones falsas verificables
- Usa conspiracy_theory para narrativas de conspiración
- Usa general si no hay contenido problemático

FORMATO DE RESPUESTA (obligatorio):
CATEGORÍA: [nombre_categoría]
EXPLICACIÓN: [2-3 frases explicando por qué pertenece a esa categoría]

Responde ahora:"""
        
        return prompt
    
    def _build_explanation_prompt(self, content: str, category: str) -> str:
        """
        Build prompt for explanation-only generation (category already known).
        """
        # Use existing prompt generator for category-specific prompts
        return self.prompt_generator.generate_explanation_prompt(
            content,
            category,
            model_type="ollama"
        )
    
    def _generate_with_ollama(self, prompt: str) -> str:
        """
        Generate response using Ollama API.
        
        Args:
            prompt: Prompt to send to the model
        
        Returns:
            Generated text response
        """
        try:
            response = self.ollama_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Eres un experto en análisis de contenido político español."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent categorization
                max_tokens=512
            )
            
            generated_text = response.choices[0].message.content.strip()
            return generated_text
            
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")
    
    def _parse_category_and_explanation(self, response: str) -> Tuple[str, str]:
        """
        Parse structured response from categorization prompt.
        
        Expected format:
        CATEGORÍA: category_name
        EXPLICACIÓN: explanation text
        
        Returns:
            Tuple of (category, explanation)
        """
        lines = response.strip().split('\n')
        
        category = Categories.GENERAL  # Default fallback
        explanation = response  # Fallback to full response
        
        # Parse structured response
        for line in lines:
            line = line.strip()
            
            # Extract category
            if line.upper().startswith("CATEGORÍA:"):
                category_text = line.split(":", 1)[1].strip().lower()
                
                # Validate against known categories
                all_categories_lower = [cat.lower() for cat in Categories.get_all_categories()]
                if category_text in all_categories_lower:
                    # Find exact case match
                    for cat in Categories.get_all_categories():
                        if cat.lower() == category_text:
                            category = cat
                            break
            
            # Extract explanation
            elif line.upper().startswith("EXPLICACIÓN:"):
                explanation = line.split(":", 1)[1].strip()
        
        # Validate explanation is not empty
        if not explanation or len(explanation.strip()) < 10:
            explanation = f"Contenido clasificado como {category}."
        
        return category, explanation
