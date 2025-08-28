"""
LLM Models and Pipeline for Spanish Far-Right Analysis
Extracted for reusability and model comparison.
"""

import json
import os
import re
import time
import warnings
from typing import Dict, List, Optional
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    pipeline,
    BitsAndBytesConfig
)

from enhanced_prompts import EnhancedPromptGenerator, AnalysisType, create_context_from_analysis

# Suppress warnings including the parameter conflict warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

class LLMModelConfig:
    """Configuration for different LLM models optimized for Spanish far-right analysis."""
    
    MODELS = {
        # === ULTRA-FAST MODELS (< 0.3GB) ===
        "tiny-bert": {
            "model_name": "huawei-noah/TinyBERT_General_4L_312D",
            "description": "Ultra-compact TinyBERT for classification",
            "size_gb": 0.1,
            "speed": "ultra_fast",
            "quality": "basic",
            "free": True,
            "task_type": "classification",
            "primary_task": "classification",  # Use classification pipeline directly
            "pipeline_type": "text-classification",
            "generation_params": {},  # No generation params for pure classification
            "max_input_length": 512,
            "language": "multilingual",
            "requires_tokenizer_config": False,
            "response_parser": "classification",
            "prompt_removal_strategy": None,
            "model_type": "text",
            "fallback_params": {}  # Classification models don't need fallback generation params
        },
        
        # === FAST MODELS (< 1GB) ===
        "distilgpt2": {
            "model_name": "distilgpt2",
            "description": "Very fast distilled GPT-2 model",
            "size_gb": 0.3,
            "speed": "very_fast",
            "quality": "basic",
            "free": True,
            "task_type": "generation",
            "primary_task": "generation",  # Use generation pipeline
            "pipeline_type": "text-generation",
            "generation_params": {
                "max_new_tokens": 150,
                "do_sample": True,
                "temperature": 0.3,
                "return_full_text": False,
                "truncation": True
            },
            "max_input_length": 800,
            "language": "english",
            "requires_tokenizer_config": True,
            "response_parser": "text_generation",
            "prompt_removal_strategy": "remove_prompt",
            "model_type": "text",
            "fallback_params": {
                "max_new_tokens": 50,
                "do_sample": False,
                "truncation": True,
                "pad_token_id": 50256
            }
        },
        
        "distilbert-multilingual": {
            "model_name": "distilbert-base-multilingual-cased",
            "description": "Fast multilingual BERT",
            "size_gb": 0.5,
            "speed": "very_fast", 
            "quality": "good",
            "free": True,
            "task_type": "classification",
            "primary_task": "mixed",  # Can do both classification and generation
            "pipeline_type": "text-classification",
            "generation_params": {
                "max_new_tokens": 120,
                "do_sample": True,
                "temperature": 0.4,
                "return_full_text": False
            },
            "max_input_length": 512,
            "language": "multilingual",
            "requires_tokenizer_config": False,
            "response_parser": "classification",
            "prompt_removal_strategy": None,
            "model_type": "text"
        },
        
        "gpt2-spanish": {
            "model_name": "DeepESP/gpt2-spanish",
            "description": "Fast Spanish GPT-2 for generation",
            "size_gb": 0.5,
            "speed": "fast",
            "quality": "good",
            "free": True,
            "task_type": "generation",
            "primary_task": "generation",  # Use generation pipeline
            "pipeline_type": "text-generation",
            "generation_params": {
                "max_new_tokens": 150,
                "do_sample": True,
                "temperature": 0.3,
                "return_full_text": False,
                "truncation": True
            },
            "max_input_length": 800,
            "language": "spanish",
            "requires_tokenizer_config": True,
            "response_parser": "text_generation",
            "prompt_removal_strategy": "remove_prompt",
            "model_type": "text"
        },
        
        # === SPANISH MODELS ===
        "beto-spanish": {
            "model_name": "dccuchile/bert-base-spanish-wwm-cased",
            "description": "Spanish BERT for classification tasks",
            "size_gb": 0.4,
            "speed": "very_fast",
            "quality": "good",
            "free": True,
            "task_type": "classification",
            "primary_task": "classification",  # Use classification pipeline directly
            "pipeline_type": "text-classification",
            "generation_params": {},
            "max_input_length": 512,
            "language": "spanish",
            "requires_tokenizer_config": False,
            "response_parser": "classification",
            "prompt_removal_strategy": None,
            "model_type": "text"
        },
        
        "roberta-spanish": {
            "model_name": "BSC-TeMU/roberta-base-bne",
            "description": "Spanish RoBERTa by Barcelona Supercomputing Center",
            "size_gb": 0.5,
            "speed": "fast",
            "quality": "very_good",
            "free": True,
            "task_type": "classification", 
            "primary_task": "classification",  # Use classification pipeline directly
            "pipeline_type": "text-classification",
            "generation_params": {},
            "max_input_length": 512,
            "language": "spanish",
            "requires_tokenizer_config": False,
            "response_parser": "classification",
            "prompt_removal_strategy": None,
            "model_type": "text"
        },
        
        # === HATE SPEECH SPECIALISTS ===
        "roberta-hate": {
            "model_name": "martin-ha/toxic-comment-model",
            "description": "Specialized hate speech detection",
            "size_gb": 0.5,
            "speed": "very_fast",
            "quality": "good",
            "free": True,
            "task_type": "classification",
            "primary_task": "classification",  # Use classification pipeline directly
            "pipeline_type": "text-classification",
            "generation_params": {},
            "max_input_length": 512,
            "language": "english",
            "requires_tokenizer_config": False,
            "response_parser": "classification",
            "prompt_removal_strategy": None,
            "model_type": "text"
        },
        
        "hate-speech-spanish": {
            "model_name": "finiteautomata/beto-sentiment-analysis",
            "description": "Spanish sentiment analysis model",
            "size_gb": 0.4,
            "speed": "fast",
            "quality": "very_good",
            "free": True,
            "task_type": "classification",
            "primary_task": "classification",  # Use classification pipeline directly
            "pipeline_type": "text-classification",
            "generation_params": {},
            "max_input_length": 512,
            "language": "spanish",
            "requires_tokenizer_config": False,
            "response_parser": "classification",
            "prompt_removal_strategy": None,
            "model_type": "text"
        },
        
        # === GEMMA MODELS (Current Generation) ===
        "gemma-2b": {
            "model_name": "google/gemma-2b-it",
            "description": "Fast Gemma 2B model, optimized for instruction following",
            "size_gb": 4.5,
            "speed": "fast",
            "quality": "good",
            "free": True,
            "task_type": "generation",
            "primary_task": "generation",  # Use generation pipeline
            "pipeline_type": "text-generation",
            "generation_params": {
                "max_new_tokens": 200,
                "do_sample": True,
                "temperature": 0.3,
                "return_full_text": False,
                "truncation": True
            },
            "max_input_length": 1000,
            "language": "multilingual",
            "requires_tokenizer_config": True,
            "response_parser": "text_generation",
            "prompt_removal_strategy": "remove_prompt",
            "model_type": "text",
            "fallback_params": {
                "max_new_tokens": 50,
                "do_sample": False,
                "truncation": True,
                "pad_token_id": 50256
            }
        },
        
        "gemma-7b": {
            "model_name": "google/gemma-7b-it", 
            "description": "Higher quality Gemma 7B model",
            "size_gb": 15,
            "speed": "medium",
            "quality": "excellent", 
            "free": True,
            "task_type": "generation",
            "primary_task": "generation",  # Use generation pipeline
            "pipeline_type": "text-generation",
            "generation_params": {
                "max_new_tokens": 300,
                "do_sample": True,
                "temperature": 0.3,
                "return_full_text": False,
                "truncation": True
            },
            "max_input_length": 1200,
            "language": "multilingual",
            "requires_tokenizer_config": True,
            "response_parser": "text_generation",
            "prompt_removal_strategy": "remove_prompt",
            "model_type": "text",
            "fallback_params": {
                "max_new_tokens": 50,
                "do_sample": False,
                "truncation": True,
                "pad_token_id": 50256
            }
        },

        # === GEMMA 3 MODELS (Latest Generation 2025) ===
        "gemma-3-270m": {
            "model_name": "google/gemma-3-270m",
            "description": "Ultra-compact Gemma 3 270M - hyper-efficient AI with 32K context",
            "size_gb": 0.3,
            "speed": "ultra_fast",
            "quality": "good",
            "free": True,
            "task_type": "generation",
            "primary_task": "generation",
            "pipeline_type": "text-generation",
            "generation_params": {
                "max_new_tokens": 200,
                "do_sample": False,  # Use greedy decoding to avoid sampling issues
                "temperature": 1.0,  # Keep at 1.0 when do_sample=False
                "return_full_text": False,
                "truncation": True,
                "pad_token_id": 2  # Gemma uses pad_token_id=2
            },
            "max_input_length": 2000,  # 32K context window
            "language": "multilingual",
            "requires_tokenizer_config": True,
            "response_parser": "text_generation",
            "prompt_removal_strategy": "remove_prompt",
            "model_type": "text",
            "fallback_params": {
                "max_new_tokens": 50,
                "do_sample": False,
                "truncation": True,
                "pad_token_id": 2
            }
        },
        
        "gemma-3-270m-it": {
            "model_name": "google/gemma-3-270m-it",
            "description": "Instruction-tuned Gemma 3 270M for chat and instruction following",
            "size_gb": 0.3,
            "speed": "ultra_fast",
            "quality": "very_good",
            "free": True,
            "task_type": "generation",
            "primary_task": "generation",
            "pipeline_type": "text-generation",
            "generation_params": {
                "max_new_tokens": 250,
                "do_sample": False,  # Use greedy decoding to avoid sampling issues
                "temperature": 1.0,  # Keep at 1.0 when do_sample=False
                "return_full_text": False,
                "truncation": True,
                "pad_token_id": 2  # Gemma uses pad_token_id=2
            },
            "max_input_length": 2000,  # 32K context window
            "language": "multilingual",
            "requires_tokenizer_config": True,
            "response_parser": "text_generation",
            "prompt_removal_strategy": "remove_prompt",
            "model_type": "text",
            "fallback_params": {
                "max_new_tokens": 50,
                "do_sample": False,
                "truncation": True,
                "pad_token_id": 2
            }
        },

        "gemma-3-4b-it": {
            "model_name": "google/gemma-3-4b-it",
            "description": "Multimodal Gemma 3 4B with image understanding and 128K context",
            "size_gb": 8.0,
            "speed": "fast",
            "quality": "excellent",
            "free": True,
            "task_type": "generation",
            "primary_task": "generation",
            "pipeline_type": "text-generation",
            "generation_params": {
                "max_new_tokens": 400,
                "do_sample": False,  # Use greedy decoding to avoid sampling issues
                "temperature": 1.0,  # Keep at 1.0 when do_sample=False
                "return_full_text": False,
                "truncation": True,
                "pad_token_id": 2  # Gemma uses pad_token_id=2
            },
            "max_input_length": 4000,  # 128K context window
            "language": "multilingual",
            "requires_tokenizer_config": True,
            "response_parser": "text_generation",
            "prompt_removal_strategy": "remove_prompt",
            "model_type": "multimodal",  # Supports images
            "fallback_params": {
                "max_new_tokens": 100,
                "do_sample": False,
                "truncation": True,
                "pad_token_id": 2
            }
        },

        # === GROK MODELS (xAI) ===
        "grok-2": {
            "model_name": "xai-org/grok-2",
            "description": "Grok 2 - large open-source model from xAI with advanced reasoning",
            "size_gb": 500,  # Very large model
            "speed": "very_slow",
            "quality": "outstanding",
            "free": True,
            "task_type": "generation",
            "primary_task": "generation",
            "pipeline_type": "text-generation",
            "generation_params": {
                "max_new_tokens": 500,
                "do_sample": True,
                "temperature": 0.2,
                "return_full_text": False,
                "truncation": True
            },
            "max_input_length": 8000,
            "language": "multilingual",
            "requires_tokenizer_config": True,
            "response_parser": "text_generation",
            "prompt_removal_strategy": "remove_prompt",
            "model_type": "text",
            "fallback_params": {
                "max_new_tokens": 100,
                "do_sample": False,
                "truncation": True
            },
            "special_requirements": "Requires 8 GPUs with >40GB memory each, uses fp8 quantization"
        },
        
        # === LIGHTWEIGHT ALTERNATIVES ===
        "distilbert": {
            "model_name": "distilbert-base-uncased",
            "description": "Fast distilled BERT model",
            "size_gb": 0.3,
            "speed": "very_fast",
            "quality": "fair",
            "free": True,
            "task_type": "classification",
            "primary_task": "mixed",  # Can do both classification and generation
            "pipeline_type": "text-classification",
            "generation_params": {
                "max_new_tokens": 100,
                "do_sample": True,
                "temperature": 0.4,
                "return_full_text": False
            },
            "max_input_length": 512,
            "language": "english",
            "requires_tokenizer_config": False,
            "response_parser": "classification",
            "prompt_removal_strategy": None,
            "model_type": "text"
        },
        
        # === MEDIUM MODELS ===
        "flan-t5-small": {
            "model_name": "google/flan-t5-small",
            "description": "Instruction-tuned T5 small model",
            "size_gb": 0.3,
            "speed": "very_fast",
            "quality": "good",
            "free": True,
            "task_type": "generation",
            "primary_task": "generation",  # Use generation pipeline
            "pipeline_type": "text2text-generation",
            "generation_params": {
                "max_length": 200,
                "do_sample": True,
                "temperature": 0.3,
                "early_stopping": True,
                "num_beams": 1
            },
            "max_input_length": 350,
            "language": "multilingual",
            "requires_tokenizer_config": False,
            "response_parser": "text2text_generation",
            "prompt_removal_strategy": None,
            "model_type": "text2text",
            "fallback_params": {
                "max_length": 150,
                "do_sample": False,
                "early_stopping": True
            }
        },
        
        "flan-t5-base": {
            "model_name": "google/flan-t5-base",
            "description": "Instruction-tuned T5 base model",
            "size_gb": 1.0,
            "speed": "fast",
            "quality": "very_good",
            "free": True,
            "task_type": "generation",
            "primary_task": "generation",  # Use generation pipeline
            "pipeline_type": "text2text-generation",
            "generation_params": {
                "max_length": 250,
                "do_sample": True,
                "temperature": 0.3,
                "early_stopping": True,
                "num_beams": 1
            },
            "max_input_length": 350,
            "language": "multilingual",
            "requires_tokenizer_config": False,
            "response_parser": "text2text_generation",
            "prompt_removal_strategy": None,
            "model_type": "text2text",
            "fallback_params": {
                "max_length": 150,
                "do_sample": False,
                "early_stopping": True
            }
        },
    }
    
    @classmethod
    def get_recommended_model(cls, task: str = "generation", priority: str = "balanced"):
        """Get recommended model based on task and priority."""
        if task == "generation":
            if priority == "speed":
                return cls.MODELS["gpt2-spanish"]  # Faster than gemma
            elif priority == "quality":
                return cls.MODELS["gemma-7b"] 
            else:  # balanced
                return cls.MODELS["gemma-2b"]
        
        elif task == "classification":
            if priority == "speed":
                return cls.MODELS["distilbert-multilingual"]
            elif priority == "quality":
                return cls.MODELS["hate-speech-spanish"]
            else:  # balanced
                return cls.MODELS["roberta-hate"]
        
        return cls.MODELS["gemma-2b"]
    
    @classmethod
    def get_fast_models(cls) -> List[str]:
        """Get list of very fast models for comparison."""
        return [
            name for name, config in cls.MODELS.items()
            if config["speed"] in ["ultra_fast", "very_fast"]
        ]
    
    @classmethod
    def get_spanish_models(cls) -> List[str]:
        """Get list of Spanish-optimized models."""
        return [
            name for name, config in cls.MODELS.items()
            if config["language"] == "spanish"
        ]
    
    @classmethod
    def get_models_by_size(cls, max_size_gb: float = 1.0) -> List[str]:
        """Get models under specified size."""
        return [
            name for name, config in cls.MODELS.items()
            if config["size_gb"] <= max_size_gb
        ]
    
    @classmethod
    def get_models_by_speed(cls, min_speed: str = "fast") -> List[str]:
        """Get models with minimum speed level."""
        speed_levels = ["ultra_fast", "very_fast", "fast", "medium", "slow"]
        min_level = speed_levels.index(min_speed) if min_speed in speed_levels else 2
        
        return [
            name for name, config in cls.MODELS.items()
            if speed_levels.index(config["speed"]) <= min_level
        ]
    
    @classmethod
    def get_models_by_task(cls, task_type: str) -> List[str]:
        """Get models for specific task type."""
        return [
            name for name, config in cls.MODELS.items()
            if config["task_type"] == task_type
        ]
    
    @classmethod
    def get_fastest_model_for_task(cls, task_type: str = "generation") -> str:
        """Get the fastest available model for a specific task."""
        models = cls.get_models_by_task(task_type)
        if not models:
            # Fallback to any fast model
            models = cls.get_fast_models()
        
        if not models:
            # Ultimate fallback - get the smallest model
            models = list(cls.MODELS.keys())
            models.sort(key=lambda x: cls.MODELS[x]["size_gb"])
        
        # Sort by speed (ultra_fast first)
        speed_order = {"ultra_fast": 0, "very_fast": 1, "fast": 2, "medium": 3, "slow": 4}
        models.sort(key=lambda x: speed_order.get(cls.MODELS[x]["speed"], 5))
        
        return models[0]
    
    @classmethod
    def get_balanced_model_selection(cls, max_models: int = 6) -> List[str]:
        """Get a balanced selection of models for comparison/testing."""
        selection = []
        
        # Get fastest classification models
        fast_classifiers = [
            name for name, config in cls.MODELS.items()
            if config["task_type"] == "classification" and config["speed"] in ["ultra_fast", "very_fast"]
        ]
        
        # Get fastest generation models  
        fast_generators = [
            name for name, config in cls.MODELS.items()
            if config["task_type"] == "generation" and config["speed"] in ["ultra_fast", "very_fast", "fast"]
        ]
        
        # Add 2-3 classification models
        selection.extend(fast_classifiers[:3])
        
        # Add 2-3 generation models
        selection.extend(fast_generators[:3])
        
        # Remove duplicates while preserving order
        seen = set()
        selection = [m for m in selection if not (m in seen or seen.add(m))]
        
        # Trim to max_models
        return selection[:max_models]

class ResponseParser:
    """Handles different types of response parsing based on pipeline type."""
    
    @staticmethod
    def parse_classification_response(response, text_input="", model_config=None):
        """Parse classification pipeline responses."""
        result = {"llm_confidence": 0.0}
        
        if not response or not isinstance(response, list) or len(response) == 0:
            return result
            
        try:
            scores = response[0] if isinstance(response[0], list) else response
            
            # Find highest scoring negative class
            toxic_score = 0.0
            hate_score = 0.0
            positive_score = 0.0
            
            for item in scores:
                label = item.get('label', '').lower()
                score = item.get('score', 0.0)
                
                # Check for various negative indicators
                if any(word in label for word in ['toxic', 'hate', 'negative', '1', 'offensive', 'bad']):
                    toxic_score = max(toxic_score, score)
                if any(word in label for word in ['hate', 'odio', 'discrimin']):
                    hate_score = max(hate_score, score)
                if any(word in label for word in ['positive', '0', 'good', 'normal']):
                    positive_score = max(positive_score, score)
            
            # Determine final scores and categories
            final_score = max(toxic_score, hate_score)
            categories = []
            
            if hate_score > 0.3:
                categories.append("hate_speech")
            if toxic_score > 0.3:
                categories.append("toxic_content")
            if final_score < 0.3 and positive_score > 0.6:
                categories.append("normal_content")
            
            if not categories:
                categories = ["general"]
            
            # Determine threat assessment
            if final_score > 0.8:
                threat = "critical"
            elif final_score > 0.6:
                threat = "high"
            elif final_score > 0.4:
                threat = "medium"
            else:
                threat = "low"
            
            return {
                "llm_confidence": final_score,
                "llm_categories": categories,
                "llm_sentiment": "negative" if final_score > 0.4 else "neutral" if final_score > 0.2 else "positive",
                "llm_threat_assessment": threat
            }
        except Exception as e:
            print(f"⚠️ Classification parsing error: {e}")
            return result
    
    @staticmethod
    def parse_text_generation_response(response, text_input="", model_config=None):
        """Parse text generation pipeline responses."""
        try:
            if not response or len(response) == 0:
                return None
                
            # Extract generated text
            if isinstance(response[0], dict) and 'generated_text' in response[0]:
                generated_text = response[0]['generated_text']
                
                # Remove prompt if configured to do so
                prompt_removal = model_config.get("prompt_removal_strategy") if model_config else None
                if prompt_removal == "remove_prompt" and text_input and text_input in generated_text:
                    generated_text = generated_text.replace(text_input, '').strip()
                
                return generated_text
            elif isinstance(response[0], str):
                return response[0]
            else:
                return str(response[0]) if response[0] else ""
                
        except Exception as e:
            print(f"⚠️ Text generation parsing error: {e}")
            return None
    
    @staticmethod
    def parse_text2text_generation_response(response, text_input="", model_config=None):
        """Parse text2text generation pipeline responses (T5, etc.)."""
        try:
            if not response or len(response) == 0:
                return None
                
            # T5 models don't include the prompt in response, so no removal needed
            if isinstance(response[0], dict):
                # Try different possible keys
                for key in ['generated_text', 'text', 'output']:
                    if key in response[0]:
                        return response[0][key]
                return str(response[0])
            elif isinstance(response[0], str):
                return response[0]
            else:
                return str(response[0]) if response[0] else ""
                
        except Exception as e:
            print(f"⚠️ Text2text generation parsing error: {e}")
            return None
    
    @classmethod
    def parse_response(cls, response, parser_type, text_input="", model_config=None):
        """Main method to parse responses based on parser type."""
        if parser_type == "classification":
            return cls.parse_classification_response(response, text_input, model_config)
        elif parser_type == "text_generation":
            return cls.parse_text_generation_response(response, text_input, model_config)
        elif parser_type == "text2text_generation":
            return cls.parse_text2text_generation_response(response, text_input, model_config)
        else:
            print(f"⚠️ Unknown parser type: {parser_type}")
            return None

class EnhancedLLMPipeline:
    """Enhanced LLM pipeline with multiple models and optimization."""
    
    def __init__(self, model_priority: str = "balanced", enable_quantization: bool = True, 
                 specific_models: Optional[Dict[str, str]] = None):
        self.model_priority = model_priority
        self.enable_quantization = enable_quantization
        self.generation_model = None
        self.classification_model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model_info = {}
        self.model_type = "text"  # Default to text generation
        
        # Allow specific model override
        self.specific_models = specific_models or {}
        
        # Initialize enhanced prompt generator
        self.prompt_generator = EnhancedPromptGenerator()
        
        print(f"🤖 Initializing Enhanced LLM Pipeline (device: {self.device})")
        print("📝 Enhanced prompt generation system loaded")
        self._load_models()
    
    def _load_models(self):
        """Load optimized models based on configuration."""
        try:
            # Get recommended models or use specific ones
            if "generation" in self.specific_models:
                gen_config = LLMModelConfig.MODELS[self.specific_models["generation"]]
            else:
                gen_config = LLMModelConfig.get_recommended_model("generation", self.model_priority)
            
            if "classification" in self.specific_models:
                class_config = LLMModelConfig.MODELS[self.specific_models["classification"]]
            else:
                class_config = LLMModelConfig.get_recommended_model("classification", self.model_priority)
            
            self.model_info = {
                "generation": gen_config,
                "classification": class_config
            }
            
            print(f"📦 Loading generation model: {gen_config['model_name']}")
            print(f"   Size: {gen_config['size_gb']}GB, Speed: {gen_config['speed']}, Quality: {gen_config['quality']}")
            
            # Track current model name for type detection
            self.current_gen_model_name = gen_config['model_name']
            self.current_class_model_name = class_config['model_name']
            
            # Load generation model
            self._load_generation_model(gen_config)
            
            # Load classification model  
            self._load_classification_model(class_config)
                
        except Exception as e:
            print(f"❌ LLM pipeline loading failed: {e}")
            self.generation_model = None
            self.classification_model = None
    
    def _load_generation_model(self, gen_config: Dict):
        """Load generation model with configuration-driven approach."""
        try:
            # Quantization config for memory efficiency
            quantization_config = None
            if self.enable_quantization and torch.cuda.is_available():
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    )
                except:
                    quantization_config = None
            
            # Model-specific configurations
            model_kwargs = {}
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            # Adjust device and precision based on model
            device = 0 if self.device == "cuda" else -1
            torch_dtype = torch.float16 if self.device in ["cuda", "mps"] else torch.float32
            
            # Get pipeline type from configuration
            pipeline_type = gen_config.get("pipeline_type", "text-generation")
            
            # Create pipeline using configuration
            self.generation_model = pipeline(
                pipeline_type,
                model=gen_config["model_name"],
                device=device,
                torch_dtype=torch_dtype,
                model_kwargs=model_kwargs
            )
            
            # Store model type from configuration
            self.model_type = gen_config.get("model_type", "text")
            
            # Load tokenizer if required by configuration
            if gen_config.get("requires_tokenizer_config", False):
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(gen_config["model_name"])
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                except:
                    self.tokenizer = None
            
            print("✅ Generation model loaded successfully")
            
        except Exception as e:
            print(f"⚠️ Generation model failed: {e}")
            # Fallback to fastest available model
            try:
                fallback_model = LLMModelConfig.get_fastest_model_for_task("generation")
                fallback_config = LLMModelConfig.MODELS[fallback_model]
                
                print(f"🔄 Trying fallback model: {fallback_model}")
                self.generation_model = pipeline(
                    "text-generation", 
                    model=fallback_config["model_name"],
                    device=-1,
                    max_new_tokens=150
                )
                print("✅ Fallback generation model loaded")
            except:
                self.generation_model = None
    
    def _load_classification_model(self, class_config: Dict):
        """Load classification model."""
        try:
            print(f"📦 Loading classification model: {class_config['model_name']}")
            
            device = 0 if self.device == "cuda" else -1
            
            self.classification_model = pipeline(
                "text-classification",
                model=class_config["model_name"],
                device=device,
                return_all_scores=True
            )
            print("✅ Classification model loaded successfully")
            
        except Exception as e:
            print(f"⚠️ Classification model failed: {e}")
            self.classification_model = None
    
    def analyze_content(self, text: str, analysis_context: Dict) -> Dict:
        """Comprehensive LLM analysis using enhanced prompting system."""
        result = {
            "llm_explanation": "",
            "llm_confidence": 0.0,
            "llm_categories": [],
            "llm_sentiment": "neutral",
            "llm_threat_assessment": "low",
            "llm_analysis_type": "comprehensive",
            "processing_time": 0.0,
            "model_info": self.get_model_info()
        }
        
        start_time = time.time()
        
        try:
            # Create prompt context from analysis results
            prompt_context = create_context_from_analysis(analysis_context)
            
            # Fast classification first (if available)
            if self.classification_model:
                class_result = self._classify_content(text)
                result.update(class_result)
            
            # Determine optimal analysis type based on context
            analysis_type = self._determine_analysis_type(analysis_context)
            
            # Generate sophisticated prompt using EnhancedPromptGenerator
            if self.generation_model:
                sophisticated_result = self._run_enhanced_analysis(text, analysis_type, prompt_context)
                result.update(sophisticated_result)
            
            result["processing_time"] = time.time() - start_time
            result["llm_analysis_type"] = analysis_type.value
            
        except Exception as e:
            print(f"⚠️ Enhanced analysis error: {e}")
            result["llm_explanation"] = f"Error en análisis LLM avanzado: {str(e)}"
            result["processing_time"] = time.time() - start_time
            
            # Provide basic fallback analysis
            result["llm_confidence"] = 0.5
            result["llm_threat_assessment"] = analysis_context.get('threat_level', 'low').lower()
            result["llm_sentiment"] = "neutral"
        
        return result
    
    def _determine_analysis_type(self, context: Dict) -> AnalysisType:
        """Determine the optimal analysis type based on context."""
        far_right_score = context.get('far_right_score', 0.0)
        threat_level = context.get('threat_level', 'LOW')
        claims_count = context.get('claims_count', 0)
        
        # High threat situations need threat assessment
        if threat_level in ['CRITICAL', 'HIGH'] or far_right_score > 0.7:
            return AnalysisType.THREAT_ASSESSMENT
        
        # Many claims need verification focus
        if claims_count >= 3:
            return AnalysisType.CLAIM_VERIFICATION
        
        # Medium threat might be misinformation
        if far_right_score > 0.4 or 'conspir' in str(context).lower():
            return AnalysisType.MISINFORMATION
        
        # Default to comprehensive analysis
        return AnalysisType.COMPREHENSIVE
    
    def _run_enhanced_analysis(self, text: str, analysis_type: AnalysisType, prompt_context) -> Dict:
        """Run enhanced analysis with sophisticated prompting."""
        try:
            # For classification models, use classification approach directly
            # Use configuration-based detection instead of hardcoded names
            gen_config = self.model_info.get("generation", {})
            class_config = self.model_info.get("classification", {})
            
            # Check if this should use pure classification approach
            use_classification_approach = (
                gen_config.get("primary_task") == "classification" or 
                class_config.get("primary_task") == "classification"
            )
            
            if use_classification_approach and self.classification_model:
                model_name = gen_config.get("model_name", "") or class_config.get("model_name", "")
                print(f"🔍 Using pure classification approach for {model_name}")
                # Use classification pipeline directly for better results - skip generation entirely
                classification_result = self._classify_content(text)
                confidence = classification_result.get("llm_confidence", 0.3)
                
                if confidence > 0.1:  # Very low threshold for classification models
                    # Create meaningful explanation based on classification
                    categories = classification_result.get('llm_categories', ['general'])
                    threat_level = classification_result.get("llm_threat_assessment", "low")
                    
                    if confidence > 0.7:
                        explanation = f"Clasificación detecta contenido altamente problemático con indicadores claros de {', '.join(categories)}. Requiere atención inmediata."
                    elif confidence > 0.5:
                        explanation = f"Clasificación detecta contenido potencialmente problemático con elementos de {', '.join(categories)}. Recomendable revisión."
                    elif confidence > 0.3:
                        explanation = f"Clasificación detecta indicios menores relacionados con {', '.join(categories)}. Nivel de riesgo bajo."
                    else:
                        explanation = "Clasificación no detecta problemas significativos en el contenido analizado."
                    
                    return {
                        "llm_explanation": explanation,
                        "llm_confidence": min(0.85, confidence + 0.15),  # Boost confidence for good classifications
                        "llm_categories": categories,
                        "llm_sentiment": classification_result.get("llm_sentiment", "neutral"),
                        "llm_threat_assessment": threat_level
                    }
                else:
                    # Even low confidence classifications provide useful information
                    return {
                        "llm_explanation": "Clasificación rápida completada sin detectar problemas significativos.",
                        "llm_confidence": 0.7,  # Good confidence for normal content
                        "llm_categories": ["normal_content"],
                        "llm_sentiment": "neutral",
                        "llm_threat_assessment": "low"
                    }
            
            # For generation models, use text generation
            model_name = gen_config.get("model_name", "unknown")
            print(f"📝 Using generation approach for {model_name}")
            # Generate sophisticated prompt
            enhanced_prompt = self.prompt_generator.generate_prompt(
                text=text,
                analysis_type=analysis_type,
                context=prompt_context
            )
            
            # Get generation parameters and input limits from configuration
            gen_config = self.model_info.get("generation", {})
            generation_params = gen_config.get("generation_params", {}).copy()
            max_input_length = gen_config.get("max_input_length", 800)
            pipeline_type = gen_config.get("pipeline_type", "text-generation")
            
            # Handle input truncation based on model configuration
            if len(enhanced_prompt) > max_input_length:
                enhanced_prompt = enhanced_prompt[:max_input_length] + "..."
                print(f"🔄 Truncated input to {max_input_length} characters for {gen_config.get('model_name', 'model')}")
            
            # Add dynamic parameters based on configuration
            requires_tokenizer = gen_config.get("requires_tokenizer_config", False)
            if requires_tokenizer and self.tokenizer:
                if self.tokenizer.eos_token_id:
                    generation_params["pad_token_id"] = self.tokenizer.eos_token_id
                elif self.tokenizer.pad_token_id:
                    generation_params["pad_token_id"] = self.tokenizer.pad_token_id
                else:
                    generation_params["pad_token_id"] = 50256
            
            # Generate response using the enhanced prompt with error handling
            try:
                # Input truncation is already handled above based on model configuration
                response = self.generation_model(enhanced_prompt, **generation_params)
            except Exception as gen_error:
                print(f"⚠️ Generation model error: {gen_error}")
                # Try with even more minimal parameters as fallback
                try:
                    # Use fallback parameters from model configuration with smart defaults
                    minimal_params = gen_config.get("fallback_params")
                    if not minimal_params:
                        # Generate sensible fallback based on response parser type
                        parser_type = gen_config.get("response_parser", "text_generation")
                        if parser_type == "text2text_generation":
                            minimal_params = {
                                "max_length": 150,
                                "do_sample": False,
                                "early_stopping": True
                            }
                        else:
                            minimal_params = {
                                "max_new_tokens": 50,
                                "do_sample": False,
                                "truncation": True,
                                "pad_token_id": 50256
                            }
                    # Further truncate for fallback
                    fallback_length = max_input_length // 2
                    short_prompt = enhanced_prompt[:fallback_length] if len(enhanced_prompt) > fallback_length else enhanced_prompt
                    response = self.generation_model(short_prompt, **minimal_params)
                except Exception as fallback_error:
                    print(f"⚠️ Fallback generation also failed: {fallback_error}")
                    # Return fallback response for generation errors
                    return {
                        "llm_explanation": f"Error en generación: {str(gen_error)[:100]}",
                        "llm_confidence": 0.1,
                        "llm_categories": [],
                        "llm_sentiment": "neutral",
                        "llm_threat_assessment": "low"
                    }
            
            # Extract and parse the response using configuration-driven approach
            if response and len(response) > 0:
                # Get parser type from configuration
                parser_type = gen_config.get("response_parser", "text_generation")
                
                # Use the ResponseParser to handle the response
                generated_text = ResponseParser.parse_response(
                    response, parser_type, enhanced_prompt, gen_config
                )
                
                # Ensure we have some text to work with
                if not generated_text or len(str(generated_text).strip()) < 5:
                    generated_text = "No response generated"
                
                # Try to extract JSON response
                json_response = self._extract_json_response(str(generated_text), enhanced_prompt)
                
                if json_response:
                    # Convert to our standard format
                    return self._convert_enhanced_response(json_response, analysis_type)
                else:
                    # Fallback to text extraction
                    return self._extract_text_response(str(generated_text), enhanced_prompt)
            else:
                # No response generated - return basic fallback
                return {
                    "llm_explanation": "No se pudo generar análisis LLM",
                    "llm_confidence": 0.3,
                    "llm_categories": [],
                    "llm_sentiment": "neutral",
                    "llm_threat_assessment": "low"
                }
            
        except Exception as e:
            print(f"⚠️ Enhanced analysis error: {e}")
            return {
                "llm_explanation": f"Error en análisis: {str(e)}",
                "llm_confidence": 0.2,
                "llm_categories": [],
                "llm_sentiment": "neutral",
                "llm_threat_assessment": "low"
            }
        
        return {
            "llm_explanation": "Análisis LLM completado con formato básico",
            "llm_confidence": 0.5
        }
    
    def _extract_json_response(self, generated_text: str, prompt: str) -> Optional[Dict]:
        """Extract JSON response from generated text."""
        try:
            # Remove the original prompt
            response_text = generated_text.replace(prompt, '').strip()
            
            # Find JSON content
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                return json.loads(json_text)
        except Exception as e:
            print(f"⚠️ JSON extraction error: {e}")
        
        return None
    
    def _convert_enhanced_response(self, json_response: Dict, analysis_type: AnalysisType) -> Dict:
        """Convert enhanced JSON response to standard format."""
        result = {
            "llm_explanation": json_response.get('explicacion', ''),
            "llm_confidence": 0.8,  # High confidence for structured responses
            "llm_categories": [],
            "llm_sentiment": "neutral",
            "llm_threat_assessment": "low"
        }
        
        # Extract threat assessment
        if 'nivel_amenaza' in json_response:
            threat_map = {
                'critico': 'critical',
                'alto': 'high', 
                'medio': 'medium',
                'bajo': 'low',
                'minimo': 'minimal'
            }
            result["llm_threat_assessment"] = threat_map.get(json_response['nivel_amenaza'], 'low')
        
        # Extract categories based on analysis type
        if analysis_type == AnalysisType.THREAT_ASSESSMENT:
            if 'tipo_amenaza' in json_response:
                result["llm_categories"] = json_response['tipo_amenaza']
        elif 'tecnicas_manipulacion' in json_response:
            result["llm_categories"] = json_response['tecnicas_manipulacion']
        
        # Extract sentiment
        if 'sesgo_politico' in json_response:
            bias = str(json_response['sesgo_politico']).lower()
            if 'extrema' in bias:
                result["llm_sentiment"] = 'negative'
            elif 'centro' in bias:
                result["llm_sentiment"] = 'neutral'
            else:
                result["llm_sentiment"] = 'negative'
        
        return result
    
    def _extract_text_response(self, generated_text: str, prompt: str) -> Dict:
        """Extract analysis from text response when JSON parsing fails."""
        response_text = generated_text.replace(prompt, '').strip()
        
        # Clean up the response
        if len(response_text) > 400:
            response_text = response_text[:397] + "..."
        
        # Try to extract useful information from the text
        confidence = 0.6
        sentiment = "neutral"
        threat_assessment = "low"
        categories = []
        
        # Simple keyword-based analysis of the generated text
        text_lower = response_text.lower()
        
        # Look for threat indicators
        if any(word in text_lower for word in ['peligro', 'amenaza', 'crítico', 'grave', 'riesgo alto']):
            threat_assessment = "high"
            confidence = 0.75
        elif any(word in text_lower for word in ['problema', 'preocup', 'medio', 'moderado']):
            threat_assessment = "medium"
            confidence = 0.65
        
        # Look for sentiment indicators
        if any(word in text_lower for word in ['negativ', 'odio', 'desprecio', 'hostil']):
            sentiment = "negative"
        elif any(word in text_lower for word in ['positiv', 'neutral', 'equilibr']):
            sentiment = "positive" if 'positiv' in text_lower else "neutral"
        
        # Look for categories
        if any(word in text_lower for word in ['odio', 'hate', 'discrimin']):
            categories.append("hate_speech")
        if any(word in text_lower for word in ['desinform', 'fake', 'falso']):
            categories.append("misinformation")
        if any(word in text_lower for word in ['manipul', 'sesgo']):
            categories.append("manipulation")
        
        if not categories:
            categories = ["general"]
        
        return {
            "llm_explanation": response_text if response_text else "Análisis completado mediante extracción de texto",
            "llm_confidence": confidence,
            "llm_categories": categories,
            "llm_sentiment": sentiment,
            "llm_threat_assessment": threat_assessment
        }
    
    def _classify_content(self, text: str) -> Dict:
        """Fast content classification using configuration-driven parsing."""
        if not self.classification_model:
            return {"llm_confidence": 0.0}
        
        try:
            # Get classification model configuration
            class_config = self.model_info.get("classification", {})
            
            # Truncate text based on model configuration
            max_length = class_config.get("max_input_length", 512)
            text_truncated = text[:max_length] if len(text) > max_length else text
            
            # Get classification results
            results = self.classification_model(text_truncated)
            
            # Use ResponseParser to handle classification results
            if results:
                return ResponseParser.parse_classification_response(results, text_truncated, class_config)
            
        except Exception as e:
            print(f"⚠️ Classification error: {e}")
        
        return {"llm_confidence": 0.0}
    
    def cleanup_memory(self):
        """Clean up GPU/memory resources."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print("🧹 Memory cleanup completed")
        except Exception as e:
            print(f"⚠️ Memory cleanup error: {e}")
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        info = {
            "device": self.device,
            "generation_model": None,
            "classification_model": None,
            "quantization_enabled": self.enable_quantization,
            "model_configs": self.model_info
        }
        
        if self.generation_model:
            try:
                if hasattr(self.generation_model, 'model'):
                    model_name = str(self.generation_model.model).split('/')[-1] if '/' in str(self.generation_model.model) else str(self.generation_model.model)
                    info["generation_model"] = model_name
                else:
                    info["generation_model"] = "loaded"
            except:
                info["generation_model"] = "loaded"
        
        if self.classification_model:
            try:
                if hasattr(self.classification_model, 'model'):
                    model_name = str(self.classification_model.model).split('/')[-1] if '/' in str(self.classification_model.model) else str(self.classification_model.model)
                    info["classification_model"] = model_name
                else:
                    info["classification_model"] = "loaded"
            except:
                info["classification_model"] = "loaded"
        
        return info
