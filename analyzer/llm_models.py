"""
LLM Models and Pipeline for Spanish Far-Right Analysis
Extracted for reusability and model comparison.
"""

import time
import warnings
from typing import Dict, List, Optional
import torch
from transformers import (
    AutoTokenizer, 
    pipeline,
    BitsAndBytesConfig
)
from openai import OpenAI
from .categories import Categories

from .prompts import (
    EnhancedPromptGenerator, 
    create_context_from_analysis, 
    build_spanish_classification_prompt,
    build_ollama_system_prompt,
    build_generation_system_prompt
)
from utils.text_utils import normalize_text

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
            "complexity_level": "simple",  # Ultra-fast models use simple prompts
            "generation_params": {},  # No generation params for pure classification
            "max_input_length": 512,
            "language": "multilingual",
            "requires_tokenizer_config": False,
            "response_parser": "classification",
            "prompt_removal_strategy": None,
            "model_type": "text",
            "special_requirements": "Classification only - no generation parameters"
        },
        
        # === FAST MODELS (< 1GB) ===
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
            "complexity_level": "simple",  # Very fast models use simple prompts
            "generation_params": {},
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
            "complexity_level": "simple",  # Small Spanish model uses simple prompts
            "generation_params": {
                "max_new_tokens": 512,
                "pad_token_id": 50256
            },
            "max_input_length": 1500,
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
            "complexity_level": "simple",  # Classification models use simple prompts
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
            "complexity_level": "simple",  # Classification models use simple prompts
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
            "complexity_level": "simple",  # Classification models use simple prompts
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
            "description": "Spanish hate speech detection model",
            "size_gb": 0.4,
            "speed": "fast",
            "quality": "very_good",
            "free": True,
            "task_type": "classification",
            "primary_task": "classification",  # Use classification pipeline directly
            "pipeline_type": "text-classification",
            "complexity_level": "simple",  # Classification models use simple prompts
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
            "complexity_level": "medium",  # Mid-tier models can handle medium complexity
            "generation_params": {
                "max_new_tokens": 512,
                "pad_token_id": 2
            },
            "max_input_length": 2500,
            "language": "multilingual",
            "requires_tokenizer_config": True,
            "response_parser": "text_generation",
            "prompt_removal_strategy": "remove_prompt",
            "model_type": "text",
        },
        
        # "gemma-7b": {
        #     "model_name": "google/gemma-7b-it", 
        #     "description": "Higher quality Gemma 7B model - COMMENTED OUT: Too slow (15GB)",
        #     "size_gb": 15,
        #     "speed": "medium",
        #     "quality": "excellent", 
        #     "free": True,
        #     "task_type": "generation",
        #     "primary_task": "generation",  # Use generation pipeline
        #     "pipeline_type": "text-generation",
        #     "complexity_level": "full",  # Large models can handle full complexity
        #     "generation_params": {
        #         "max_new_tokens": 512,
        #         "pad_token_id": 2
        #     },
        #     "max_input_length": 5000,
        #     "language": "multilingual",
        #     "requires_tokenizer_config": True,
        #     "response_parser": "text_generation",
        #     "prompt_removal_strategy": "remove_prompt",
        #     "model_type": "text",
        # },

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
            "complexity_level": "simple",  # Small models use simple prompts
            "generation_params": {
                "max_new_tokens": 512,
                "pad_token_id": 2  # Only essential parameters
            },
            "max_input_length": 2000,  # 32K context window
            "language": "multilingual",
            "requires_tokenizer_config": True,
            "response_parser": "text_generation",
            "prompt_removal_strategy": "remove_prompt",
            "model_type": "text",
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
            "complexity_level": "medium",  # Instruction-tuned can handle medium complexity
            "generation_params": {
                "max_new_tokens": 512,
                "pad_token_id": 2
            },
            "max_input_length": 3000,  # 32K context window
            "language": "multilingual",
            "requires_tokenizer_config": True,
            "response_parser": "text_generation",
            "prompt_removal_strategy": "remove_prompt",
            "model_type": "text",
        },

        # "gemma-3-4b-it": {
        #     "model_name": "google/gemma-3-4b-it",
        #     "description": "Multimodal Gemma 3 4B with image understanding - COMMENTED OUT: Too large (8GB)",
        #     "size_gb": 8.0,
        #     "speed": "fast",
        #     "quality": "excellent",
        #     "free": True,
        #     "task_type": "generation",
        #     "primary_task": "generation",
        #     "pipeline_type": "text-generation",
        #     "complexity_level": "full",  # Large multimodal models can handle full complexity
        #     "generation_params": {
        #         "max_new_tokens": 512,
        #         "pad_token_id": 2
        #     },
        #     "max_input_length": 6000,  # 128K context window
        #     "language": "multilingual",
        #     "requires_tokenizer_config": True,
        #     "response_parser": "text_generation",
        #     "prompt_removal_strategy": "remove_prompt",
        #     "model_type": "multimodal",  # Supports images
        # },

        # === GROK MODELS (xAI) ===
        # "grok-2": {
        #     "model_name": "xai-org/grok-2",
        #     "description": "Grok 2 - COMMENTED OUT: Extremely large (500GB)",
        #     "size_gb": 500,  # Very large model
        #     "speed": "very_slow",
        #     "quality": "outstanding",
        #     "free": True,
        #     "task_type": "generation",
        #     "primary_task": "generation",
        #     "pipeline_type": "text-generation",
        #     "complexity_level": "full",  # Large reasoning models can handle full complexity
        #     "generation_params": {
        #         "max_new_tokens": 512,
        #         "do_sample": True,
        #         "temperature": 0.2,
        #         "return_full_text": False
        #     },
        #     "max_input_length": 8000,
        #     "language": "multilingual",
        #     "requires_tokenizer_config": True,
        #     "response_parser": "text_generation",
        #     "prompt_removal_strategy": "remove_prompt",
        #     "model_type": "text",
        # },
        
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
            "complexity_level": "simple",  # Fast models use simple prompts
            "generation_params": {},
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
            "complexity_level": "simple",  # Small T5 models use simple prompts
            "generation_params": {
                "max_length": 512
            },
            "max_input_length": 1000,
            "language": "multilingual",
            "requires_tokenizer_config": False,
            "response_parser": "text2text_generation",
            "prompt_removal_strategy": None,
            "model_type": "text2text"
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
            "complexity_level": "medium",  # Base T5 models can handle medium complexity
            "generation_params": {
                "max_length": 512
            },
            "max_input_length": 1200,
            "language": "multilingual",
            "requires_tokenizer_config": False,
            "response_parser": "text2text_generation",
            "prompt_removal_strategy": None,
            "model_type": "text2text"
        },

        # === OLLAMA MODELS ===
        "gpt-oss-20b": {
            "model_name": "gpt-oss:20b",
            "description": "High-quality 20B parameter model via Ollama - excellent for fact-checking and analysis",
            "size_gb": 20.0,  # Approximate size
            "speed": "medium",
            "quality": "excellent",
            "free": True,
            "task_type": "generation",  # Primary task
            "primary_task": "generation",
            "secondary_tasks": ["classification"],  # Can also do classification via prompting
            "pipeline_type": "ollama",  # Special pipeline type for Ollama
            "complexity_level": "full",  # Large models can handle full complexity
            "generation_params": {
                "temperature": 0.3,
                "max_tokens": 512
            },
            "max_input_length": 8000,  # Generous context window
            "language": "multilingual",
            "requires_tokenizer_config": False,
            "response_parser": "ollama_chat",
            "prompt_removal_strategy": None,
            "model_type": "chat",
            "ollama_config": {
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama"
            }
        },
        "llama3.1-8b": {
            "model_name": "llama3.1:8b",
            "description": "Llama 3.1 8B via Ollama - fast and capable for both generation and classification",
            "size_gb": 8.0,  # Approximate size
            "speed": "fast",
            "quality": "very_good",
            "free": True,
            "task_type": "generation",  # Primary task
            "primary_task": "generation", 
            "secondary_tasks": ["classification"],  # Can also do classification via prompting
            "pipeline_type": "ollama",  # Special pipeline type for Ollama
            "complexity_level": "full",  # Large models can handle full complexity
            "generation_params": {
                "temperature": 0.2,
                "max_tokens": 512
            },
            "max_input_length": 8000,  # Generous context window
            "language": "multilingual",
            "requires_tokenizer_config": False,
            "response_parser": "ollama_chat",
            "prompt_removal_strategy": None,
            "model_type": "chat",
            "ollama_config": {
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama"
            }
        },

        "gpt-oss-fast": {
            "model_name": "gpt-oss-fast:latest",
            "description": "Optimized 20B model with reduced context (2048) and CPU processing - faster for classification",
            "size_gb": 20.0,
            "speed": "fast",  # Faster due to optimizations
            "quality": "excellent",
            "free": True,
            "task_type": "generation",
            "primary_task": "generation",
            "secondary_tasks": ["classification"],
            "pipeline_type": "ollama",
            "complexity_level": "full",
            "generation_params": {
                "temperature": 0.1,  # Lower for more consistent classification
                "max_tokens": 128   # Shorter for classification tasks
            },
            "max_input_length": 2000,  # Reduced context window
            "language": "multilingual",
            "requires_tokenizer_config": False,
            "response_parser": "ollama_chat",
            "prompt_removal_strategy": None,
            "model_type": "chat",
            "ollama_config": {
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama"
            }
        }
    }
    
    @classmethod
    def get_recommended_model(cls, task: str = "generation", priority: str = "balanced"):
        """Get recommended model based on task and priority."""
        if task == "generation":
            if priority == "speed":
                return cls.MODELS["flan-t5-small"]  # Fastest proven model (4s vs 11s+)
            elif priority == "quality":
                return cls.MODELS["gemma-7b"] 
            else:  # balanced
                return cls.MODELS["gpt-oss-20b"]  # Original model is best overall (46s vs 2min+)
        
        elif task == "classification":
            if priority == "speed":
                return cls.MODELS["distilbert-multilingual"]
            elif priority == "quality":
                return cls.MODELS["hate-speech-spanish"]
            else:  # balanced
                return cls.MODELS["roberta-hate"]
        
        return cls.MODELS["flan-t5-small"]  # Default to fastest proven model
    
    @classmethod
    def get_fast_models(cls) -> List[str]:
        """Get list of very fast models for comparison, excluding incompatible ones."""
        compatible_models = []
        for name, config in cls.MODELS.items():
            # Check if model is fast
            if config["speed"] in ["ultra_fast", "very_fast"]:
                # Exclude models with known compatibility issues
                if not config.get("compatibility_issues"):
                    compatible_models.append(name)
                else:
                    print(f"⚠️ Skipping {name}: {config.get('special_requirements', 'compatibility issues')}")
        return compatible_models
    

    
    @classmethod
    def get_models_by_task(cls, task_type: str) -> List[str]:
        """Get models for specific task type."""
        return [
            name for name, config in cls.MODELS.items()
            if config["task_type"] == task_type
        ]
    
    @classmethod
    def get_fast_models(cls) -> List[str]:
        """Get list of very fast models for comparison, excluding incompatible ones."""
        compatible_models = []
        for name, config in cls.MODELS.items():
            # Check if model is fast
            if config["speed"] in ["ultra_fast", "very_fast"]:
                # Exclude models with known compatibility issues
                if not config.get("compatibility_issues"):
                    compatible_models.append(name)
                else:
                    print(f"⚠️ Skipping {name}: {config.get('special_requirements', 'compatibility issues')}")
        return compatible_models
    
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
    def get_fastest_model_for_task(cls, task_type: str = "generation") -> str:
        """Get the fastest available model for a specific task, excluding incompatible ones."""
        models = cls.get_models_by_task(task_type)
        
        # Filter out incompatible models
        compatible_models = [
            name for name in models 
            if not cls.MODELS[name].get("compatibility_issues")
        ]
        
        if not compatible_models:
            # Fallback to any fast model if no compatible models found
            all_models = cls.get_fast_models()
            if all_models:
                compatible_models = all_models
            else:
                # Ultimate fallback - get the smallest compatible model
                all_compatible = [
                    name for name, config in cls.MODELS.items()
                    if not config.get("compatibility_issues")
                ]
                if all_compatible:
                    all_compatible.sort(key=lambda x: cls.MODELS[x]["size_gb"])
                    compatible_models = all_compatible
                else:
                    # Last resort - return any model
                    compatible_models = list(cls.MODELS.keys())
        
        # Sort by speed (ultra_fast first)
        speed_order = {"ultra_fast": 0, "very_fast": 1, "fast": 2, "medium": 3, "slow": 4}
        compatible_models.sort(key=lambda x: speed_order.get(cls.MODELS[x]["speed"], 5))
        
        return compatible_models[0]
    
    @classmethod
    def get_ollama_models(cls) -> List[str]:
        """Get list of Ollama models."""
        return [
            name for name, config in cls.MODELS.items()
            if config.get("pipeline_type") == "ollama"
        ]
    


class ResponseParser:
    """Handles different types of response parsing based on pipeline type."""
    
    @staticmethod
    def parse_classification_response(response, text_input="", model_config=None):
        """Parse classification pipeline responses."""
        result = {"llm_categories": []}
        
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
            
            # Use lower thresholds for detection
            if hate_score > 0.3:
                categories.append(Categories.HATE_SPEECH)
            if toxic_score > 0.3:
                categories.append("toxic_content")
            if final_score < 0.3 and positive_score > 0.6:
                categories.append("normal_content")
            
            if not categories:
                categories = [Categories.GENERAL]
            
            return {
                "llm_categories": categories
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
        
        # Ollama-specific attributes
        self.ollama_client = None
        self.ollama_model_name = None
        
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
            pipeline_type = gen_config.get("pipeline_type", "text-generation")
            
            # Check if this is an Ollama model
            if pipeline_type == "ollama":
                print(f"📦 Loading Ollama model: {gen_config['model_name']}")
                ollama_config = gen_config.get("ollama_config", {})
                
                # Create OpenAI client for Ollama
                self.ollama_client = OpenAI(
                    base_url=ollama_config.get("base_url", "http://localhost:11434/v1"),
                    api_key=ollama_config.get("api_key", "ollama")
                )
                
                # Store model name for Ollama calls
                self.ollama_model_name = gen_config["model_name"]
                self.generation_model = "ollama"  # Special marker
                self.model_type = gen_config.get("model_type", "chat")
                
                print("✅ Ollama model configured successfully")
                return
            
            # Regular transformers model loading
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
            
            # Use CPU by default for stability
            device = -1  # Default to CPU for stability
            model_name = gen_config["model_name"]
            
            # Create pipeline using configuration - minimal setup for stability
            pipeline_kwargs = {
                "device": device,
                "torch_dtype": torch.float32,  # Use float32 for stability
                "trust_remote_code": False
            }
            
            # Add model-specific kwargs if needed
            if model_kwargs:
                pipeline_kwargs.update(model_kwargs)
            
            self.generation_model = pipeline(
                pipeline_type,
                model=model_name,
                **pipeline_kwargs
            )
            
            # Store model type from configuration
            self.model_type = gen_config.get("model_type", "text")
            
            # Load tokenizer if required by configuration
            if gen_config.get("requires_tokenizer_config", False):
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                except:
                    self.tokenizer = None
            
            print("✅ Generation model loaded successfully")
            
        except Exception as e:
            print(f"❌ Generation model failed: {e}")
            # Don't use fallbacks - expose the real error
            raise e
    
    def _load_classification_model(self, class_config: Dict):
        """Load classification model."""
        try:
            print(f"📦 Loading classification model: {class_config['model_name']}")
            
            # Force CPU for classification models to avoid device conflicts
            device = -1
            
            # Ensure we don't pass generation parameters to classification models
            pipeline_kwargs = {
                "device": device,
                "return_all_scores": True,
                "torch_dtype": torch.float32,
                "trust_remote_code": False
            }
            
            # Check for special requirements
            special_reqs = class_config.get("special_requirements", "")
            if "Classification only" in special_reqs:
                print(f"   ⚠️ {special_reqs}")
            
            self.classification_model = pipeline(
                "text-classification",
                model=class_config["model_name"],
                **pipeline_kwargs
            )
            print("✅ Classification model loaded successfully")
            
        except Exception as e:
            print(f"❌ Classification model failed: {e}")
            # Don't hide errors - expose them for debugging
            raise e
    
    def get_category(self, text: str) -> str:
        """
        Fast category classification using LLM with optimized prompts.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected category string
        """
        try:
            print(f"🔍 LLM category classification for content: {text[:50]}...")
            
            result = "general"  # Default fallback
            
            # Prioritize Ollama classification if available  
            if self.ollama_client:
                ollama_result = self._classify_with_ollama(text)
                ollama_categories = ollama_result.get("llm_categories", [])
                if ollama_categories:
                    result = ollama_categories[0]

            # Use generation model with classification-specific prompts
            elif self.generation_model:
                result = self._classify_with_generation_model(text)
            
            # Classification model fallback
            elif self.classification_model:
                class_result = self._classify_content(text)
                categories = class_result.get("llm_categories", [])
                if categories:
                    result = categories[0]
            
            return result
                
        except Exception as e:
            print(f"⚠️ Category detection error: {e}")
            return Categories.GENERAL
    
    def get_explanation(self, text: str, category: str = None, analysis_context: Dict = None) -> str:
        """
        Generate detailed explanation using generation model only.
        
        Args:
            text: Text to analyze
            category: Content category (hate_speech, disinformation, etc.)
            analysis_context: Dict with prior analysis results (optional)
            
        Returns:
            Detailed explanation string
        """
        try:
            if not self.generation_model:
                return "ERROR: Generation model not available - explanation impossible"
            
            # Use provided analysis_context or create empty dict
            if analysis_context is None:
                analysis_context = {}
            
            # Add category to context if provided
            if category:
                analysis_context['detected_category'] = category
            
            # Use the new explanation-specific prompt for better results
            explanation = self._generate_explanation_with_specific_prompt(text, category or Categories.GENERAL)
            
            if explanation and len(explanation.strip()) > 5:
                return explanation
            else:
                # Surface the actual issue instead of generic fallback
                return f"ERROR: LLM analysis returned insufficient explanation - got: '{explanation}' (length: {len(explanation.strip()) if explanation else 0})"
            
        except Exception as e:
            print(f"⚠️ Explanation generation error: {e}")
            # Keep the existing behavior but make it clearer this is an actual error
            return f"ERROR: Exception in explanation generation - {type(e).__name__}: {str(e)}"
    
    def _generate_explanation_with_specific_prompt(self, text: str, category: str) -> str:
        """
        Generate explanation using the specialized explanation prompt method.
        """
        try:
            if not self.generation_model or self.generation_model != "ollama":
                return "ERROR: Explanation generation requires Ollama model"
            
            # No need to convert since Categories uses string constants
            # Just ensure we have a valid category string
            if not category or not hasattr(Categories, category.upper()):
                category = Categories.GENERAL
            
            # Generate explanation-specific prompt
            explanation_prompt = self.prompt_generator.generate_explanation_prompt(text, category, model_type="ollama")
            
            # Get explanation from Ollama using the same pattern as classification
            response = self.ollama_client.chat.completions.create(
                model=self.ollama_model_name,
                messages=[
                    {"role": "system", "content": "Eres un experto analista de contenido especializado en explicar por qué un texto pertenece a una categoría específica."},
                    {"role": "user", "content": explanation_prompt}
                ],
                temperature=0.3  # Slightly higher temperature for more natural explanations
            )
            
            explanation = response.choices[0].message.content.strip()
            
            if explanation and len(explanation.strip()) > 10:
                return explanation
            else:
                return f"ERROR: Insufficient explanation generated - got: '{explanation}'"
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"ERROR: Exception in explanation prompt generation - {type(e).__name__}: {str(e)}"
    
    def analyze_content(self, text: str, analysis_context: Dict = None, category: str = None) -> Dict:
        """
        Comprehensive LLM analysis using enhanced prompting system.
        
        Args:
            text: Text to analyze
            analysis_context: Dict with prior analysis results (optional)
            category: Content category to focus on (optional)
        """
        result = {
            "llm_explanation": "",
            "llm_categories": [],
            "llm_analysis_type": "comprehensive",
            "processing_time": 0.0,
            "model_info": self.get_model_info()
        }

        start_time = time.time()

        # Normalize text before classification and generation to avoid
        # issues with combining marks or invisible characters that may
        # affect pattern/LLM prompting. Emojis and symbols are preserved.
        text_normalized = normalize_text(text)

        try:
            # Handle case where category is passed as first argument for backwards compatibility
            if isinstance(analysis_context, str):
                category = analysis_context
                analysis_context = {}
            
            # Use provided analysis_context or create empty dict
            if analysis_context is None:
                analysis_context = {}
            
            # Create prompt context from analysis results
            prompt_context = create_context_from_analysis(analysis_context)
            
            # Fast classification first (if available)
            if self.classification_model:
                # Use normalized text for classification
                class_result = self._classify_content(text_normalized)
                result.update(class_result)
            
            # Generate sophisticated prompt using EnhancedPromptGenerator with category
            if self.generation_model:
                # Use normalized text for generation prompts
                sophisticated_result = self._run_enhanced_analysis(text_normalized, category or Categories.GENERAL, prompt_context)
                result.update(sophisticated_result)
            
            result["processing_time"] = time.time() - start_time
            result["llm_analysis_type"] = category or Categories.GENERAL
            
        except Exception as e:
            print(f"⚠️ Enhanced analysis error: {e}")
            result["llm_explanation"] = f"Error en análisis LLM avanzado: {str(e)}"
            result["processing_time"] = time.time() - start_time
        
        return result
    
    def _run_enhanced_analysis(self, text: str, category: str, prompt_context) -> Dict:
        """Run enhanced analysis with sophisticated prompting using content categories."""
        try:
            # For classification models, use classification approach directly
            # Use configuration-based detection instead of hardcoded names
            gen_config = self.model_info.get("generation", {})
            class_config = self.model_info.get("classification", {})
            
            # Check if this should use pure classification approach
            # Only use classification approach if the GENERATION model is primarily for classification
            use_classification_approach = (
                gen_config.get("primary_task") == "classification"
            )
            
            if use_classification_approach and self.classification_model:
                model_name = gen_config.get("model_name", "") or class_config.get("model_name", "")
                print(f"🔍 Using pure classification approach for {model_name}")
                # Use classification pipeline directly for better results - skip generation entirely
                classification_result = self._classify_content(text)
                
                if classification_result.get("llm_categories"):
                    # Create meaningful explanation based on classification
                    categories = classification_result.get('llm_categories', ['general'])
                    
                    if 'hate_speech' in categories or 'toxic_content' in categories:
                        explanation = f"Clasificación detecta contenido problemático con indicadores de {', '.join(categories)}. Requiere atención."
                    elif 'normal_content' in categories:
                        explanation = "Clasificación completada sin detectar problemas significativos."
                    else:
                        explanation = f"Clasificación detecta elementos relacionados con {', '.join(categories)}."
                    
                    return {
                        "llm_explanation": explanation,
                        "llm_categories": categories
                    }
                else:
                    # Even if no specific categories, classification ran successfully
                    return {
                        "llm_explanation": "Clasificación rápida completada sin detectar problemas significativos.",
                        "llm_categories": ["normal_content"]
                    }
            
            # For generation models, use text generation
            model_name = gen_config.get("model_name", "unknown")
            complexity_level = gen_config.get("complexity_level", "medium")  # Default to medium if not specified
            
            print(f"📝 Using generation approach for {model_name} (complexity: {complexity_level})")
            
            # Generate prompt with appropriate complexity level and model type
            model_type = "ollama" if self.generation_model == "ollama" else "transformers"
            enhanced_prompt = self.prompt_generator.generate_prompt(
                text=text,
                category=category,
                context=prompt_context,
                complexity_level=complexity_level,
                model_type=model_type
            )
            
            # Get generation parameters from configuration
            generation_params = gen_config.get("generation_params", {}).copy()
            
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
                # Check if this is an Ollama model
                if self.generation_model == "ollama":
                    # Use Ollama chat completion
                    response = self.ollama_client.chat.completions.create(
                        model=self.ollama_model_name,
                        messages=[{"role": "user", "content": enhanced_prompt}],
                        **generation_params
                    )
                    # Extract response text from Ollama format
                    generated_text = response.choices[0].message.content
                else:
                    # Use transformers pipeline
                    response = self.generation_model(enhanced_prompt, **generation_params)
                    # Parse using ResponseParser
                    parser_type = gen_config.get("response_parser", "text_generation")
                    generated_text = ResponseParser.parse_response(
                        response, parser_type, enhanced_prompt, gen_config
                    )
            except Exception as gen_error:
                print(f"❌ Generation model error: {gen_error}")
                # Don't hide errors - raise them for debugging
                raise gen_error
            
            # Ensure we have some text to work with
            if not generated_text or len(str(generated_text).strip()) < 5:
                # LLM failed - return error message instead of fallback
                return {
                    "llm_explanation": "Error: El modelo LLM no pudo generar una respuesta",
                    "llm_categories": ["generation_error"],
                    "llm_sentiment": "neutral"
                }
            
            # Use unified extraction for all models - no JSON parsing
            return self._extract_text_response(str(generated_text), enhanced_prompt)
            
        except Exception as e:
            print(f"⚠️ Enhanced analysis error: {e}")
            return {
                "llm_explanation": f"Error en análisis: {str(e)}",
                "llm_categories": []
            }
    
    def _extract_text_response(self, generated_text: str, prompt: str) -> Dict:
        """Extract explanation from model response without truncation or modifications."""
        # Remove the prompt if present
        if prompt in generated_text:
            explanation = generated_text.replace(prompt, '').strip()
        else:
            explanation = generated_text.strip()
        
        # Only remove surrounding quotes if they completely wrap the response
        if explanation.startswith('"') and explanation.endswith('"'):
            explanation = explanation[1:-1].strip()
        
        # Handle empty responses
        if not explanation or len(explanation) < 5:
            explanation = "Análisis completado sin respuesta detallada del modelo."
        
        return {
            "llm_explanation": explanation,
            "llm_categories": ["analyzed"]
        }
    
    
    def _classify_content(self, text: str) -> Dict:
        """Fast content classification using configuration-driven parsing."""
        # Prioritize Ollama model if available (better quality than classification model)
        if self.ollama_client:
            return self._classify_with_ollama(text)
        
        # Fallback to classification model if no Ollama
        if not self.classification_model:
            return {"llm_categories": []}
        
        try:
            # Get classification model configuration
            class_config = self.model_info.get("classification", {})
            
            # Use full text for classification - no truncation
            
            # Get classification results
            results = self.classification_model(text)
            
            # Use ResponseParser to handle classification results
            if results:
                return ResponseParser.parse_classification_response(results, text, class_config)
            
        except Exception as e:
            print(f"⚠️ Classification error: {e}")
        
        return {"llm_categories": []}
    
    def _classify_with_ollama(self, text: str) -> Dict:
        """Perform classification using Ollama model with category-based analysis."""
        try:
            # Use EnhancedPromptGenerator for sophisticated classification
            classification_prompt = self.prompt_generator.generate_classification_prompt(text, model_type="ollama")
            
            print(f"🔍 Sending to Ollama model: {self.ollama_model_name}")
            print(f"🔍 Classification prompt length: {len(classification_prompt)} characters")

            response = self.ollama_client.chat.completions.create(
                model=self.ollama_model_name,
                messages=[
                    {"role": "system", "content": build_ollama_system_prompt()},
                    {"role": "user", "content": classification_prompt}
                ],
                temperature=0.1  # Lower temperature for more consistent results
                # Note: max_tokens parameter causes empty responses with gpt-oss:20b
            )
            
            # Debug: Show response object only in verbose mode
            # print(f"🔍 Response object: {response}")  # Commented out to reduce noise
            result = response.choices[0].message.content.strip().lower()
            print(f"🔍 Raw LLM response: '{result}'")
            
            # Extract category from response
            valid_categories = Categories.get_all_categories()
            
            detected_category = Categories.GENERAL  # Default fallback
            for category in valid_categories:
                if category in result:
                    detected_category = category
                    break
                    
            print(f"🔍 Extracted category: '{detected_category}'")
            print(f"🔍 Fast LLM category result: {detected_category}")
            
            return {
                "llm_categories": [detected_category],
                "llm_confidence": 0.8,  # Default confidence for Ollama
            }
                
        except Exception as e:
            print(f"⚠️ Ollama classification error: {e}")
            import traceback
            traceback.print_exc()
            return {"llm_categories": []}
    
    def _classify_with_generation_model(self, text: str) -> str:
        """
        Perform classification using generation model with optimized fast prompt.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected category string
        """
        try:
            print(f"🔍 Using generation model for classification")
            
            # PERFORMANCE OPTIMIZATION: Use much shorter, simpler classification prompt
            simple_classification_prompt = build_spanish_classification_prompt(text)
            
            print(f"🔍 Classification prompt length: {len(simple_classification_prompt)} characters")
            
            # Get generation parameters from configuration
            gen_config = self.model_info.get("generation", {})
            generation_params = gen_config.get("generation_params", {}).copy()
            generation_params.update({
                "temperature": 0.1,  # PERFORMANCE OPTIMIZATION: Lower temperature for faster, more consistent responses
                "max_tokens": 20  # PERFORMANCE OPTIMIZATION: Limit output length for speed (OpenAI API parameter)
            })
            
            # Generate response
            if self.generation_model == "ollama":
                # PERFORMANCE OPTIMIZATION: Simplified system prompt for Ollama
                print(f"🔍 Sending to Ollama model: {self.ollama_model_name}")
                
                # Fix: Use OpenAI parameter names for Ollama's OpenAI-compatible API
                ollama_params = {
                    "temperature": 0.1,    # Very low temperature for deterministic responses
                    "max_tokens": 100      # Generous limit to avoid truncation
                }
                
                response = self.ollama_client.chat.completions.create(
                    model=self.ollama_model_name,
                    messages=[
                        {"role": "system", "content": build_generation_system_prompt()},
                        {"role": "user", "content": f"Text: {text}\nCategory:"}
                    ],
                    **ollama_params
                )
                result = response.choices[0].message.content.strip().lower() if response.choices[0].message.content else ""
                print(f"🔍 Raw generation model response: '{result}'")
            else:
                # Use transformers pipeline  
                response = self.generation_model(simple_classification_prompt, **generation_params)
                parser_type = gen_config.get("response_parser", "text_generation")
                result = ResponseParser.parse_response(response, parser_type, simple_classification_prompt, gen_config)
                result = result.strip().lower() if result else ""
                print(f"🔍 Raw transformers response: '{result}'")
            
            # PERFORMANCE OPTIMIZATION: Simplified validation - just check for key categories
            valid_categories = Categories.get_all_categories()
            
            # Quick category matching
            for category in valid_categories:
                if category in result:
                    return category
            
            # Fallback to general if no valid category found
            return Categories.GENERAL
            
        except Exception as e:
            print(f"⚠️ Generation model classification error: {e}")
            return Categories.GENERAL

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
