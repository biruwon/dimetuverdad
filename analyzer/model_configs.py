"""
Simplified model configurations for the two models actually used in the system.

Only keeps gemini-2.5-flash and gemma3:4b configurations.
"""

from typing import Dict, List, Optional


class LLMModelConfig:
    """Simplified configuration for only the two models actually used."""

    MODELS = {
        # === GEMINI MODELS ===
        "gemini-2.5-flash": {
            "model_name": "gemini-2.5-flash",
            "description": "Google Gemini 2.5 Flash - fast multimodal analysis",
            "size_gb": None,  # Cloud model
            "speed": "fast",
            "quality": "excellent",
            "free": False,
            "task_type": "generation",
            "primary_task": "generation",
            "pipeline_type": "gemini",
            "complexity_level": "full",
            "generation_params": {
                "temperature": 0.2,
                "max_tokens": 512
            },
            "max_input_length": 1000000,  # Large context window
            "language": "multilingual",
            "requires_tokenizer_config": False,
            "response_parser": "gemini",
            "prompt_removal_strategy": None,
            "model_type": "multimodal",
            "api_required": True
        },

        # === OLLAMA MODELS ===
        "gemma3:4b": {
            "model_name": "gemma3:4b",
            "description": "Gemma 3 4B via Ollama - local multimodal analysis",
            "size_gb": 8.0,
            "speed": "medium",
            "quality": "very_good",
            "free": True,
            "task_type": "generation",
            "primary_task": "generation",
            "pipeline_type": "ollama",
            "complexity_level": "full",
            "generation_params": {
                "temperature": 0.3,
                "max_tokens": 512
            },
            "max_input_length": 8000,
            "language": "multilingual",
            "requires_tokenizer_config": False,
            "response_parser": "ollama_chat",
            "prompt_removal_strategy": None,
            "model_type": "multimodal",
            "ollama_config": {
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama"
            }
        },

        # === CLASSIFICATION MODELS ===
        "tiny-bert": {
            "model_name": "huawei-noah/TinyBERT_General_4L_312D",
            "description": "TinyBERT for fast classification tasks",
            "size_gb": 0.1,
            "speed": "ultra_fast",
            "quality": "basic",
            "free": True,
            "task_type": "classification",
            "primary_task": "classification",
            "pipeline_type": "transformers",
            "complexity_level": "basic",
            "max_input_length": 512,
            "language": "multilingual",
            "requires_tokenizer_config": False,
            "response_parser": "classification",
            "prompt_removal_strategy": None,
            "model_type": "text_only"
        }
    }

    @classmethod
    def get_recommended_model(cls, task: str = "generation", priority: str = "balanced"):
        """Get recommended model based on task and priority."""
        if task == "generation":
            if priority == "speed":
                return cls.MODELS["gemini-2.5-flash"]  # Fastest cloud model
            elif priority == "quality":
                return cls.MODELS["gemma3:4b"]  # Local high-quality model
            else:  # balanced
                return cls.MODELS["gemini-2.5-flash"]  # Default to cloud model
        elif task == "classification":
            if priority == "speed":
                return cls.MODELS["tiny-bert"]  # Fastest classification model
            elif priority == "quality":
                return cls.MODELS["tiny-bert"]  # Only classification model available
            else:  # balanced
                return cls.MODELS["tiny-bert"]  # Default to fast classification
        else:
            return cls.MODELS["gemini-2.5-flash"]  # Default fallback

    @classmethod
    def get_models_by_task(cls, task_type: str) -> List[str]:
        """Get models for specific task type."""
        return [
            name for name, config in cls.MODELS.items()
            if config["task_type"] == task_type
        ]

    @classmethod
    def get_ollama_models(cls) -> List[str]:
        """Get list of Ollama models."""
        return [
            name for name, config in cls.MODELS.items()
            if config.get("pipeline_type") == "ollama"
        ]

    @classmethod
    def get_fast_models(cls) -> List[str]:
        """Get list of fast models."""
        return [
            name for name, config in cls.MODELS.items()
            if config["speed"] in ["ultra_fast", "very_fast", "fast"]
        ]

    @classmethod
    def get_spanish_models(cls) -> List[str]:
        """Get Spanish-optimized models."""
        return [
            name for name, config in cls.MODELS.items()
            if config.get("language") == "spanish" or config.get("language") == "multilingual"
        ]

    @classmethod
    def get_models_by_size(cls, max_size_gb: float) -> List[str]:
        """Get models under size limit."""
        return [
            name for name, config in cls.MODELS.items()
            if config.get("size_gb") and config["size_gb"] <= max_size_gb
        ]

    @classmethod
    def get_fastest_model_for_task(cls, task_type: str) -> str:
        """Get fastest model for specific task."""
        candidates = [
            name for name, config in cls.MODELS.items()
            if config["task_type"] == task_type
        ]
        
        if not candidates:
            return list(cls.MODELS.keys())[0]  # Fallback to first model
        
        # Sort by speed priority
        speed_priority = {"ultra_fast": 0, "very_fast": 1, "fast": 2, "medium": 3, "slow": 4}
        return min(candidates, key=lambda x: speed_priority.get(cls.MODELS[x]["speed"], 5))

    @classmethod
    def get_model_config(cls, model_name: str) -> Optional[Dict]:
        """Get configuration for a specific model by name."""
        return cls.MODELS.get(model_name)