"""
Enhanced LLM Pipeline for managing model loading, inference, and response processing.

Handles concurrent model loading, device management, and provides unified interface
for different LLM tasks in the Spanish far-right content analysis system.
"""

import asyncio
import gc
import os
import time
from typing import Dict, List, Optional, Any, Union
import torch
from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import psutil

from .model_configs import LLMModelConfig
from .response_parser import ResponseParser
from .categories import Categories


class EnhancedLLMPipeline:
    """Enhanced pipeline for LLM operations with model management and response parsing."""

    def __init__(self, model_name=None, model_config=None, device=None, verbose=False):
        """Initialize the enhanced LLM pipeline."""
        self.verbose = verbose
        self.model_name = model_name
        self.model_config = model_config or LLMModelConfig.get_model_config(model_name)
        self.device = device or self._determine_device()
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.load_time = None
        self.memory_usage = None

        if self.verbose:
            print(f"🔧 Initializing EnhancedLLMPipeline with model: {model_name}")

    def _determine_device(self):
        """Determine the best available device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _get_quantization_config(self):
        """Get quantization configuration for the model."""
        if self.model_config and self.model_config.get("quantization") == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        return None

    def _load_model_direct(self):
        """Load model directly using Auto classes for more control."""
        try:
            quantization_config = self._get_quantization_config()

            # Determine model class based on task
            task = self.model_config.get("task", "text-generation")
            if task == "text2text-generation":
                model_class = AutoModelForSeq2SeqLM
            else:
                model_class = AutoModelForCausalLM

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Load model with quantization if specified
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto" if self.device == "cuda" else None,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config

            self.model = model_class.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            if self.verbose:
                print(f"✅ Model {self.model_name} loaded directly")

        except Exception as e:
            print(f"❌ Direct model loading failed: {e}")
            raise

    def load_model(self):
        """Load the model and create pipeline."""
        if self.is_loaded:
            return True

        start_time = time.time()

        try:
            # Try pipeline approach first
            task = self.model_config.get("task", "text-generation")
            quantization_config = self._get_quantization_config()

            pipeline_kwargs = {
                "model": self.model_name,
                "device": 0 if self.device == "cuda" else -1,
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }

            if quantization_config:
                pipeline_kwargs["model_kwargs"] = {"quantization_config": quantization_config}

            # Special handling for different tasks
            if task == "text-generation":
                pipeline_kwargs.update({
                    "max_new_tokens": self.model_config.get("max_new_tokens", 512),
                    "do_sample": self.model_config.get("do_sample", True),
                    "temperature": self.model_config.get("temperature", 0.7),
                    "top_p": self.model_config.get("top_p", 0.9),
                })
            elif task == "text2text-generation":
                pipeline_kwargs.update({
                    "max_length": self.model_config.get("max_length", 512),
                    "do_sample": self.model_config.get("do_sample", False),
                })

            self.pipeline = pipeline(task=task, **pipeline_kwargs)

            self.load_time = time.time() - start_time
            self.memory_usage = psutil.virtual_memory().percent
            self.is_loaded = True

            if self.verbose:
                print(f"✅ Model {self.model_name} loaded in {self.load_time:.2f}s")
                print(f"📊 Memory usage: {self.memory_usage:.1f}%")

            return True

        except Exception as e:
            if self.verbose:
                print(f"⚠️ Pipeline loading failed, trying direct loading: {e}")

            try:
                self._load_model_direct()
                self.load_time = time.time() - start_time
                self.memory_usage = psutil.virtual_memory().percent
                self.is_loaded = True

                if self.verbose:
                    print(f"✅ Model {self.model_name} loaded directly in {self.load_time:.2f}s")

                return True

            except Exception as e2:
                print(f"❌ Both loading methods failed: {e2}")
                return False

    def unload_model(self):
        """Unload the model to free memory."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None

        if self.model:
            del self.model
            self.model = None

        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

        self.is_loaded = False

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.verbose:
            print(f"🗑️ Model {self.model_name} unloaded")

    def _get_category_from_response(self, response, task_type):
        """Extract category from parsed response."""
        if not response:
            return Categories.GENERAL

        if task_type == "classification":
            categories = response.get("llm_categories", [])
            return categories[0] if categories else Categories.GENERAL
        else:
            # For generation tasks, return the generated text
            return response

    def run_inference(self, text_input, task_type="classification", **kwargs):
        """Run inference with the loaded model."""
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError(f"Failed to load model {self.model_name}")

        try:
            # Prepare input
            if task_type == "classification":
                inputs = text_input
            else:
                # For generation tasks, format as instruction
                if "instruction_template" in self.model_config:
                    template = self.model_config["instruction_template"]
                    inputs = template.format(text=text_input)
                else:
                    inputs = text_input

            # Run inference
            if self.pipeline:
                # Use pipeline
                if task_type == "classification":
                    result = self.pipeline(inputs)
                else:
                    result = self.pipeline(inputs, **kwargs)
            else:
                # Use direct model inference
                if not self.tokenizer or not self.model:
                    raise RuntimeError("Model not properly loaded")

                inputs_tokenized = self.tokenizer(inputs, return_tensors="pt")
                if self.device != "cpu":
                    inputs_tokenized = {k: v.to(self.device) for k, v in inputs_tokenized.items()}

                with torch.no_grad():
                    outputs = self.model.generate(**inputs_tokenized, **kwargs)

                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                result = [result]  # Wrap in list for consistency

            # Parse response
            parser_type = self.model_config.get("parser_type", task_type)
            parsed_response = ResponseParser.parse_response(
                result, parser_type, text_input, self.model_config
            )

            return parsed_response

        except Exception as e:
            print(f"❌ Inference failed: {e}")
            return None

    def get_category(self, text_input, **kwargs):
        """Get category classification for input text."""
        response = self.run_inference(text_input, task_type="classification", **kwargs)
        return self._get_category_from_response(response, "classification")

    def generate_text(self, text_input, **kwargs):
        """Generate text response for input."""
        return self.run_inference(text_input, task_type="text_generation", **kwargs)

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.unload_model()