#!/usr/bin/env python3
"""
Comprehensive test script for multimodal image + text analysis.
Tests multiple local Ollama vision models and measures performance.
"""

import os
import tempfile
import requests
import time
import sys
from datetime import datetime
from ollama import chat
from ollama import ChatResponse
from google import genai
from dotenv import load_dotenv
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

# Load environment variables from .env file
load_dotenv()

def analyze_image_with_gemini(image_path, text_content="", api_key=None, model_name='gemini-2.5-flash'):
    """Analyze image using Google Gemini API (similar to video analysis approach)."""

    try:
        # Get API key
        if not api_key:
            api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("Error: GOOGLE_API_KEY not found in environment variables")
            return None, 0

        # Initialize the client (same as video script)
        client = genai.Client(api_key=api_key)

        # Start timing
        start_time = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Gemini {model_name} analysis...")
        sys.stdout.flush()

        # Create the prompt for analysis in Spanish - focused on Twitter posts and far-right content
        prompt = f"""Analiza esta imagen y el texto adjunto que pertenecen al MISMO POST en una cuenta de Twitter/X para detectar contenido político de extrema derecha o desinformación.

TEXTO DEL POST: "{text_content}"

El contenido está en español y proviene de redes sociales. Evalúa si el post promueve o contiene:

CONTEXTO DEL POST:
- El texto e imagen son parte del mismo tweet/publicación en Twitter
- El texto proporciona contexto adicional al contenido visual de la imagen

ANÁLISIS REQUERIDO:
1. Descripción detallada del contenido visual de la imagen (¿Quiénes aparecen? ¿Qué están haciendo?)
2. Análisis del texto adjunto para detectar:
   - Discurso político de extrema derecha
   - Teorías conspirativas
   - Llamados a la acción política
   - Ataques a instituciones democráticas
   - Desinformación o fake news
   - Retórica nacionalista o anti-inmigración
3. Evaluación de la relación entre texto e imagen
4. Clasificación por categorías: hate_speech, disinformation, conspiracy_theory, far_right_bias, call_to_action, general
5. Nivel de credibilidad y sesgo político detectado

IMPORTANTE: Responde completamente en español y sé específico sobre el contenido político español. Si reconoces personas públicas, identifícalas claramente."""

        # Upload the image file to Gemini (same as video approach)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Uploading image to Gemini...")
        sys.stdout.flush()
        
        image_file = client.files.upload(file=image_path)

        # Wait for processing to complete
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for image processing...")
        sys.stdout.flush()
        
        import time as time_module
        while image_file.state.name == "PROCESSING":
            time_module.sleep(1)
            image_file = client.files.get(name=image_file.name)

        if image_file.state.name == "FAILED":
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Image processing failed: {image_file.state.name}")
            sys.stdout.flush()
            return None, time.time() - start_time

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Image uploaded and processed successfully")
        sys.stdout.flush()

        # Generate content with both image and text (same as video approach)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Analyzing with Gemini {model_name}...")
        sys.stdout.flush()
        
        response = client.models.generate_content(
            model=model_name,
            contents=[image_file, prompt]
        )

        # End timing
        end_time = time.time()
        analysis_time = end_time - start_time

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Gemini {model_name} completed in {analysis_time:.2f}s")
        sys.stdout.flush()

        return response.text, analysis_time

    except Exception as e:
        error_time = time.time() - start_time if 'start_time' in locals() else 0
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error in Gemini {model_name} analysis after {error_time:.2f}s: {e}")
        sys.stdout.flush()
        return None, error_time

def analyze_image_with_huggingface(image_path, text_content=""):
    """Analyze image using Hugging Face Transformers (BLIP-2 model)."""

    try:
        # Start timing
        start_time = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting HuggingFace BLIP-2 analysis...")
        sys.stdout.flush()

        # Load BLIP-2 model and processor
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

        # Load and process image
        image = Image.open(image_path).convert('RGB')

        # Create the prompt for analysis in Spanish
        prompt = f"Analiza esta imagen y el texto adjunto que pertenecen al MISMO POST en una cuenta de Twitter/X para detectar contenido político de extrema derecha o desinformación. TEXTO DEL POST: '{text_content}'. Describe detalladamente qué ves en la imagen y analiza si el contenido es político."

        # Process inputs
        inputs = processor(image, prompt, return_tensors="pt")

        # Generate analysis
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=500, num_beams=5, early_stopping=True)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # End timing
        end_time = time.time()
        analysis_time = end_time - start_time

        print(f"[{datetime.now().strftime('%H:%M:%S')}] HuggingFace BLIP-2 completed in {analysis_time:.2f}s")
        sys.stdout.flush()

        return generated_text, analysis_time

    except Exception as e:
        error_time = time.time() - start_time if 'start_time' in locals() else 0
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error in HuggingFace analysis after {error_time:.2f}s: {e}")
        sys.stdout.flush()
        return None, error_time

def download_image_to_temp_file(image_url):
    """Download image from URL to a temporary file and return the file path."""
    try:
        response = requests.get(image_url, stream=True, timeout=30)
        response.raise_for_status()

        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        with open(temp_file.name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded image to: {temp_file.name}")
        return temp_file.name

    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def analyze_image_with_model(model_name, image_path, text_content=""):
    """Analyze image using specified Ollama model with multimodal capabilities."""

    try:
        # Create the prompt for analysis in Spanish - focused on Twitter posts and far-right content
        prompt = f"""Analiza esta imagen y el texto adjunto que pertenecen al MISMO POST en una cuenta de Twitter/X para detectar contenido político de extrema derecha o desinformación.

TEXTO DEL POST: "{text_content}"

El contenido está en español y proviene de redes sociales. Evalúa si el post promueve o contiene:

CONTEXTO DEL POST:
- El texto e imagen son parte del mismo tweet/publicación en Twitter
- El texto proporciona contexto adicional al contenido visual de la imagen

ANÁLISIS REQUERIDO:
1. Descripción detallada del contenido visual de la imagen (¿Quiénes aparecen? ¿Qué están haciendo?)
2. Análisis del texto adjunto para detectar:
   - Discurso político de extrema derecha
   - Teorías conspirativas
   - Llamados a la acción política
   - Ataques a instituciones democráticas
   - Desinformación o fake news
   - Retórica nacionalista o anti-inmigración
3. Evaluación de la relación entre texto e imagen
4. Clasificación por categorías: hate_speech, disinformation, conspiracy_theory, far_right_bias, call_to_action, general
5. Nivel de credibilidad y sesgo político detectado

IMPORTANTE: Responde completamente en español y sé específico sobre el contenido político español. Si reconoces personas públicas, identifícalas claramente."""

        # Start timing
        start_time = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting analysis with {model_name}...")
        sys.stdout.flush()

        # Send image + text to multimodal model
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Sending image + text to {model_name}...")
        sys.stdout.flush()
        
        # Use shorter timeout for problematic models
        timeout_value = 30 if model_name == 'llama3.2-vision' else 60
        
        response: ChatResponse = chat(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_path]  # Ollama accepts file paths directly
            }],
            options={'timeout': timeout_value}  # Shorter timeout for llama3.2-vision
        )

        # End timing
        end_time = time.time()
        analysis_time = end_time - start_time

        print(f"[{datetime.now().strftime('%H:%M:%S')}] {model_name} completed in {analysis_time:.2f}s")
        sys.stdout.flush()

        return response.message.content, analysis_time

    except Exception as e:
        error_time = time.time() - start_time if 'start_time' in locals() else 0
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error in {model_name} analysis after {error_time:.2f}s: {e}")
        sys.stdout.flush()
        return None, error_time

def test_gemini_image_analysis(image_url, text_content, api_key=None, model_name='gemini-2.5-flash'):
    """Test Gemini image analysis (similar to video analysis approach)."""
    print(f"🧪 Testing Gemini {model_name.upper()} Image Analysis (similar to video approach)")
    print(f"Image URL: {image_url}")
    print(f"Text: {text_content[:100]}...")
    print()

    # Download image
    image_path = download_image_to_temp_file(image_url)
    if not image_path:
        print("❌ Failed to download image")
        return

    try:
        result, analysis_time = analyze_image_with_gemini(image_path, text_content, api_key, model_name)

        if result:
            print(f"\n✅ SUCCESS - Gemini {model_name.upper()} completed in {analysis_time:.2f}s")
            print("RESULT:")
            print("-" * 50)
            print(result)
            print("-" * 50)

            # Check if recognized president
            recognized = "✅" if ("pedro sánchez" in result.lower() or "presidente" in result.lower()) else "❌"
            print(f"Recognized Spanish President: {recognized}")

        else:
            print(f"❌ FAILED - Gemini {model_name.upper()} failed after {analysis_time:.2f}s")

    finally:
        # Clean up
        try:
            os.unlink(image_path)
        except:
            pass

def test_single_model(model_name, image_url, text_content):
    """Test a single vision model."""
    print(f"🧪 Testing single model: {model_name}")
    print(f"Image URL: {image_url}")
    print(f"Text: {text_content[:100]}...")
    print()

    # Download image
    image_path = download_image_to_temp_file(image_url)
    if not image_path:
        print("❌ Failed to download image")
        return

    try:
        result, analysis_time = analyze_image_with_model(model_name, image_path, text_content)

        if result:
            print(f"\n✅ SUCCESS - {model_name} completed in {analysis_time:.2f}s")
            print("RESULT:")
            print("-" * 50)
            print(result)
            print("-" * 50)

            # Check if recognized president
            recognized = "✅" if ("pedro sánchez" in result.lower() or "presidente" in result.lower()) else "❌"
            print(f"Recognized Spanish President: {recognized}")

        else:
            print(f"❌ FAILED - {model_name} failed after {analysis_time:.2f}s")

    finally:
        # Clean up
        try:
            os.unlink(image_path)
        except:
            pass

def test_all_vision_models(image_url, text_content, output_file):
    """Test all available vision models and compare their performance."""

    # List of vision models to test (based on what we have available)
    vision_models = [
        'gemini-2.0-flash',  # Google Gemini 2.0 Flash (higher RPM)
        'gemini-2.5-flash',  # Google Gemini 2.5 Flash (higher quality)
        'huggingface-blip2', # HuggingFace BLIP-2 model
        # 'minicpm-v',        # 5.5 GB - Tested, doesn't identify people properly
        # 'llava-llama3',     # 5.5 GB - Tested, doesn't identify people properly
        # 'llama3.2-vision',  # 7.8 GB - Works but very slow (92s)
        # 'qwen2.5',          # 4.7 GB - Text-only model (removed)
    ]

    results = []

    # Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"image_analysis_results_{timestamp}.txt"

    print(f"📁 Saving results to: {output_filename}")
    print(f"Image URL: {image_url}")
    print(f"Text content: {text_content}")
    print(f"\nExpected: Image should show Spanish President Pedro Sánchez with his wife")
    print(f"Text refers to: 'el presidente' (the president) and his family members\n")
    sys.stdout.flush()

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("=== COMPREHENSIVE MULTIMODAL IMAGE ANALYSIS TEST ===\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Image URL: {image_url}\n")
        f.write(f"Text Content: {text_content}\n")
        f.write("Expected: Image should show Spanish President Pedro Sánchez with his wife\n")
        f.write("Text refers to: 'el presidente' (the president) and his family members\n\n")
        f.write("="*80 + "\n\n")

    # Download image once for all tests
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Downloading image...")
    sys.stdout.flush()
    image_path = download_image_to_temp_file(image_url)
    if not image_path:
        print("❌ Failed to download image")
        return

    try:
        for i, model in enumerate(vision_models, 1):
            print(f"\n{'='*60}")
            print(f"TESTING MODEL {i}/{len(vision_models)}: {model}")
            print(f"{'='*60}")
            sys.stdout.flush()

            result, analysis_time = analyze_image_with_model(model, image_path, text_content)

            # Special handling for different model types
            if model == 'huggingface-blip2':
                result, analysis_time = analyze_image_with_huggingface(image_path, text_content)
            elif model.startswith('gemini-'):
                result, analysis_time = analyze_image_with_gemini(image_path, text_content, api_key=None, model_name=model)

            if result:
                print(f"\n⏱️  Analysis Time: {analysis_time:.2f} seconds")
                print(f"\n=== {model.upper()} ANALYSIS RESULT ===")
                print(result[:500] + "..." if len(result) > 500 else result)  # Truncate for terminal
                print(f"{'='*60}")
                sys.stdout.flush()

                results.append({
                    'model': model,
                    'time': analysis_time,
                    'result': result
                })

                # Save to file
                with open(output_filename, 'a', encoding='utf-8') as f:
                    f.write(f"MODEL: {model}\n")
                    f.write(f"Analysis Time: {analysis_time:.2f} seconds\n")
                    f.write(f"Success: Yes\n")
                    f.write(f"Result:\n{result}\n")
                    f.write("="*80 + "\n\n")

            else:
                print(f"❌ {model} failed to analyze the image after {analysis_time:.2f}s")
                sys.stdout.flush()
                results.append({
                    'model': model,
                    'time': analysis_time,
                    'result': None
                })

                # Save to file
                with open(output_filename, 'a', encoding='utf-8') as f:
                    f.write(f"MODEL: {model}\n")
                    f.write(f"Analysis Time: {analysis_time:.2f} seconds\n")
                    f.write(f"Success: No\n")
                    f.write(f"Result: Failed to analyze image\n")
                    f.write("="*80 + "\n\n")

        # Summary comparison
        print(f"\n{'='*80}")
        print("PERFORMANCE COMPARISON SUMMARY")
        print(f"{'='*80}")
        sys.stdout.flush()

        print(f"{'Model':<20} {'Time (sec)':<12} {'Success':<10} {'Recognized People'}")
        print("-" * 80)

        for result in results:
            success = "✅" if result['result'] else "❌"
            # Quick check if the model recognized the Spanish president
            recognized = "✅" if result['result'] and ("pedro sánchez" in result['result'].lower() or "presidente" in result['result'].lower()) else "❌"
            print(f"{result['model']:<20} {result['time']:<12.2f} {success:<10} {recognized}")

        print(f"\n{'='*80}")
        print(f"📁 Full results saved to: {output_filename}")
        sys.stdout.flush()

        # Save summary to file
        with open(output_filename, 'a', encoding='utf-8') as f:
            f.write("PERFORMANCE COMPARISON SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"{'Model':<20} {'Time (sec)':<12} {'Success':<10} {'Recognized People'}\n")
            f.write("-" * 80 + "\n")

            for result in results:
                success = "Yes" if result['result'] else "No"
                recognized = "Yes" if result['result'] and ("pedro sánchez" in result['result'].lower() or "presidente" in result['result'].lower()) else "No"
                f.write(f"{result['model']:<20} {result['time']:<12.2f} {success:<10} {recognized}\n")

            f.write(f"\n{'='*80}\n")
            f.write("Test completed successfully!\n")

    finally:
        # Clean up downloaded image
        try:
            os.unlink(image_path)
            print(f"Cleaned up image: {image_path}")
        except:
            pass

def main():
    """Main test function."""
    import sys

    # Test image and content
    image_url = "https://pbs.twimg.com/card_img/1973243448871284736/bPeZHL3l?format=jpg&name=small"
    text_content = "A juicio el hermano, la mujer, el fiscal y dos secretarios de organización del presidente. Si esto pasa en Francia, Alemania o Portugal, leeríamos que tal o cual Gobierno ha caído y nos parecería lo más normal del mundo. Los raros somos nosotros."

    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--ollama-single" and len(sys.argv) > 2:
            # Test single Ollama model (keeping for reference)
            model_name = sys.argv[2]
            test_single_model(model_name, image_url, text_content)
        elif sys.argv[1] == "--gemini":
            # Test Gemini (default approach)
            test_gemini_image_analysis(image_url, text_content)
        elif sys.argv[1] == "--all-models":
            # Test all vision models and compare performance
            test_all_vision_models(image_url, text_content, None)
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python quick_test_image_analysis.py --gemini     # Test Gemini (recommended)")
            print("  python quick_test_image_analysis.py --all-models # Test all vision models")
            print("  python quick_test_image_analysis.py --ollama-single <model>  # Test single Ollama model")
            print("  python quick_test_image_analysis.py --help       # Show this help")
            print("\nGemini is recommended for image analysis (similar to video approach)")
        else:
            print("Invalid arguments. Use --help for usage information.")
    else:
        # Default: Test Gemini
        print("=== Gemini Image Analysis Test (Similar to Video Approach) ===\n")
        print("Testing Gemini for Spanish political content recognition\n")
        test_gemini_image_analysis(image_url, text_content)
        print("\n=== Test Completed ===")

if __name__ == "__main__":
    main()