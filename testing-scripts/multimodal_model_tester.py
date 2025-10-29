#!/usr/bin/env python3
"""
Simple test: Download a small image and ask each multimodal model to describe it.
"""

import asyncio
import time
import base64
import requests
import ollama
from analyzer.prompts import EnhancedPromptGenerator

async def test_model_simple(model_name: str):
    """Test a single model with image + text analysis task."""
    print(f"\n🧪 Testing {model_name}...")

    start_time = time.time()

    try:
        # Use test images from the post
        image_url = "https://pbs.twimg.com/media/G215HUGXIAA0oPI?format=jpg&name=large"
        #image_url_2 = "https://pbs.twimg.com/media/G2-U8hFWYAAVS4B?format=jpg&name=small"
        print(f"📥 Downloading media: {image_url}")
        #print(f"� Downloading media: {image_url_2}")

        # Download first image
        response1 = requests.get(image_url, timeout=10)
        response1.raise_for_status()
        image_data_1 = base64.b64encode(response1.content).decode('utf-8')
        print(f"✅ Downloaded image 1: {len(image_data_1)} bytes")

        # Download second image
        #response2 = requests.get(image_url_2, timeout=10)
        #response2.raise_for_status()
        #image_data_2 = base64.b64encode(response2.content).decode('utf-8')
        #print(f"✅ Downloaded image 2: {len(image_data_2)} bytes")

        # Test the model with both images and text
        print("🤖 Asking model to analyze images + text...")

        # Post text is included directly in the prompt above

        # Use standardized Gemini analysis prompt for image content
        prompt = f"""
TEXTO DEL POST:
"A la Barbie Gaza la cazan yéndose a Ibiza con maletas de Louis Vuitton (más de 1.000 euros) con el dinero que ha ganado esta semana en las televisiones corruptas con la farsa de la flotilla antisemita. Gracias a Dios que Trump le he cortado el chollo con la paz en Palestina. Esta gente solo usa los conflictos y dramas para hacer caja. Así nos va."

CONTEXTO:
Publicación en red social con texto e imágenes. Analizar la combinación visual y textual para identificar mensajes problemáticos, sesgo político, desinformación o propaganda.

INSTRUCCIONES DE ANÁLISIS:
1. Examina tanto el texto como los elementos visuales (imágenes) para identificar discurso de odio, sesgo político, extremismo o manipulación mediática.  
2. Observa símbolos, figuras públicas, memes, banderas, o lenguaje cargado que indique ideología extremista o far-right.  
3. Evalúa si se presentan afirmaciones falsas, información fuera de contexto o narrativas conspirativas.  
4. Determina la categoría más apropiada según la lista del sistema.  
5. Proporciona una breve explicación que indique los elementos clave que justifican la categoría.

FORMATO DE RESPUESTA:
CATEGORÍA: [categoría elegida]  
EXPLICACIÓN: [razonamiento breve y neutral]
"""
        
        # Add system prompt for Ollama
        system_prompt = """
🔍 SISTEMA DE MODERACIÓN DE CONTENIDO — DETECCIÓN DE SESGO POLÍTICO Y DESINFORMACIÓN

PROPÓSITO:
Eres un analista automático de publicaciones en redes sociales en español. 
Tu función es identificar y clasificar contenido que pueda contener discurso de odio, desinformación, propaganda o sesgo político extremo, con énfasis en ideología de extrema derecha y manipulación mediática. 
No generes ni reproduzcas contenido dañino. Resume y analiza de forma neutral y objetiva.

TAREA PRINCIPAL:
Analiza texto, imágenes y videos de publicaciones y clasifica el contenido en una sola categoría de la lista a continuación. Proporciona una explicación breve y objetiva que indique los elementos clave (símbolos, tono, texto, referencias visuales).

CATEGORÍAS DISPONIBLES:
- hate_speech: ataques directos o degradación de grupos por etnia, religión, orientación sexual, género o nacionalidad.
- anti_immigration: rechazo explícito o simbólico hacia inmigrantes o minorías.
- anti_lgbtq: ridiculización o negación de derechos de personas LGBTQ+.
- anti_feminism: oposición al feminismo o promoción de roles de género tradicionales.
- nationalism: exaltación nacionalista o símbolos patrios con carga política o de superioridad nacional.
- anti_government: cuestionamiento extremo o burla hacia instituciones o líderes gubernamentales.
- disinformation: afirmaciones falsas, manipuladas o fuera de contexto que distorsionan la realidad.
- conspiracy_theory: narrativas de élites ocultas, manipulación global o complots.
- call_to_action: exhortaciones explícitas a actuar o movilizarse políticamente.
- political_general: contenido político sin sesgo extremo.
- historical_revisionism: reinterpretación falsa de hechos históricos.
- general: sin contenido problemático, neutral.

DIRECTRICES DE RESPUESTA:
1. Evalúa todo el contenido disponible: texto, imágenes y videos.  
2. Selecciona la categoría que mejor describa el mensaje global de la publicación.  
3. Escribe una explicación breve (2–4 oraciones) destacando los elementos clave que sustentan la decisión.  
4. Mantén tono neutral, objetivo y analítico.  
5. Si no hay señales de contenido problemático, responde “general”.

FORMATO DE RESPUESTA:
CATEGORÍA: [una sola categoría de la lista]  
EXPLICACIÓN: [2–4 oraciones en español, neutrales, descriptivas]

"""
        client = ollama.AsyncClient()

        response = await asyncio.wait_for(
            client.generate(
                model=model_name,
                system=system_prompt,
                prompt=prompt,
                images=[image_data_1],
                options = {
                    "temperature": 0.2,   # deterministic
                    "top_p": 0.7,         # focused token distribution
                    "num_predict": 250,   # short explanation, fewer tokens
                },
                keep_alive="10m",           # keep model in memory for repeated calls
            ),
            timeout=120.0
        )

        end_time = time.time()
        timing = end_time - start_time

        analysis = response.get('response', '').strip()

        print("✅ Success!")
        print(f"⏱️  Time: {timing:.2f}s")
        print("📝 Full Analysis:")
        print("-" * 50)
        print(analysis)
        print("-" * 50)

    except asyncio.TimeoutError:
        end_time = time.time()
        timing = end_time - start_time
        print(f"❌ TIMEOUT: Model took {timing:.2f}s (limit: 60s)")

    except Exception as e:
        end_time = time.time()
        timing = end_time - start_time
        print(f"❌ Error: {e} (took {timing:.2f}s)")

async def main():
    """Test all multimodal models with image + text analysis."""
    models_to_test = [
        "gemma3:4b",
        "gemma3:12b",
        "gemma3:27b-it-qat",
    ]

    print("🚀 Multimodal Model Test: Image + Text Analysis")
    print("📷 Using test image from X post")
    print("📝 Analyzing with Spanish political post text")
    print("❓ Asking for detailed analysis of image + text context")
    print("⏰ 60-second timeout per model")

    for model in models_to_test:
        await test_model_simple(model)

    print("\n✅ All analysis tests completed!")

if __name__ == "__main__":
    asyncio.run(main())