#!/usr/bin/env python3
"""
Test script for Gemini multimodal analysis of videos and text.
Tests the Google GenAI client library with video upload and analysis.
"""

import os
import tempfile
import requests
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Simple in-memory cache for uploaded files (in production, use persistent storage)
uploaded_files_cache = {}

def download_video_to_temp_file(video_url):
    """Download video from URL to a temporary file and return the file path."""
    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()

        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        with open(temp_file.name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded video to: {temp_file.name}")
        return temp_file.name

    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

def analyze_video_with_gemini(video_url, text_content=""):
    """Analyze video using Gemini 2.5 Flash with multimodal capabilities."""

    # Get API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables")
        return None

    try:
        # Initialize the client
        client = genai.Client(api_key=api_key)

        # Check if video is already uploaded (simple URL-based cache)
        if video_url in uploaded_files_cache:
            video_file = uploaded_files_cache[video_url]
            print(f"Reusing previously uploaded video file: {video_file.name}")
        else:
            # Download video to temporary file
            video_file_path = download_video_to_temp_file(video_url)
            if not video_file_path:
                return None

            try:
                # Upload the video file to Gemini
                print("Uploading video to Gemini...")
                video_file = client.files.upload(file=video_file_path)

                # Wait for processing to complete
                print("Waiting for video processing...")
                while video_file.state.name == "PROCESSING":
                    import time
                    time.sleep(2)
                    video_file = client.files.get(name=video_file.name)

                if video_file.state.name == "FAILED":
                    print(f"Video processing failed: {video_file.state.name}")
                    return None

                print("Video uploaded and processed successfully")

                # Cache the uploaded file for future reuse
                uploaded_files_cache[video_url] = video_file

            finally:
                # Clean up temporary file
                try:
                    os.unlink(video_file_path)
                    print(f"Cleaned up temporary file: {video_file_path}")
                except:
                    pass

        # Create the prompt for analysis in Spanish - focused on Twitter posts and far-right content
        prompt = f"""Analiza este video y el texto adjunto que pertenecen al MISMO POST en una cuenta de Twitter/X para detectar contenido político de extrema derecha o desinformación.

TEXTO DEL POST: "{text_content}"

El contenido está en español y proviene de redes sociales. Evalúa si el post promueve o contiene:

CONTEXTO DEL POST:
- El texto y video son parte del mismo tweet/publicación en Twitter
- El texto proporciona contexto adicional al contenido visual del video

ANÁLISIS REQUERIDO:
1. Resumen del contenido visual del video
2. Análisis del texto adjunto para detectar:
   - Discurso político de extrema derecha
   - Teorías conspirativas
   - Llamados a la acción política
   - Ataques a instituciones democráticas
   - Desinformación o fake news
   - Retórica nacionalista o anti-inmigración
3. Evaluación de la relación entre texto y video
4. Clasificación por categorías: hate_speech, disinformation, conspiracy_theory, far_right_bias, call_to_action, general
5. Nivel de credibilidad y sesgo político detectado

IMPORTANTE: Responde completamente en español y sé específico sobre el contenido político español."""

        # Generate content with both video and text
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[video_file, prompt]
        )

        return response.text

    except Exception as e:
        print(f"Error in Gemini analysis: {e}")
        return None

def main():
    """Main test function."""
    print("=== Gemini Multimodal Video Analysis Test ===\n")

    # Test video analysis
    print("Testing video analysis with file caching...")

    # Example video URL (you can replace with actual URLs)
    test_video_url = "https://video.twimg.com/amplify_video/1972307252796141568/vid/avc1/320x568/GftH9VZYZuygizQc.mp4"

    # Example text content that would accompany the video
    test_text = '🔴 #URGENTE | Increpan a la ministra de Igualdad en plena calle por la polémica de las pulseras: "Pedro Sánchez, hijo de puta"\n\nEl pueblo se ha cansado.'

    print(f"Video URL: {test_video_url}")
    print(f"Text content: {test_text[:100]}...")

    # Single analysis of the Twitter post (video + text)
    result = analyze_video_with_gemini(test_video_url, test_text)

    if result:
        print("\n=== Twitter Post Analysis Result (Spanish) ===")
        print(result)
        print("\n=== Test Completed Successfully ===")
        print("✅ Video analysis completed with caching enabled for future reuse!")
    else:
        print("Video analysis failed.")

if __name__ == "__main__":
    main()