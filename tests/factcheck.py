"""
GPT-OSS:20B Fact-Checking System
Uses real web browsing with GPT-OSS:20B model for accurate fact-checking.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from typing import Dict, Any, List
from openai import OpenAI
import time
import re

def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """Perform actual web search using DuckDuckGo (no API key required)."""
    try:
        # Use DuckDuckGo search (doesn't require API key)
        search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        for result in soup.find_all('div', class_='result')[:num_results]:
            title_elem = result.find('a', class_='result__a')
            snippet_elem = result.find('a', class_='result__snippet')

            if title_elem:
                title = title_elem.get_text().strip()
                url = title_elem.get('href')

                # Extract actual URL from DuckDuckGo redirect
                if url and url.startswith('//duckduckgo.com/l/?uddg='):
                    url = url.split('uddg=')[1].split('&')[0]

                snippet = ""
                if snippet_elem:
                    snippet = snippet_elem.get_text().strip()

                if url and title:
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet[:300] if snippet else ""
                    })

        return results

    except Exception as e:
        print(f"Search error: {e}")
        return []

def browse_website(url: str) -> Dict[str, str]:
    """Actually browse a website and extract its content."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title
        title = ""
        title_elem = soup.find('title')
        if title_elem:
            title = title_elem.get_text().strip()

        # Extract main content (try different selectors)
        content = ""
        for selector in ['article', '.content', '.post-content', '.entry-content', 'main', '.article-body']:
            content_elem = soup.select_one(selector)
            if content_elem:
                content = content_elem.get_text(separator=' ', strip=True)
                break

        # Fallback to body text if no specific content found
        if not content:
            body = soup.find('body')
            if body:
                content = body.get_text(separator=' ', strip=True)

        # Clean up content
        content = re.sub(r'\s+', ' ', content).strip()
        content = content[:2000] if len(content) > 2000 else content

        return {
            'url': url,
            'title': title,
            'content': content,
            'status': 'success'
        }

    except Exception as e:
        return {
            'url': url,
            'title': '',
            'content': f"Error browsing website: {str(e)}",
            'status': 'error'
        }

def fact_check_with_model(claim: str, model_config: Dict[str, str]) -> Dict[str, Any]:
    """
    Fact-check a claim using the specified model configuration.

    Args:
        claim: The claim to fact-check
        model_config: Configuration for the LLM model

    Returns:
        dict: Analysis result from the model
    """
    try:
        # Step 1: Search the web for the claim
        search_results = search_web(claim, num_results=5)

        # Step 2: Browse the top results to get actual content
        browsed_pages = []
        for i, result in enumerate(search_results[:3]):  # Browse top 3 results
            page_content = browse_website(result['url'])
            if page_content['status'] == 'success':
                browsed_pages.append({
                    'search_result': result,
                    'page_content': page_content
                })
            time.sleep(1)  # Be respectful to websites

        # Step 3: Prepare evidence for the model
        evidence_text = f"""
AFIRMACIÓN A VERIFICAR: "{claim}"

RESULTADOS DE BÚSQUEDA ENCONTRADOS:
"""
        for i, result in enumerate(search_results, 1):
            evidence_text += f"""
{i}. {result['title']}
   URL: {result['url']}
   Fragmento: {result['snippet']}
"""

        evidence_text += "\n\nCONTENIDO DE PÁGINAS VISITADAS:\n"
        for i, page_data in enumerate(browsed_pages, 1):
            evidence_text += f"""
{i}. {page_data['page_content']['title']}
   URL: {page_data['page_content']['url']}
   Contenido: {page_data['page_content']['content'][:1000]}...
"""

        # Step 4: Use the configured model to analyze the real evidence
        client = OpenAI(
            base_url=model_config["base_url"],
            api_key=model_config["api_key"]
        )

        analysis_prompt = f"""
Eres un experto verificador de hechos. Analiza la siguiente afirmación utilizando los resultados de búsqueda web REALES y el contenido de las páginas visitadas que se proporciona a continuación.

{evidence_text}

Basándote en esta EVIDENCIA REAL de sitios web auténticos, proporciona un análisis completo de verificación de hechos.

FORMATO DE RESPUESTA:
===================

## Resumen de Evidencia
Resume los hallazgos clave de las fuentes web reales.

## Credibilidad de las Fuentes
Evalúa la credibilidad de las fuentes que fueron visitadas realmente.

## Veredicto de Verificación
Conclusión clara: VERDADERO, FALSO, ENGANOSO o NO VERIFICADO basado en la evidencia real.

## Nivel de Confianza
Califica la confianza (0-100%) basada en la calidad y consistencia de las fuentes reales.

## Hallazgos Clave
Lista la evidencia más importante encontrada del contenido web real.

IMPORTANTE: Responde ÚNICAMENTE en español, ya que esta herramienta se utilizará con textos en español.
"""

        response = client.chat.completions.create(
            model=model_config["model"],
            messages=[
                {
                    "role": "system",
                    "content": "Eres un experto verificador de hechos analizando contenido web real. Base tu análisis únicamente en la evidencia real proporcionada de sitios web visitados. Responde en español."
                },
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ],
            temperature=0.3,
            max_tokens=3000
        )

        analysis = response.choices[0].message.content

        return {
            "claim": claim,
            "analysis": analysis,
            "search_results": search_results,
            "browsed_pages": browsed_pages,
            "model_used": model_config["model"],
            "provider": model_config["provider"],
            "research_method": "real_web_browsing",
            "sources_found": len(search_results),
            "pages_browsed": len(browsed_pages),
            "success": True
        }

    except Exception as e:
        return {
            "claim": claim,
            "error": str(e),
            "model_used": model_config["model"],
            "provider": model_config["provider"],
            "research_method": "real_web_browsing",
            "success": False
        }

def fact_check_with_gpt_oss(claim: str) -> Dict[str, Any]:
    """
    Fact-check a claim using GPT-OSS:20B with real web browsing.

    Args:
        claim: The claim to fact-check

    Returns:
        dict: Analysis result from GPT-OSS:20B
    """
    try:
        print(f"🔍 Researching claim: {claim[:60]}...")
        print("🌐 Performing actual web search and browsing...")

        # Step 1: Search the web for the claim
        search_results = search_web(claim, num_results=5)
        print(f"🔍 Found {len(search_results)} search results")

        # Step 2: Browse the top results to get actual content
        browsed_pages = []
        for i, result in enumerate(search_results[:3]):  # Browse top 3 results
            print(f"   • Browsing: {result['url'][:60]}...")
            page_content = browse_website(result['url'])
            if page_content['status'] == 'success':
                browsed_pages.append({
                    'search_result': result,
                    'page_content': page_content
                })
            time.sleep(1)  # Be respectful to websites

        print(f"📄 Successfully browsed {len(browsed_pages)} pages")

        # Step 3: Configure GPT-OSS:20B model
        gpt_oss_config = {
            "model": "gpt-oss:20b",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "provider": "ollama"
        }

        # Step 4: Get analysis from GPT-OSS:20B using real web evidence
        print("\n🎯 Getting analysis from GPT-OSS:20B...")
        result = fact_check_with_model(claim, gpt_oss_config)

        return {
            "claim": claim,
            "search_results": search_results,
            "browsed_pages": browsed_pages,
            "gpt_analysis": result,
            "sources_found": len(search_results),
            "pages_browsed": len(browsed_pages),
            "success": True
        }

    except Exception as e:
        return {
            "claim": claim,
            "error": str(e),
            "success": False
        }

def main():
    """Main function to fact-check claims using GPT-OSS:20B with real web browsing."""

    # Test claims in Spanish
    test_claims = [
        "España saldrá de la Unión Europea en 2025",
        "La vacuna COVID-19 contiene microchips para rastrear personas",
        "El cambio climático es completamente causado por la actividad humana",
        "Los coches eléctricos son más caros de mantener que los de gasolina",
        "La Gran Muralla China es visible desde el espacio a simple vista"
    ]

    print("🎯 SISTEMA DE VERIFICACIÓN DE HECHOS: GPT-OSS:20B")
    print("=" * 70)
    print("Usando navegación web real - MODELO LOCAL GRATUITO")
    print("=" * 70)
    print("🎯 GPT-OSS:20B: Modelo local (Ollama)")
    print("=" * 70)

    for i, claim in enumerate(test_claims, 1):
        print(f"\n📋 AFIRMACIÓN {i}: {claim}")
        print("-" * 60)

        try:
            result = fact_check_with_gpt_oss(claim)

            if result["success"]:
                print("✅ ANÁLISIS COMPLETADO")
                print(f"🔍 Fuentes encontradas: {result.get('sources_found', 0)}")
                print(f"📄 Páginas visitadas: {result.get('pages_browsed', 0)}")

                # Display search results
                search_results = result.get('search_results', [])
                if search_results:
                    print(f"\n🔍 RESULTADOS DE BÚSQUEDA:")
                    for j, res in enumerate(search_results[:3], 1):
                        print(f"   {j}. {res['title'][:60]}...")
                        print(f"      URL: {res['url']}")

                # Display browsed pages
                browsed_pages = result.get('browsed_pages', [])
                if browsed_pages:
                    print(f"\n📄 PÁGINAS VISITADAS:")
                    for j, page_data in enumerate(browsed_pages, 1):
                        page = page_data['page_content']
                        print(f"   {j}. {page['title'][:60]}...")
                        print(f"      URL: {page['url']}")

                # Display GPT-OSS:20B analysis
                gpt_result = result.get('gpt_analysis', {})

                print(f"\n{'='*35} 🎯 GPT-OSS:20B {'='*35}")
                if gpt_result.get("success"):
                    print(gpt_result.get("analysis", "Error en análisis"))
                else:
                    print(f"❌ Error: {gpt_result.get('error', 'Error desconocido')}")

            else:
                print(f"❌ ERROR: {result.get('error', 'Error desconocido')}")

        except Exception as e:
            print(f"❌ ERROR: {e}")

        print("\n" + "=" * 70)

    print("\n🎯 RESUMEN DEL SISTEMA:")
    print("• Usa navegación web real sin costos de API")
    print("• Modelo GPT-OSS:20B local y gratuito")
    print("• Análisis estructurado en español")
    print("• Fuentes web auténticas y verificables")

if __name__ == "__main__":
    main()
