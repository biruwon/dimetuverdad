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
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def search_web(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """Perform actual web search using DuckDuckGo (no API key required)."""
    try:
        # Use DuckDuckGo search (doesn't require API key)
        search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(search_url, headers=headers, timeout=5)  # Reduced timeout
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
                        'snippet': snippet[:200] if snippet else ""  # Reduced snippet length
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

        response = requests.get(url, headers=headers, timeout=5)  # Reduced timeout
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

        # Clean up content - reduced length for faster processing
        content = re.sub(r'\s+', ' ', content).strip()
        content = content[:1500] if len(content) > 1500 else content  # Reduced from 2000 to 1500

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
   Contenido: {page_data['page_content']['content'][:500]}...
"""

        # Step 4: Use the configured model to analyze the real evidence
        client = OpenAI(
            base_url=model_config["base_url"],
            api_key=model_config["api_key"]
        )

        # Simplified and more focused prompt for faster processing
        analysis_prompt = f"""Analiza esta afirmación usando la evidencia web real:

AFIRMACIÓN: {claim}

EVIDENCIA:
{evidence_text}

Responde en español con formato simple:
- **Resumen**: Hallazgos clave
- **Fuentes**: Credibilidad de las fuentes
- **Veredicto**: VERDADERO/FALSO/ENGANOSO/NO VERIFICADO
- **Confianza**: 0-100%
- **Evidencia clave**: 2-3 puntos principales"""

        response = client.chat.completions.create(
            model=model_config["model"],
            messages=[
                {
                    "role": "system",
                    "content": "Eres un verificador de hechos. Analiza basado en evidencia real. Responde conciso en español."
                },
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ],
            temperature=0.1,  # Reduced from 0.3 for faster generation
            max_tokens=1500  # Reduced from 2500 to 1500
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

def fact_check_with_model_comparison(claim: str, models_to_test: List[str] = None) -> Dict[str, Any]:
    """
    Compare multiple models for fact-checking speed and quality.

    Args:
        claim: The claim to fact-check
        models_to_test: List of model names to test (default: fast models)

    Returns:
        dict: Comparison results with timing and analysis for each model
    """
    if models_to_test is None:
        models_to_test = ["llama3.1:8b", "gpt-oss:20b"]  # Default comparison

    start_time = time.time()

    try:
        print(f"🔍 Researching claim: {claim[:60]}...")
        print("🌐 Performing actual web search and browsing...")

        # Step 1: Search the web for the claim (shared across all models)
        search_start = time.time()
        search_results = search_web(claim, num_results=3)
        search_time = time.time() - search_start
        print(f"🔍 Found {len(search_results)} search results (took {search_time:.1f}s)")

        # Step 2: Browse the top results (shared across all models)
        browse_start = time.time()
        browsed_pages = []
        for i, result in enumerate(search_results[:2]):
            print(f"   • Browsing: {result['url'][:60]}...")
            page_content = browse_website(result['url'])
            if page_content['status'] == 'success':
                browsed_pages.append({
                    'search_result': result,
                    'page_content': page_content
                })
            time.sleep(0.3)  # Reduced delay

        browse_time = time.time() - browse_start
        print(f"📄 Successfully browsed {len(browsed_pages)} pages (took {browse_time:.1f}s)")

        # Step 3: Test each model
        model_results = {}
        for model_name in models_to_test:
            print(f"\n🎯 Testing {model_name}...")

            model_config = {
                "model": model_name,
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
                "provider": "ollama"
            }

            analysis_start = time.time()
            result = fact_check_with_model(claim, model_config)
            analysis_time = time.time() - analysis_start

            model_results[model_name] = {
                'result': result,
                'analysis_time': analysis_time,
                'success': result.get('success', False)
            }

            status = "✅" if result.get('success', False) else "❌"
            print(f"{status} {model_name}: {analysis_time:.1f}s")

        total_time = time.time() - start_time

        return {
            "claim": claim,
            "search_results": search_results,
            "browsed_pages": browsed_pages,
            "model_results": model_results,
            "shared_timing": {
                'search_time': search_time,
                'browse_time': browse_time,
                'total_shared_time': search_time + browse_time
            },
            "total_time": total_time,
            "success": any(r['success'] for r in model_results.values())
        }

    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ Error after {total_time:.1f}s: {e}")
        return {
            "claim": claim,
            "error": str(e),
            "success": False,
            "total_time": total_time
        }

# Global connection pool for web requests
session_lock = threading.Lock()
_shared_session = None

def get_shared_session():
    """Get or create a shared requests session for connection reuse."""
    global _shared_session
    if _shared_session is None:
        with session_lock:
            if _shared_session is None:
                _shared_session = requests.Session()
                # Configure connection pooling
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=10,
                    pool_maxsize=20,
                    max_retries=3,
                    pool_block=False
                )
                _shared_session.mount('http://', adapter)
                _shared_session.mount('https://', adapter)
    return _shared_session

def search_web_optimized(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """Optimized web search with connection reuse."""
    try:
        session = get_shared_session()
        search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = session.get(search_url, headers=headers, timeout=5)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        for result in soup.find_all('div', class_='result')[:num_results]:
            title_elem = result.find('a', class_='result__a')
            snippet_elem = result.find('a', class_='result__snippet')

            if title_elem:
                title = title_elem.get_text().strip()
                url = title_elem.get('href')

                if url and url.startswith('//duckduckgo.com/l/?uddg='):
                    url = url.split('uddg=')[1].split('&')[0]

                snippet = ""
                if snippet_elem:
                    snippet = snippet_elem.get_text().strip()

                if url and title:
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet[:200] if snippet else ""
                    })

        return results

    except Exception as e:
        print(f"Search error: {e}")
        return []

def browse_website_optimized(url: str) -> Dict[str, str]:
    """Optimized website browsing with connection reuse."""
    try:
        session = get_shared_session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = session.get(url, headers=headers, timeout=5)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        title = ""
        title_elem = soup.find('title')
        if title_elem:
            title = title_elem.get_text().strip()

        content = ""
        for selector in ['article', '.content', '.post-content', '.entry-content', 'main', '.article-body']:
            content_elem = soup.select_one(selector)
            if content_elem:
                content = content_elem.get_text(separator=' ', strip=True)
                break

        if not content:
            body = soup.find('body')
            if body:
                content = body.get_text(separator=' ', strip=True)

        content = re.sub(r'\s+', ' ', content).strip()
        content = content[:1500] if len(content) > 1500 else content

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

def process_single_claim_optimized(claim: str, model_name: str = "llama3.1:8b") -> Dict[str, Any]:
    """Process a single claim with optimized settings for batch processing."""
    start_time = time.time()

    try:
        # Step 1: Search the web (optimized)
        search_results = search_web_optimized(claim, num_results=3)

        # Step 2: Browse top results (optimized)
        browsed_pages = []
        for result in search_results[:2]:
            page_content = browse_website_optimized(result['url'])
            if page_content['status'] == 'success':
                browsed_pages.append({
                    'search_result': result,
                    'page_content': page_content
                })
            time.sleep(0.2)  # Reduced delay for batch processing

        # Step 3: Prepare evidence
        evidence_text = f"AFIRMACIÓN: {claim}\n\nEVIDENCIA:\n"
        for i, result in enumerate(search_results, 1):
            evidence_text += f"{i}. {result['title']} - {result['snippet'][:100]}...\n"

        for i, page_data in enumerate(browsed_pages, 1):
            evidence_text += f"\nPágina {i}: {page_data['page_content']['content'][:400]}..."

        # Step 4: Model analysis (ultra-fast settings for batch)
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

        prompt = f"Verifica: {claim}\n\nEvidencia:\n{evidence_text}\n\nResponde conciso: VERDADERO/FALSO/ENGANOSO + confianza 0-100 + explicación breve."

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Verificador de hechos conciso en español."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Deterministic for batch processing
            max_tokens=800  # Much shorter for speed
        )

        analysis = response.choices[0].message.content
        total_time = time.time() - start_time

        return {
            "claim": claim,
            "analysis": analysis,
            "model_used": model_name,
            "total_time": total_time,
            "success": True
        }

    except Exception as e:
        total_time = time.time() - start_time
        return {
            "claim": claim,
            "error": str(e),
            "total_time": total_time,
            "success": False
        }

def process_claims_batch_parallel(claims: List[str], model_name: str = "llama3.1:8b",
                                max_workers: int = 3) -> List[Dict[str, Any]]:
    """Process multiple claims in parallel for maximum speed."""
    print(f"🚀 Processing {len(claims)} claims in parallel (max {max_workers} workers)...")

    results = []
    total_start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all claims for processing
        future_to_claim = {
            executor.submit(process_single_claim_optimized, claim, model_name): claim
            for claim in claims
        }

        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_claim):
            claim = future_to_claim[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                status = "✅" if result['success'] else "❌"
                print(f"   {status} [{completed}/{len(claims)}] {claim[:50]}... ({result['total_time']:.1f}s)")
            except Exception as e:
                results.append({
                    "claim": claim,
                    "error": str(e),
                    "success": False,
                    "total_time": 0
                })
                completed += 1
                print(f"   ❌ [{completed}/{len(claims)}] {claim[:50]}... (ERROR)")

    total_time = time.time() - total_start
    print(f"🎯 Batch completed in {total_time:.1f}s")
    print(f"📈 Average time per claim: {total_time/len(claims):.1f}s")

    return results

def main():
    """Main function to fact-check claims using GPT-OSS:20B with real web browsing."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Sistema de verificación de hechos con GPT-OSS:20B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python factcheck.py                                    # Ejecuta ejemplos de prueba
  python factcheck.py "La vacuna COVID contiene microchips"  # Verifica una afirmación específica
  python factcheck.py --file claims.txt                 # Lee afirmaciones desde archivo
  python factcheck.py --compare "La luna es habitable"  # Compara modelos (llama3.1:8b vs gpt-oss:20b)
  python factcheck.py --compare --models llama3.1:8b phi3:3.8b "El sol es una estrella"  # Modelos específicos
  python factcheck.py --batch --file claims.txt         # Procesamiento batch ultra-rápido
  python factcheck.py --batch --workers 5 --fast-model llama3.1:8b --file claims.txt  # Batch personalizado
        """
    )

    parser.add_argument(
        'claim',
        nargs='?',
        help='La afirmación a verificar (entre comillas si contiene espacios)'
    )

    parser.add_argument(
        '--file', '-f',
        help='Archivo con afirmaciones a verificar (una por línea)'
    )

    parser.add_argument(
        '--compare', '-c',
        action='store_true',
        help='Comparar múltiples modelos para velocidad y calidad'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=['llama3.1:8b', 'gpt-oss:20b'],
        help='Modelos a comparar (default: llama3.1:8b gpt-oss:20b)'
    )

    parser.add_argument(
        '--batch', '-b',
        action='store_true',
        help='Procesar en modo batch paralelo para máxima velocidad'
    )

    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=3,
        help='Número máximo de workers para procesamiento paralelo (default: 3)'
    )

    parser.add_argument(
        '--fast-model',
        default='llama3.1:8b',
        help='Modelo rápido para modo batch (default: llama3.1:8b)'
    )

    args = parser.parse_args()

    # Determine what claims to process
    claims_to_check = []

    if args.file:
        # Read claims from file
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                claims_to_check = [line.strip() for line in f if line.strip()]
            print(f"📂 Cargadas {len(claims_to_check)} afirmaciones desde {args.file}")
        except FileNotFoundError:
            print(f"❌ Error: No se encontró el archivo {args.file}")
            return
        except Exception as e:
            print(f"❌ Error al leer el archivo: {e}")
            return
    elif args.claim:
        # Single claim from command line
        claims_to_check = [args.claim]
    else:
        # No arguments provided, use test examples
        claims_to_check = [
            "España saldrá de la Unión Europea en 2025",
            "La vacuna COVID-19 contiene microchips para rastrear personas",
            "El cambio climático es completamente causado por la actividad humana",
            "Los coches eléctricos son más caros de mantener que los de gasolina",
            "La Gran Muralla China es visible desde el espacio a simple vista"
        ]
        print("🧪 Ejecutando ejemplos de prueba...")

    print("🎯 SISTEMA DE VERIFICACIÓN DE HECHOS: COMPARACIÓN DE MODELOS")
    print("=" * 70)
    print("Usando navegación web real - MODELOS LOCALES GRATUITOS")
    print("=" * 70)
    if args.batch:
        print(f"🚀 MODO BATCH: Procesamiento paralelo con {args.workers} workers")
        print(f"🎯 Modelo rápido: {args.fast_model}")
    elif args.compare:
        print(f"🔬 Comparando modelos: {', '.join(args.models)}")
    else:
        print("🎯 GPT-OSS:20B: Modelo local (Ollama)")
    print("=" * 70)

    total_start_time = time.time()
    timing_summary = []

    # Handle batch processing mode
    if args.batch and len(claims_to_check) > 1:
        print(f"\n🚀 PROCESAMIENTO BATCH DE {len(claims_to_check)} AFIRMACIONES")
        print("=" * 70)

        batch_results = process_claims_batch_parallel(
            claims_to_check,
            model_name=args.fast_model,
            max_workers=args.workers
        )

        # Convert batch results to timing summary format
        for result in batch_results:
            timing_summary.append({
                'claim': result['claim'][:50] + '...' if len(result['claim']) > 50 else result['claim'],
                'time': result.get('total_time', 0),
                'success': result.get('success', False)
            })

        # Display batch results
        print(f"\n📊 RESULTADOS DEL BATCH:")
        print("=" * 70)

        successful_results = [r for r in batch_results if r['success']]
        failed_results = [r for r in batch_results if not r['success']]

        print(f"✅ Procesadas exitosamente: {len(successful_results)}")
        print(f"❌ Con error: {len(failed_results)}")

        for i, result in enumerate(batch_results, 1):
            status = "✅" if result['success'] else "❌"
            print(f"\n{status} AFIRMACIÓN {i}: {result['claim']}")
            print("-" * 50)

            if result['success']:
                print(f"⏱️  Tiempo: {result['total_time']:.1f}s")
                print(f"🎯 Modelo: {result['model_used']}")
                print(f"\n{result['analysis']}")
            else:
                print(f"❌ Error: {result.get('error', 'Error desconocido')}")

        total_time = time.time() - total_start_time

    else:
        # Original sequential processing
        for i, claim in enumerate(claims_to_check, 1):
            claim_start_time = time.time()
            print(f"\n📋 AFIRMACIÓN {i}: {claim}")
            print("-" * 60)

            try:
                if args.compare:
                    # Model comparison mode
                    result = fact_check_with_model_comparison(claim, args.models)
                else:
                    # Single model mode (default GPT-OSS:20B)
                    result = fact_check_with_model_comparison(claim, ["gpt-oss:20b"])

                if result["success"]:
                    print("✅ ANÁLISIS COMPLETADO")
                    print(f"🔍 Fuentes encontradas: {result.get('sources_found', 0)}")
                    print(f"📄 Páginas visitadas: {result.get('pages_browsed', 0)}")

                    # Display timing information
                    if args.compare:
                        shared_timing = result.get('shared_timing', {})
                        print(f"⏱️  Tiempos compartidos: Búsqueda {shared_timing.get('search_time', 0):.1f}s | "
                              f"Navegación {shared_timing.get('browse_time', 0):.1f}s")
                        print(f"⏱️  Tiempo total: {result.get('total_time', 0):.1f}s")

                        # Display model comparison results
                        model_results = result.get('model_results', {})
                        print(f"\n🔬 COMPARACIÓN DE MODELOS:")
                        for model_name, model_data in model_results.items():
                            status = "✅" if model_data['success'] else "❌"
                            analysis_time = model_data['analysis_time']
                            print(f"   {status} {model_name}: {analysis_time:.1f}s")
                    else:
                        timing = result.get('timing', {})
                        print(f"⏱️  Tiempos: Búsqueda {timing.get('search_time', 0):.1f}s | "
                              f"Navegación {timing.get('browse_time', 0):.1f}s | "
                              f"Análisis {timing.get('analysis_time', 0):.1f}s | "
                              f"Total {timing.get('total_time', 0):.1f}s")

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

                    # Display analysis results
                    if args.compare:
                        model_results = result.get('model_results', {})
                        for model_name, model_data in model_results.items():
                            model_result = model_data['result']
                            print(f"\n{'='*35} 🎯 {model_name.upper()} {'='*35}")
                            if model_result.get("success") and model_result.get("analysis"):
                                analysis_text = model_result.get("analysis", "")
                                if len(analysis_text.strip()) > 10:
                                    print(analysis_text)
                                else:
                                    print(f"⚠️ Análisis generado pero muy corto: '{analysis_text}'")
                            elif model_result.get("error"):
                                print(f"❌ Error en análisis: {model_result.get('error')}")
                            else:
                                print(f"❌ Error: No se pudo obtener el análisis de {model_name}")
                    else:
                        # Single model display (GPT-OSS:20B)
                        print(f"\n{'='*35} 🎯 GPT-OSS:20B {'='*35}")
                        if result.get("success") and result.get("analysis"):
                            analysis_text = result.get("analysis", "")
                            if len(analysis_text.strip()) > 10:
                                print(analysis_text)
                            else:
                                print(f"⚠️ Análisis generado pero muy corto: '{analysis_text}'")
                        elif result.get("error"):
                            print(f"❌ Error en análisis: {result.get('error')}")
                        else:
                            print("❌ Error: No se pudo obtener el análisis de GPT-OSS:20B")
                            print(f"Debug - result keys: {list(result.keys())}")
                            print(f"Debug - analysis length: {len(result.get('analysis', ''))}")

                else:
                    print(f"❌ ERROR: {result.get('error', 'Error desconocido')}")

                # Store timing for summary
                claim_time = time.time() - claim_start_time
                timing_summary.append({
                    'claim': claim[:50] + '...' if len(claim) > 50 else claim,
                    'time': claim_time,
                    'success': result.get('success', False)
                })

            except Exception as e:
                claim_time = time.time() - claim_start_time
                print(f"❌ ERROR: {e}")
                timing_summary.append({
                    'claim': claim[:50] + '...' if len(claim) > 50 else claim,
                    'time': claim_time,
                    'success': False
                })

            print("\n" + "=" * 70)

        # Display timing summary
        total_time = time.time() - total_start_time
    print("\n📊 RESUMEN DE TIEMPOS:")
    print("=" * 70)

    successful_claims = [t for t in timing_summary if t['success']]
    failed_claims = [t for t in timing_summary if not t['success']]

    print(f"✅ Afirmaciones procesadas exitosamente: {len(successful_claims)}")
    print(f"❌ Afirmaciones con error: {len(failed_claims)}")
    print(f"⏱️  Tiempo total: {total_time:.1f} segundos")
    print(f"📈 Tiempo promedio por afirmación: {total_time/len(timing_summary):.1f} segundos")

    if successful_claims:
        avg_success_time = sum(t['time'] for t in successful_claims) / len(successful_claims)
        print(f"📈 Tiempo promedio (solo exitosas): {avg_success_time:.1f} segundos")

    print("\n📋 DETALLE POR AFIRMACIÓN:")
    for i, timing in enumerate(timing_summary, 1):
        status = "✅" if timing['success'] else "❌"
        print(f"   {i}. {status} {timing['claim']} - {timing['time']:.1f}s")

    print("\n🎯 RESUMEN DEL SISTEMA:")
    print("• Usa navegación web real sin costos de API")
    print("• Modelo GPT-OSS:20B local y gratuito")
    print("• Análisis estructurado en español")
    print("• Fuentes web auténticas y verificables")
    print("• Modo batch paralelo para procesamiento masivo")
    print("• Optimizaciones aplicadas: menos resultados de búsqueda, menos páginas navegadas, timeouts reducidos")
    print(f"• Modelo rápido por defecto: llama3.1:8b (37.9s vs 44.4s de gpt-oss:20b)")

if __name__ == "__main__":
    main()
