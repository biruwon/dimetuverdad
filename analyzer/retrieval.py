import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import re
import time
from typing import List, Dict
import warnings
from bs4 import XMLParsedAsHTMLWarning

# Suppress XMLParsedAsHTMLWarning for RSS parsing
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Minimal Spanish stopwords list (small and conservative)
SPANISH_STOPWORDS = {
    'que','de','la','el','y','a','en','un','ser','se','no','haber','por','con','su','para','como',
    'más','o','pero','sus','le','ya','o','este','esta','son','entre','cuando','todo','nos','ya',
}

# Curated sources with a simple search URL template (use {q} for query)
SOURCE_TEMPLATES = {
    'google_news_rss': {
        'label': 'Google News (RSS)',
        'template': 'https://news.google.com/rss/search?q={q}&hl=es-ES&gl=ES&ceid=ES:es'
    },
    'bing_news_rss': {
        'label': 'Bing News (RSS)',
        'template': 'https://www.bing.com/news/search?q={q}&format=rss'
    },
    'maldita': {
        'label': 'Maldita.es',
        'template': 'https://www.maldita.es/buscador/?s={q}'
    },
    'newtral': {
        'label': 'Newtral',
        'template': 'https://www.newtral.es/?s={q}'
    },
    'snopes': {
        'label': 'Snopes',
        'template': 'https://www.snopes.com/?s={q}'
    },
    'politifact': {
        'label': 'PolitiFact',
        'template': 'https://www.politifact.com/search/?q={q}'
    },
    'factcheck': {
        'label': 'FactCheck.org',
        'template': 'https://www.factcheck.org/?s={q}'
    }
    ,
    'elpais': {
        'label': 'El País (buscar)'
        , 'template': 'https://elpais.com/buscar/?q={q}'
    },
    'elmundo': {
        'label': 'El Mundo (buscar)'
        , 'template': 'https://www.elmundo.es/buscador.html?query={q}'
    },
    'abc': {
        'label': 'ABC (buscar)'
        , 'template': 'https://sevilla.abc.es/buscar/?q={q}'
    },
    'lavanguardia': {
        'label': 'La Vanguardia (buscar)'
        , 'template': 'https://www.lavanguardia.com/buscar?q={q}'
    }
}

# Sites considered high-trust for debunking in Spain / international
FACT_CHECKERS = {'maldita', 'newtral', 'factcheck', 'politifact', 'snopes'}

# Keywords that often indicate a debunking/correction
DEBUNK_KEYWORDS = [
    'falso', 'falsedad', 'desmiente', 'desmentido', 'no es cierto', 'mentira', 'error', 'corrección', 'aclara',
    'verifica', 'verificado', 'comprobado', 'desmentir', 'desmentido', 'fraude'
]


def extract_query_terms(text: str, max_terms: int = 6) -> str:
    """Create a compact query from the post by picking the most common non-stopword tokens."""
    text = re.sub(r"https?://\S+", " ", text)
    tokens = re.findall(r"\w{4,}", text.lower(), flags=re.UNICODE)
    tokens = [t for t in tokens if t not in SPANISH_STOPWORDS]
    if not tokens:
        return text[:200]
    # frequency
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    chosen = [t for t, _ in sorted_tokens[:max_terms]]
    return " ".join(chosen)


def build_search_urls(query: str, sources: List[str] = None) -> List[Dict[str,str]]:
    urls = []
    q = quote_plus(query)
    keys = sources if sources else list(SOURCE_TEMPLATES.keys())
    for k in keys:
        if k in SOURCE_TEMPLATES:
            tpl = SOURCE_TEMPLATES[k]['template']
            urls.append({'source': k, 'label': SOURCE_TEMPLATES[k]['label'], 'url': tpl.format(q=q)})
    return urls


def fetch_links_from_search_url(url: str, max_links: int = 3, timeout: int = 6) -> List[Dict[str,str]]:
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; EvidenceFetcher/1.0; +https://example.org)'
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        content_type = r.headers.get('content-type', '')
        links = []
        # If the response looks like RSS/XML, parse items
        if 'xml' in content_type or url.lower().endswith('.rss') or '<rss' in r.text[:200].lower():
            soup = BeautifulSoup(r.text, 'xml')
            items = soup.find_all('item')[:max_links]
            for it in items:
                link = it.find('link')
                title = it.find('title')
                links.append({'title': title.text if title else '', 'url': link.text if link else ''})
        else:
            # otherwise try to parse HTML anchors
            soup = BeautifulSoup(r.text, 'html.parser')
            anchors = soup.find_all('a', href=True)
            seen = set()
            for a in anchors:
                href = a['href']
                # normalize relative links and skip empty
                if href.startswith('#') or href.lower().startswith('javascript:'):
                    continue
                title = (a.get_text() or '').strip()
                # handle common relative links by ignoring them
                if href.startswith('/'):
                    continue
                if href in seen:
                    continue
                seen.add(href)
                links.append({'title': title, 'url': href})
                if len(links) >= max_links:
                    break

        # After collecting links, perform a lightweight debunk check on each
        enriched = []
        for it in links:
            verdict = 'unclear'
            snippet = ''
            try:
                r2 = requests.get(it['url'], headers=headers, timeout=timeout)
                r2.raise_for_status()
                # Try to detect content type and use appropriate parser
                content_type = r2.headers.get('content-type', '').lower()
                if 'xml' in content_type or 'rss' in content_type or '<rss' in r2.text[:200].lower():
                    # Use XML parser for XML content
                    soup = BeautifulSoup(r2.text, 'xml')
                    text = soup.get_text(separator=' ', strip=True).lower()
                else:
                    # Use HTML parser for HTML content
                    soup = BeautifulSoup(r2.text, 'html.parser')
                    text = soup.get_text(separator=' ', strip=True).lower()
                snippet = text[:300]
                for k in DEBUNK_KEYWORDS:
                    if k in text:
                        verdict = 'debunk'
                        break
            except Exception:
                # ignore per-link errors
                pass
            enriched.append({'title': it.get('title', ''), 'url': it.get('url', ''), 'verdict': verdict, 'snippet': snippet})
            if len(enriched) >= max_links:
                break
        return enriched
    except Exception:
        # On any error just return empty so caller can fallback to search URL
        return []


def retrieve_evidence_for_post(text: str, max_per_source: int = 2, sources: List[str] = None) -> List[Dict[str,str]]:
    """Return a list of evidence links gathered from curated sources for the given post.

    Each item is a dict: {source, label, search_url, results: [{title, url}, ...]}
    """
    q = extract_query_terms(text)
    # By default restrict retrieval strictly to high-trust fact-checkers and Spanish mainstream newspaper search pages
    # If caller provides explicit sources, use them; otherwise use a whitelist only
    mainstream = [k for k in ['elpais', 'elmundo', 'abc', 'lavanguardia'] if k in SOURCE_TEMPLATES]
    trusted_keys = list(FACT_CHECKERS) + mainstream
    urls = build_search_urls(q, sources=(sources if sources is not None else trusted_keys))
    results = []
    for u in urls:
        time.sleep(0.2)
        found = fetch_links_from_search_url(u['url'], max_links=max_per_source)
        if not found:
            # if nothing found, provide the search URL so user can click
            results.append({'source': u['source'], 'label': u['label'], 'search_url': u['url'], 'results': []})
        else:
            results.append({'source': u['source'], 'label': u['label'], 'search_url': u['url'], 'results': found})
    return results


def format_evidence(results: List[Dict[str,str]]) -> str:
    lines = []
    for r in results:
        header = f"{r.get('label')} -> {r.get('search_url')}\n"
        lines.append(header)
        for it in r.get('results', []):
            lines.append(f" - {it.get('title','').strip()[:120]} | {it.get('url')}")
    return "\n".join(lines)
