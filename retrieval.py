import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urljoin
import re
import time
import json
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Comprehensive Spanish stopwords list
SPANISH_STOPWORDS = {
    'que','de','la','el','y','a','en','un','ser','se','no','haber','por','con','su','para','como',
    'más','o','pero','sus','le','ya','o','este','esta','son','entre','cuando','todo','nos','ya',
    'del','al','lo','los','las','una','uno','dos','tres','cuatro','cinco','muy','bien','mal',
    'han','has','hay','fue','era','será','está','son','estar','tener','hacer','poder','decir',
    'ir','ver','dar','saber','querer','cada','tanto','mismo','desde','hasta','donde','porque',
    'mientras','durante','antes','después','dentro','fuera','sobre','bajo','contra','sin','según'
}

class SourceType(Enum):
    FACT_CHECKER = "fact_checker"
    MAINSTREAM_MEDIA = "mainstream_media"
    GOVERNMENT = "government"
    ACADEMIC = "academic"
    INTERNATIONAL = "international"
    SPECIALIZED = "specialized"

class SourceReliability(Enum):
    VERY_HIGH = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    VERY_LOW = 1

@dataclass
class EvidenceResult:
    title: str
    url: str
    snippet: str
    source_name: str
    source_type: SourceType
    reliability: SourceReliability
    verdict: str
    language: str
    date_published: Optional[str] = None
    confidence_score: float = 0.0

# Comprehensive evidence sources with reliability ratings and specializations
EVIDENCE_SOURCES = {
    # Spanish fact-checkers (highest reliability for debunking)
    'maldita': {
        'label': 'Maldita.es',
        'template': 'https://www.maldita.es/buscador/?s={q}',
        'type': SourceType.FACT_CHECKER,
        'reliability': SourceReliability.VERY_HIGH,
        'language': 'es',
        'specialization': ['misinformation', 'politics', 'health', 'technology'],
        'search_prefix': '',
        'debunk_indicators': ['falso', 'fake', 'bulo', 'desmentido', 'verificación']
    },
    'newtral': {
        'label': 'Newtral',
        'template': 'https://www.newtral.es/?s={q}',
        'type': SourceType.FACT_CHECKER,
        'reliability': SourceReliability.VERY_HIGH,
        'language': 'es',
        'specialization': ['politics', 'economics', 'health', 'climate'],
        'search_prefix': '',
        'debunk_indicators': ['falso', 'verificación', 'fact-check', 'comprobado']
    },
    'verificat': {
        'label': 'Verificat.cat',
        'template': 'https://www.verificat.cat/?s={q}',
        'type': SourceType.FACT_CHECKER,
        'reliability': SourceReliability.VERY_HIGH,
        'language': 'es',
        'specialization': ['politics', 'catalonia', 'regional_politics'],
        'search_prefix': '',
        'debunk_indicators': ['fals', 'falso', 'verificació', 'verificación']
    },
    
    # International fact-checkers
    'snopes': {
        'label': 'Snopes',
        'template': 'https://www.snopes.com/?s={q}',
        'type': SourceType.FACT_CHECKER,
        'reliability': SourceReliability.HIGH,
        'language': 'en',
        'specialization': ['international', 'conspiracy_theories', 'urban_legends'],
        'search_prefix': '',
        'debunk_indicators': ['false', 'misleading', 'unproven', 'fact check']
    },
    'factcheck': {
        'label': 'FactCheck.org',
        'template': 'https://www.factcheck.org/?s={q}',
        'type': SourceType.FACT_CHECKER,
        'reliability': SourceReliability.HIGH,
        'language': 'en',
        'specialization': ['us_politics', 'international', 'health', 'science'],
        'search_prefix': '',
        'debunk_indicators': ['false', 'misleading', 'incorrect', 'distorts']
    },
    'politifact': {
        'label': 'PolitiFact',
        'template': 'https://www.politifact.com/search/?q={q}',
        'type': SourceType.FACT_CHECKER,
        'reliability': SourceReliability.HIGH,
        'language': 'en',
        'specialization': ['politics', 'elections', 'policy'],
        'search_prefix': '',
        'debunk_indicators': ['false', 'mostly false', 'pants on fire', 'misleading']
    },
    
    # Spanish mainstream media
    'elpais': {
        'label': 'El País',
        'template': 'https://elpais.com/buscar/?q={q}',
        'type': SourceType.MAINSTREAM_MEDIA,
        'reliability': SourceReliability.HIGH,
        'language': 'es',
        'specialization': ['politics', 'international', 'economics', 'culture'],
        'search_prefix': '',
        'debunk_indicators': []
    },
    'elmundo': {
        'label': 'El Mundo',
        'template': 'https://www.elmundo.es/buscador.html?query={q}',
        'type': SourceType.MAINSTREAM_MEDIA,
        'reliability': SourceReliability.HIGH,
        'language': 'es',
        'specialization': ['politics', 'national', 'sports', 'technology'],
        'search_prefix': '',
        'debunk_indicators': []
    },
    'abc': {
        'label': 'ABC',
        'template': 'https://sevilla.abc.es/buscar/?q={q}',
        'type': SourceType.MAINSTREAM_MEDIA,
        'reliability': SourceReliability.HIGH,
        'language': 'es',
        'specialization': ['politics', 'national', 'international'],
        'search_prefix': '',
        'debunk_indicators': []
    },
    'lavanguardia': {
        'label': 'La Vanguardia',
        'template': 'https://www.lavanguardia.com/buscar?q={q}',
        'type': SourceType.MAINSTREAM_MEDIA,
        'reliability': SourceReliability.HIGH,
        'language': 'es',
        'specialization': ['politics', 'catalonia', 'business', 'culture'],
        'search_prefix': '',
        'debunk_indicators': []
    },
    'eldiario': {
        'label': 'ElDiario.es',
        'template': 'https://www.eldiario.es/buscar/?q={q}',
        'type': SourceType.MAINSTREAM_MEDIA,
        'reliability': SourceReliability.HIGH,
        'language': 'es',
        'specialization': ['politics', 'social_issues', 'progressive'],
        'search_prefix': '',
        'debunk_indicators': []
    },
    'publico': {
        'label': 'Público',
        'template': 'https://www.publico.es/buscar/{q}',
        'type': SourceType.MAINSTREAM_MEDIA,
        'reliability': SourceReliability.MEDIUM,
        'language': 'es',
        'specialization': ['politics', 'social_issues', 'progressive'],
        'search_prefix': '',
        'debunk_indicators': []
    },
    
    # Government and official sources
    'boe': {
        'label': 'BOE - Boletín Oficial del Estado',
        'template': 'https://www.boe.es/buscar/doc.php?q={q}',
        'type': SourceType.GOVERNMENT,
        'reliability': SourceReliability.VERY_HIGH,
        'language': 'es',
        'specialization': ['legal', 'legislation', 'official'],
        'search_prefix': '',
        'debunk_indicators': []
    },
    'ine': {
        'label': 'INE - Instituto Nacional de Estadística',
        'template': 'https://www.ine.es/buscar/?q={q}',
        'type': SourceType.GOVERNMENT,
        'reliability': SourceReliability.VERY_HIGH,
        'language': 'es',
        'specialization': ['statistics', 'demographics', 'economics'],
        'search_prefix': '',
        'debunk_indicators': []
    },
    'sanidad': {
        'label': 'Ministerio de Sanidad',
        'template': 'https://www.sanidad.gob.es/buscar/?q={q}',
        'type': SourceType.GOVERNMENT,
        'reliability': SourceReliability.VERY_HIGH,
        'language': 'es',
        'specialization': ['health', 'covid', 'vaccines', 'medical'],
        'search_prefix': '',
        'debunk_indicators': []
    },
    
    # Academic and research sources
    'csic': {
        'label': 'CSIC - Consejo Superior de Investigaciones Científicas',
        'template': 'https://www.csic.es/buscar?q={q}',
        'type': SourceType.ACADEMIC,
        'reliability': SourceReliability.VERY_HIGH,
        'language': 'es',
        'specialization': ['science', 'research', 'climate', 'technology'],
        'search_prefix': '',
        'debunk_indicators': []
    },
    
    # Specialized sources
    'ecdc': {
        'label': 'ECDC - European Centre for Disease Prevention',
        'template': 'https://www.ecdc.europa.eu/en/search?q={q}',
        'type': SourceType.INTERNATIONAL,
        'reliability': SourceReliability.VERY_HIGH,
        'language': 'en',
        'specialization': ['health', 'epidemiology', 'vaccines'],
        'search_prefix': '',
        'debunk_indicators': []
    },
    'who': {
        'label': 'WHO - World Health Organization',
        'template': 'https://www.who.int/search?query={q}',
        'type': SourceType.INTERNATIONAL,
        'reliability': SourceReliability.VERY_HIGH,
        'language': 'en',
        'specialization': ['health', 'global_health', 'pandemic'],
        'search_prefix': '',
        'debunk_indicators': []
    },
    
    # News aggregators and RSS feeds
    'google_news_es': {
        'label': 'Google News España',
        'template': 'https://news.google.com/rss/search?q={q}&hl=es-ES&gl=ES&ceid=ES:es',
        'type': SourceType.MAINSTREAM_MEDIA,
        'reliability': SourceReliability.MEDIUM,
        'language': 'es',
        'specialization': ['news_aggregation', 'current_events'],
        'search_prefix': '',
        'debunk_indicators': []
    },
    'bing_news': {
        'label': 'Bing News',
        'template': 'https://www.bing.com/news/search?q={q}&format=rss',
        'type': SourceType.MAINSTREAM_MEDIA,
        'reliability': SourceReliability.MEDIUM,
        'language': 'en',
        'specialization': ['news_aggregation', 'international'],
        'search_prefix': '',
        'debunk_indicators': []
    }
}

# Sites considered high-trust for debunking in Spain / international
FACT_CHECKERS = {'maldita', 'newtral', 'factcheck', 'politifact', 'snopes'}

# Keywords that often indicate a debunking/correction
DEBUNK_KEYWORDS = [
    'falso', 'falsedad', 'desmiente', 'desmentido', 'no es cierto', 'mentira', 'error', 'corrección', 'aclara',
    'verifica', 'verificado', 'comprobado', 'desmentir', 'desmentido', 'fraude'
]


# Sites considered high-trust for debunking in Spain / international
FACT_CHECKERS = {'maldita', 'newtral', 'verificat', 'snopes', 'factcheck', 'politifact'}

# Enhanced keywords that indicate debunking/correction with weights
DEBUNK_KEYWORDS = [
    ('falso', 3.0), ('falsedad', 3.0), ('fake', 3.0), ('bulo', 3.5),
    ('desmiente', 2.5), ('desmentido', 2.5), ('no es cierto', 2.5),
    ('mentira', 2.0), ('error', 1.5), ('corrección', 2.0), ('aclara', 1.5),
    ('verifica', 2.0), ('verificado', 2.0), ('verificación', 2.0),
    ('comprobado', 2.0), ('desmentir', 2.5), ('fraude', 2.5),
    ('misleading', 2.0), ('incorrect', 2.0), ('distorts', 2.0),
    ('unproven', 1.5), ('fact check', 2.5), ('pants on fire', 3.0)
]

class EnhancedEvidenceRetriever:
    """Enhanced evidence retrieval system with sophisticated source management."""
    
    def __init__(self):
        self.sources = EVIDENCE_SOURCES
        self.debunk_keywords = dict(DEBUNK_KEYWORDS)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; EvidenceBot/2.0; +https://example.org)'
        })
    
    def get_relevant_sources(self, 
                           query_terms: List[str], 
                           claim_type: str = None,
                           priority_types: List[SourceType] = None) -> List[str]:
        """Select most relevant sources based on query and claim type."""
        if priority_types is None:
            priority_types = [SourceType.FACT_CHECKER, SourceType.GOVERNMENT, SourceType.MAINSTREAM_MEDIA]
        
        relevant_sources = []
        
        # Always include fact-checkers for political content
        fact_checkers = [k for k, v in self.sources.items() 
                        if v['type'] == SourceType.FACT_CHECKER]
        relevant_sources.extend(fact_checkers)
        
        # Add specialized sources based on query content
        query_text = ' '.join(query_terms).lower()
        
        for source_key, source_info in self.sources.items():
            if source_key in relevant_sources:
                continue
                
            # Check if source specializes in relevant topics
            specializations = source_info.get('specialization', [])
            for spec in specializations:
                if any(keyword in query_text for keyword in self._get_topic_keywords(spec)):
                    relevant_sources.append(source_key)
                    break
        
        # Add high-reliability sources for important claims
        high_rel_sources = [k for k, v in self.sources.items() 
                           if v['reliability'] == SourceReliability.VERY_HIGH 
                           and k not in relevant_sources]
        relevant_sources.extend(high_rel_sources[:3])  # Limit to avoid too many sources
        
        # Prioritize based on source type preferences
        def source_priority(source_key):
            source_type = self.sources[source_key]['type']
            try:
                return priority_types.index(source_type)
            except ValueError:
                return len(priority_types)
        
        relevant_sources.sort(key=source_priority)
        return relevant_sources[:8]  # Limit total sources
    
    def _get_topic_keywords(self, topic: str) -> List[str]:
        """Get keywords associated with each topic for matching."""
        topic_keywords = {
            'misinformation': ['bulo', 'fake', 'falso', 'desinformación'],
            'politics': ['gobierno', 'político', 'elecciones', 'partido'],
            'health': ['salud', 'vacuna', 'covid', 'médico', 'sanitario'],
            'technology': ['tecnología', '5g', 'digital', 'internet'],
            'climate': ['clima', 'cambio climático', 'temperatura', 'medio ambiente'],
            'economics': ['economía', 'pib', 'inflación', 'desempleo', 'euro'],
            'legal': ['ley', 'tribunal', 'sentencia', 'constitución'],
            'statistics': ['estadística', 'datos', 'cifras', 'porcentaje'],
            'international': ['internacional', 'europa', 'eeuu', 'mundial'],
            'conspiracy_theories': ['conspiración', 'élite', 'control', 'oculto'],
            'catalonia': ['cataluña', 'catalán', 'barcelona', 'independencia'],
            'vaccines': ['vacuna', 'inmunización', 'dosis', 'efectos'],
            'covid': ['coronavirus', 'covid', 'pandemia', 'confinamiento'],
            'science': ['científico', 'investigación', 'estudio', 'evidencia']
        }
        return topic_keywords.get(topic, [topic])


def extract_query_terms(text: str, max_terms: int = 8, include_entities: bool = True) -> str:
    """Enhanced query extraction with entity recognition and relevance scoring."""
    # Clean and prepare text
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Extract potential entities (capitalized words, acronyms)
    entities = []
    if include_entities:
        # Political entities
        political_entities = re.findall(r'\b(?:PSOE|PP|VOX|Podemos|Ciudadanos|ERC|Junts|PNV)\b', text, re.IGNORECASE)
        entities.extend(political_entities)
        
        # Person names (capitalized sequences)
        person_names = re.findall(r'\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)+\b', text)
        entities.extend(person_names)
        
        # Acronyms
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        entities.extend(acronyms)
    
    # Extract regular tokens
    tokens = re.findall(r"\w{3,}", text.lower(), flags=re.UNICODE)
    tokens = [t for t in tokens if t not in SPANISH_STOPWORDS and len(t) > 2]
    
    if not tokens and not entities:
        return text[:200]
    
    # Score tokens by frequency and importance
    token_scores = {}
    
    # Frequency scoring
    for token in tokens:
        token_scores[token] = token_scores.get(token, 0) + 1
    
    # Boost important terms
    important_terms = [
        'inmigr', 'refugiad', 'ilegal', 'invas', 'sustituc', 'reemplaz',
        'soros', 'globalist', 'élite', 'conspir', 'deep state',
        'vacun', 'covid', 'plandemia', 'microchip', 'control',
        'gobierno', 'sánchez', 'dictadura', 'régimen', 'traidor',
        'falang', 'extrem', 'nazi', 'fascist', 'reconquist'
    ]
    
    for token in token_scores:
        for important in important_terms:
            if important in token:
                token_scores[token] *= 2
                break
    
    # Select best tokens
    sorted_tokens = sorted(token_scores.items(), key=lambda x: (-x[1], x[0]))
    selected_tokens = [token for token, _ in sorted_tokens[:max_terms-len(entities)]]
    
    # Combine entities and tokens
    final_terms = entities + selected_tokens
    return " ".join(final_terms[:max_terms])


def build_search_urls(query: str, sources: List[str] = None) -> List[Dict[str,str]]:
    """Build search URLs for selected sources."""
    urls = []
    q = quote_plus(query)
    keys = sources if sources else list(EVIDENCE_SOURCES.keys())
    
    for k in keys:
        if k in EVIDENCE_SOURCES:
            source_info = EVIDENCE_SOURCES[k]
            tpl = source_info['template']
            urls.append({
                'source': k, 
                'label': source_info['label'], 
                'url': tpl.format(q=q),
                'type': source_info['type'].value,
                'reliability': source_info['reliability'].value,
                'language': source_info['language']
            })
    return urls


def fetch_links_from_search_url(url: str, 
                               source_info: Dict, 
                               max_links: int = 4, 
                               timeout: int = 8) -> List[EvidenceResult]:
    """Enhanced link fetching with better parsing and analysis."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; EvidenceFetcher/2.0; +https://example.org)',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'es-ES,es;q=0.8,en;q=0.6',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive'
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        content_type = r.headers.get('content-type', '')
        
        results = []
        
        # Parse RSS/XML feeds
        if 'xml' in content_type or url.lower().endswith('.rss') or '<rss' in r.text[:200].lower():
            results = _parse_rss_feed(r.text, source_info, max_links)
        else:
            # Parse HTML search results
            results = _parse_html_search_results(r.text, url, source_info, max_links)
        
        # Enhance results with detailed analysis
        enhanced_results = []
        for result in results:
            enhanced = _enhance_result_with_analysis(result, source_info)
            if enhanced:
                enhanced_results.append(enhanced)
        
        return enhanced_results[:max_links]
        
    except Exception as e:
        print(f"Error fetching from {url}: {e}")
        return []


def _parse_rss_feed(xml_content: str, source_info: Dict, max_items: int) -> List[EvidenceResult]:
    """Parse RSS feed content."""
    try:
        soup = BeautifulSoup(xml_content, 'xml')
        items = soup.find_all('item')[:max_items]
        results = []
        
        for item in items:
            link = item.find('link')
            title = item.find('title')
            description = item.find('description')
            pub_date = item.find('pubDate')
            
            if link and title:
                results.append(EvidenceResult(
                    title=title.text.strip() if title else '',
                    url=link.text.strip() if link else '',
                    snippet=description.text.strip()[:300] if description else '',
                    source_name=source_info['label'],
                    source_type=source_info['type'],
                    reliability=source_info['reliability'],
                    verdict='unknown',
                    language=source_info['language'],
                    date_published=pub_date.text.strip() if pub_date else None
                ))
        
        return results
    except Exception as e:
        print(f"Error parsing RSS: {e}")
        return []


def _parse_html_search_results(html_content: str, 
                              search_url: str, 
                              source_info: Dict, 
                              max_results: int) -> List[EvidenceResult]:
    """Parse HTML search results page."""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        results = []
        seen_urls = set()
        
        # Common selectors for search results
        selectors = [
            'a[href*="http"]',  # Generic links
            '.result a', '.search-result a', '.article-title a',  # Common result selectors
            'h2 a', 'h3 a', '.title a',  # Title links
        ]
        
        for selector in selectors:
            links = soup.select(selector)
            for link in links:
                if len(results) >= max_results:
                    break
                    
                href = link.get('href')
                if not href or href in seen_urls:
                    continue
                
                # Clean and validate URL
                if href.startswith('/'):
                    base_url = '/'.join(search_url.split('/')[:3])
                    href = urljoin(base_url, href)
                
                if not href.startswith('http'):
                    continue
                
                # Skip irrelevant links
                if any(skip in href.lower() for skip in ['javascript:', 'mailto:', '#', 'twitter.com', 'facebook.com']):
                    continue
                
                seen_urls.add(href)
                
                title = link.get_text().strip()
                if not title or len(title) < 10:
                    continue
                
                # Try to find snippet/description
                snippet = ''
                parent = link.find_parent()
                if parent:
                    snippet_elem = parent.find_next_sibling() or parent.find('p') or parent.find('.description')
                    if snippet_elem:
                        snippet = snippet_elem.get_text().strip()[:300]
                
                results.append(EvidenceResult(
                    title=title,
                    url=href,
                    snippet=snippet,
                    source_name=source_info['label'],
                    source_type=source_info['type'],
                    reliability=source_info['reliability'],
                    verdict='unknown',
                    language=source_info['language']
                ))
        
        return results
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return []


def _enhance_result_with_analysis(result: EvidenceResult, source_info: Dict) -> Optional[EvidenceResult]:
    """Enhance result with detailed content analysis."""
    try:
        # Fetch the actual article content for analysis
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; ContentAnalyzer/1.0)',
            'Accept': 'text/html,application/xhtml+xml'
        }
        
        response = requests.get(result.url, headers=headers, timeout=6)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract main content
        content_text = ''
        for content_selector in ['article', '.content', '.post-content', '.entry-content', 'main']:
            content_elem = soup.select_one(content_selector)
            if content_elem:
                content_text = content_elem.get_text(separator=' ', strip=True)
                break
        
        if not content_text:
            content_text = soup.get_text(separator=' ', strip=True)
        
        content_text = content_text.lower()
        
        # Analyze for debunking indicators
        verdict = 'unknown'
        confidence = 0.0
        
        if source_info['type'] == SourceType.FACT_CHECKER:
            # Check for explicit fact-checking verdicts
            for keyword, weight in DEBUNK_KEYWORDS:
                if keyword.lower() in content_text:
                    if keyword.lower() in ['falso', 'false', 'fake', 'bulo', 'pants on fire']:
                        verdict = 'false'
                        confidence += weight
                    elif keyword.lower() in ['misleading', 'inexacto']:
                        verdict = 'misleading'
                        confidence += weight * 0.7
                    elif keyword.lower() in ['verificado', 'confirmed', 'true']:
                        verdict = 'true'
                        confidence += weight * 0.8
                    else:
                        verdict = 'debunked'
                        confidence += weight * 0.9
        
        # Update result with enhanced information
        result.snippet = content_text[:400] if len(content_text) > len(result.snippet) else result.snippet
        result.verdict = verdict
        result.confidence_score = min(confidence / 10.0, 1.0)  # Normalize
        
        return result
        
    except Exception as e:
        # Return original result if enhancement fails
        return result


def retrieve_evidence_for_post(text: str, 
                              max_per_source: int = 3, 
                              sources: List[str] = None,
                              claim_type: str = None) -> List[Dict[str,str]]:
    """Enhanced evidence retrieval with intelligent source selection."""
    
    retriever = EnhancedEvidenceRetriever()
    
    # Extract enhanced query terms
    query_terms = extract_query_terms(text, max_terms=8, include_entities=True).split()
    query = ' '.join(query_terms)
    
    if not query.strip():
        return []
    
    # Select relevant sources if not specified
    if sources is None:
        sources = retriever.get_relevant_sources(query_terms, claim_type)
    
    # Build search URLs
    urls = build_search_urls(query, sources)
    
    all_results = []
    
    for url_info in urls:
        time.sleep(0.3)  # Rate limiting
        
        source_key = url_info['source']
        source_info = EVIDENCE_SOURCES[source_key]
        
        evidence_results = fetch_links_from_search_url(
            url_info['url'], 
            source_info, 
            max_links=max_per_source
        )
        
        # Convert to legacy format for compatibility
        legacy_results = []
        for result in evidence_results:
            legacy_results.append({
                'title': result.title,
                'url': result.url,
                'snippet': result.snippet,
                'verdict': result.verdict,
                'confidence': result.confidence_score,
                'source_type': result.source_type.value,
                'reliability': result.reliability.value
            })
        
        all_results.append({
            'source': source_key,
            'label': source_info['label'],
            'search_url': url_info['url'],
            'type': source_info['type'].value,
            'reliability': source_info['reliability'].value,
            'results': legacy_results
        })
    
    return all_results


# Legacy function name for compatibility
def format_evidence(results: List[Dict[str,str]]) -> str:
    lines = []
    for r in results:
        header = f"{r.get('label')} -> {r.get('search_url')}\n"
        lines.append(header)
        for it in r.get('results', []):
            lines.append(f" - {it.get('title','').strip()[:120]} | {it.get('url')}")
    return "\n".join(lines)
