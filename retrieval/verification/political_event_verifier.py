"""
Political Event Verifier - Specialized verification for political events like arrests, trials, etc.
Integrates with fact-checking sites and official sources to verify political claims.
"""

import asyncio
import aiohttp
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from database import get_db_connection
from ..core.models import VerificationResult, VerificationVerdict, EvidenceSource
from .claim_verifier import VerificationContext


@dataclass
class PoliticalEventClaim:
    """Represents a political event claim to be verified."""
    person_name: str
    event_type: str  # 'arrest', 'trial', 'conviction', 'resignation', etc.
    institution: Optional[str] = None  # 'prisión', 'tribunal', 'gobierno', etc.
    date_mentioned: Optional[datetime] = None
    context: str = ""


@dataclass
class VerificationEvidence:
    """Evidence found during verification."""
    source_name: str
    source_url: str
    verdict: VerificationVerdict
    confidence: float
    explanation: str
    publish_date: Optional[datetime] = None


class PoliticalEventVerifier:
    """
    Specialized verifier for political events (arrests, trials, judicial proceedings, etc.).
    
    Searches multiple sources:
    - Official government press releases
    - Court records and judicial databases
    - Reputable news sources (El País, El Mundo, ABC, etc.)
    - Fact-checking sites (Maldita.es, Newtral.es)
    - Official political party statements
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize the political event verifier.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

        # Fact-checking and news sources
        self.fact_checking_sources = [
            {
                'name': 'Maldita.es',
                'search_url': 'https://maldita.es/buscar/',
                'base_url': 'https://maldita.es'
            },
            {
                'name': 'Newtral',
                'search_url': 'https://www.newtral.es/buscar/',
                'base_url': 'https://www.newtral.es'
            },
            {
                'name': 'El País',
                'search_url': 'https://elpais.com/buscador/',
                'base_url': 'https://elpais.com'
            },
            {
                'name': 'El Mundo',
                'search_url': 'https://www.elmundo.es/buscar.html',
                'base_url': 'https://www.elmundo.es'
            }
        ]

        # Official sources
        self.official_sources = [
            'boe.es',  # Official State Gazette
            'congreso.es',  # Congress of Deputies
            'senado.es',  # Senate
            'agpd.es',  # Data Protection Agency
            'tribunalconstitucional.es',  # Constitutional Court
        ]

    async def verify_arrest_claim(self, person_name: str, institution: Optional[str] = None) -> VerificationResult:
        """
        Verify if a person was actually arrested or imprisoned.
        
        Args:
            person_name: Name of the person claimed to be arrested
            institution: Institution mentioned (e.g., 'Soto del Real', 'prisión')
            
        Returns:
            VerificationResult with evidence and verdict
        """
        self.logger.info(f"Verifying arrest claim for {person_name}")

        claim = PoliticalEventClaim(
            person_name=person_name,
            event_type='arrest',
            institution=institution
        )

        # Search across multiple sources
        evidence_list = await self._search_multiple_sources(claim)

        # Analyze evidence to determine overall verdict
        return self._analyze_evidence(claim, evidence_list)

    async def verify_judicial_event(self, claim_text: str) -> VerificationResult:
        """
        Verify judicial events like trials, convictions, court orders.
        
        Args:
            claim_text: Text describing the judicial event
            
        Returns:
            VerificationResult with evidence and verdict
        """
        self.logger.info(f"Verifying judicial event: {claim_text[:100]}...")

        # Extract person and event type from claim
        person_name, event_type = self._extract_judicial_claim_info(claim_text)

        claim = PoliticalEventClaim(
            person_name=person_name,
            event_type=event_type,
            context=claim_text
        )

        evidence_list = await self._search_multiple_sources(claim)
        return self._analyze_evidence(claim, evidence_list)

    async def verify_political_event(self, event_description: str) -> VerificationResult:
        """
        General method to verify any political event claim.
        
        Args:
            event_description: Description of the political event
            
        Returns:
            VerificationResult with evidence and verdict
        """
        self.logger.info(f"Verifying political event: {event_description[:100]}...")

        # Parse the event description
        claim = self._parse_event_description(event_description)

        evidence_list = await self._search_multiple_sources(claim)
        return self._analyze_evidence(claim, evidence_list)

    async def _search_multiple_sources(self, claim: PoliticalEventClaim) -> List[VerificationEvidence]:
        """
        Search across multiple sources for evidence about the claim.
        Uses parallel execution to search all sources simultaneously.
        
        Args:
            claim: The claim to verify
            
        Returns:
            List of evidence found
        """
        evidence_list = []

        # Create tasks for parallel execution
        tasks = []
        
        # Add fact-checking site searches
        for source in self.fact_checking_sources:
            tasks.append(self._search_fact_checking_site_safe(claim, source))
        
        # Add official sources search
        tasks.append(self._search_official_sources_safe(claim))
        
        # Add recent news search
        tasks.append(self._search_recent_news_safe(claim))
        
        # Execute all searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect all evidence from successful searches
        for result in results:
            if isinstance(result, list) and result:
                evidence_list.extend(result)
            elif isinstance(result, Exception):
                self.logger.warning(f"Search task failed: {result}")
        
        return evidence_list
    
    async def _search_fact_checking_site_safe(self, claim: PoliticalEventClaim, source: Dict[str, str]) -> List[VerificationEvidence]:
        """
        Wrapper for _search_fact_checking_site that handles exceptions.
        
        Args:
            claim: The claim to verify
            source: Source configuration
            
        Returns:
            List of evidence found (empty list on error)
        """
        try:
            return await self._search_fact_checking_site(claim, source)
        except Exception as e:
            self.logger.warning(f"Error searching {source['name']}: {e}")
            return []
    
    async def _search_official_sources_safe(self, claim: PoliticalEventClaim) -> List[VerificationEvidence]:
        """
        Wrapper for _search_official_sources that handles exceptions.
        
        Args:
            claim: The claim to verify
            
        Returns:
            List of evidence found (empty list on error)
        """
        try:
            return await self._search_official_sources(claim)
        except Exception as e:
            self.logger.warning(f"Error searching official sources: {e}")
            return []
    
    async def _search_recent_news_safe(self, claim: PoliticalEventClaim) -> List[VerificationEvidence]:
        """
        Wrapper for _search_recent_news that handles exceptions.
        
        Args:
            claim: The claim to verify
            
        Returns:
            List of evidence found (empty list on error)
        """
        try:
            return await self._search_recent_news(claim)
        except Exception as e:
            self.logger.warning(f"Error searching recent news: {e}")
            return []

    async def _search_fact_checking_site(self, claim: PoliticalEventClaim, source: Dict[str, str]) -> List[VerificationEvidence]:
        """
        Search a fact-checking site for the claim.
        
        Args:
            claim: Claim to search for
            source: Source configuration
            
        Returns:
            List of evidence found
        """
        evidence_list = []

        # Construct search query
        search_query = f'"{claim.person_name}" {claim.event_type}'
        if claim.institution:
            search_query += f' "{claim.institution}"'

        # URL encode the search query
        from urllib.parse import quote
        encoded_query = quote(search_query)
        search_url = f"{source['search_url']}{encoded_query}"

        # Perform actual web search
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            try:
                async with session.get(search_url) as response:
                    if response.status == 200:
                        html = await response.text()

                        # Analyze the search results
                        if self._contains_contradiction_indicators(html, claim):
                            evidence_list.append(VerificationEvidence(
                                source_name=source['name'],
                                source_url=search_url,
                                verdict=VerificationVerdict.DEBUNKED,
                                confidence=0.8,
                                explanation=f"{source['name']} indica que la afirmación sobre {claim.person_name} es falsa o engañosa.",
                                publish_date=datetime.now()
                            ))
                        elif self._contains_confirmation_indicators(html, claim):
                            evidence_list.append(VerificationEvidence(
                                source_name=source['name'],
                                source_url=search_url,
                                verdict=VerificationVerdict.VERIFIED,
                                confidence=0.7,
                                explanation=f"{source['name']} confirma que el evento con {claim.person_name} ocurrió.",
                                publish_date=datetime.now()
                            ))
                        else:
                            # No clear indication found
                            evidence_list.append(VerificationEvidence(
                                source_name=source['name'],
                                source_url=search_url,
                                verdict=VerificationVerdict.QUESTIONABLE,
                                confidence=0.4,
                                explanation=f"{source['name']} no encontró información concluyente sobre la afirmación.",
                                publish_date=datetime.now()
                            ))

            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout searching {source['name']}")
            except Exception as e:
                self.logger.warning(f"Error searching {source['name']}: {e}")

        return evidence_list

    async def _search_official_sources(self, claim: PoliticalEventClaim) -> List[VerificationEvidence]:
        """
        Search official government and judicial sources.
        
        Args:
            claim: Claim to verify
            
        Returns:
            List of evidence found
        """
        evidence_list = []

        # Define official sources to check
        official_sources = [
            {
                'name': 'BOE (Boletín Oficial del Estado)',
                'search_url': 'https://www.boe.es/buscar.php?campo%5B0%5D=DEM&dato%5B0%5D=',
                'type': 'judicial'
            },
            {
                'name': 'Congreso de los Diputados',
                'search_url': 'https://www.congreso.es/busqueda-de-diputados?p_p_id=diputadobusqueda_WAR_diputadobusqueda_INSTANCE_5H9i&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view&p_p_col_id=column-1&p_p_col_count=1&_diputadobusqueda_WAR_diputadobusqueda_INSTANCE_5H9i_action=search&_diputadobusqueda_WAR_diputadobusqueda_INSTANCE_5H9i_apellido=',
                'type': 'parliamentary'
            },
            {
                'name': 'Senado',
                'search_url': 'https://www.senado.es/web/relacionesciudadanos/senadores/senadores/index.html',
                'type': 'parliamentary'
            }
        ]

        # Search each official source
        for source in official_sources:
            source_evidence = await self._search_official_source(claim, source)
            evidence_list.extend(source_evidence)

        return evidence_list

    async def _search_official_source(self, claim: PoliticalEventClaim, source: Dict[str, str]) -> List[VerificationEvidence]:
        """
        Search a specific official source for the claim.
        
        Args:
            claim: Claim to search for
            source: Source configuration
            
        Returns:
            List of evidence found
        """
        evidence_list = []

        # Construct search query based on source type
        if source['type'] == 'judicial':
            # For judicial sources like BOE, search for legal proceedings
            search_query = f'"{claim.person_name}"'
            if claim.event_type in ['arrest', 'prisión', 'condena']:
                search_query += ' resolución judicial'
        elif source['type'] == 'parliamentary':
            # For parliamentary sources, search for member status
            search_query = f'"{claim.person_name}"'
        else:
            search_query = f'"{claim.person_name}" {claim.event_type}'

        # URL encode the search query
        from urllib.parse import quote
        encoded_query = quote(search_query)
        search_url = f"{source['search_url']}{encoded_query}"

        # Perform actual web search
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            try:
                async with session.get(search_url) as response:
                    if response.status == 200:
                        html = await response.text()

                        # Analyze the results
                        if self._contains_contradiction_indicators(html, claim):
                            evidence_list.append(VerificationEvidence(
                                source_name=source['name'],
                                source_url=search_url,
                                verdict=VerificationVerdict.DEBUNKED,
                                confidence=0.9,
                                explanation=f"{source['name']} no registra ningún {claim.event_type} de {claim.person_name}.",
                                publish_date=datetime.now()
                            ))
                        elif self._contains_confirmation_indicators(html, claim):
                            evidence_list.append(VerificationEvidence(
                                source_name=source['name'],
                                source_url=search_url,
                                verdict=VerificationVerdict.VERIFIED,
                                confidence=0.95,
                                explanation=f"{source['name']} confirma el {claim.event_type} de {claim.person_name}.",
                                publish_date=datetime.now()
                            ))
                        else:
                            # No clear information found - for official sources, this often means no record
                            if claim.event_type in ['arrest', 'prisión', 'condena']:
                                evidence_list.append(VerificationEvidence(
                                    source_name=source['name'],
                                    source_url=search_url,
                                    verdict=VerificationVerdict.DEBUNKED,
                                    confidence=0.7,
                                    explanation=f"{source['name']} no tiene registro de {claim.event_type} para {claim.person_name}.",
                                    publish_date=datetime.now()
                                ))
                            else:
                                evidence_list.append(VerificationEvidence(
                                    source_name=source['name'],
                                    source_url=search_url,
                                    verdict=VerificationVerdict.QUESTIONABLE,
                                    confidence=0.3,
                                    explanation=f"{source['name']} no proporciona información concluyente sobre {claim.person_name}.",
                                    publish_date=datetime.now()
                                ))

            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout searching {source['name']}")
            except Exception as e:
                self.logger.warning(f"Error searching {source['name']}: {e}")

        return evidence_list

    async def _search_recent_news(self, claim: PoliticalEventClaim) -> List[VerificationEvidence]:
        """
        Search recent news for confirmation or contradiction.
        
        Args:
            claim: Claim to verify
            
        Returns:
            List of evidence found
        """
        evidence_list = []

        # Define news sources to check
        news_sources = [
            {
                'name': 'El País',
                'search_url': 'https://elpais.com/buscador/?q='
            },
            {
                'name': 'El Mundo',
                'search_url': 'https://www.elmundo.es/buscar.html?query='
            },
            {
                'name': 'ABC',
                'search_url': 'https://www.abc.es/buscar?q='
            },
            {
                'name': 'Google News',
                'search_url': 'https://news.google.com/search?q='
            }
        ]

        # Search each news source
        for source in news_sources:
            source_evidence = await self._search_news_source(claim, source)
            evidence_list.extend(source_evidence)

        return evidence_list

    async def _search_news_source(self, claim: PoliticalEventClaim, source: Dict[str, str]) -> List[VerificationEvidence]:
        """
        Search a specific news source for the claim.
        
        Args:
            claim: Claim to search for
            source: Source configuration
            
        Returns:
            List of evidence found
        """
        evidence_list = []

        # Construct search query
        search_query = f'"{claim.person_name}" {claim.event_type}'
        if claim.institution:
            search_query += f' "{claim.institution}"'

        # URL encode the search query
        from urllib.parse import quote
        encoded_query = quote(search_query)
        search_url = f"{source['search_url']}{encoded_query}"

        # Perform actual web search
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            try:
                async with session.get(search_url) as response:
                    if response.status == 200:
                        html = await response.text()

                        # Analyze the search results
                        if self._contains_contradiction_indicators(html, claim):
                            evidence_list.append(VerificationEvidence(
                                source_name=source['name'],
                                source_url=search_url,
                                verdict=VerificationVerdict.DEBUNKED,
                                confidence=0.8,
                                explanation=f"{source['name']} indica que la afirmación sobre {claim.person_name} es falsa o no confirmada.",
                                publish_date=datetime.now()
                            ))
                        elif self._contains_confirmation_indicators(html, claim):
                            evidence_list.append(VerificationEvidence(
                                source_name=source['name'],
                                source_url=search_url,
                                verdict=VerificationVerdict.VERIFIED,
                                confidence=0.7,
                                explanation=f"{source['name']} confirma que el evento con {claim.person_name} ocurrió.",
                                publish_date=datetime.now()
                            ))
                        else:
                            # No clear indication found - for major political events, lack of coverage suggests it's false
                            if claim.event_type in ['arrest', 'prisión', 'condena']:
                                evidence_list.append(VerificationEvidence(
                                    source_name=source['name'],
                                    source_url=search_url,
                                    verdict=VerificationVerdict.DEBUNKED,
                                    confidence=0.6,
                                    explanation=f"{source['name']} no reporta ningún {claim.event_type} de {claim.person_name}. Un evento de esta magnitud tendría cobertura inmediata.",
                                    publish_date=datetime.now()
                                ))
                            else:
                                evidence_list.append(VerificationEvidence(
                                    source_name=source['name'],
                                    source_url=search_url,
                                    verdict=VerificationVerdict.QUESTIONABLE,
                                    confidence=0.4,
                                    explanation=f"{source['name']} no encontró información concluyente sobre la afirmación.",
                                    publish_date=datetime.now()
                                ))

            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout searching {source['name']}")
            except Exception as e:
                self.logger.warning(f"Error searching {source['name']}: {e}")

        return evidence_list

    def _analyze_evidence(self, claim: PoliticalEventClaim, evidence_list: List[VerificationEvidence]) -> VerificationResult:
        """
        Analyze collected evidence to determine overall verdict.
        
        Args:
            claim: Original claim
            evidence_list: Evidence collected
            
        Returns:
            Overall verification result
        """
        if not evidence_list:
            return VerificationResult(
                claim=claim.context,
                verdict=VerificationVerdict.QUESTIONABLE,
                confidence=0.3,
                evidence_sources=[],
                explanation="No evidence found for this claim",
                claim_type="political_event",
                extracted_value=claim.person_name
            )

        # Count verdicts
        verdicts = {}
        total_confidence = 0

        for evidence in evidence_list:
            verdict_key = evidence.verdict.value
            if verdict_key not in verdicts:
                verdicts[verdict_key] = []
            verdicts[verdict_key].append(evidence)
            total_confidence += evidence.confidence

        # Determine overall verdict
        if VerificationVerdict.DEBUNKED.value in verdicts:
            overall_verdict = VerificationVerdict.DEBUNKED
            confidence = min(0.9, total_confidence / len(evidence_list))
            contradictions = [f"Desmentido por {ev.source_name}: {ev.explanation}" for ev in verdicts[VerificationVerdict.DEBUNKED.value]]
        elif VerificationVerdict.VERIFIED.value in verdicts:
            overall_verdict = VerificationVerdict.VERIFIED
            confidence = min(0.8, total_confidence / len(evidence_list))
            contradictions = []
        else:
            overall_verdict = VerificationVerdict.QUESTIONABLE
            confidence = 0.4
            contradictions = []

        # Convert evidence to EvidenceSource objects
        evidence_sources = [
            EvidenceSource(
                source_name=ev.source_name,
                source_type='fact_checker',
                url=ev.source_url,
                title=f"Verificación: {claim.event_type} de {claim.person_name}",
                credibility_score=0.8,  # Default credibility for known sources
                content_snippet=ev.explanation,
                verdict_contribution=ev.verdict,
                confidence=ev.confidence
            )
            for ev in evidence_list
        ]

        return VerificationResult(
            claim=claim.context,
            verdict=overall_verdict,
            confidence=confidence,
            evidence_sources=evidence_sources,
            explanation=f"Political event verification result: {overall_verdict.value}",
            claim_type="political_event",
            extracted_value=claim.person_name
        )

    def _contains_contradiction_indicators(self, html: str, claim: PoliticalEventClaim) -> bool:
        """
        Check if HTML contains indicators that the claim is false.
        
        Args:
            html: HTML content to analyze
            claim: Claim being checked
            
        Returns:
            True if contradiction indicators found
        """
        html_lower = html.lower()

        contradiction_indicators = [
            'falso', 'false', 'mentira', 'desmentido', 'desmiente',
            'no es cierto', 'sin fundamento', 'bulo', 'fake news',
            'no hay registro', 'no consta', 'no se ha producido'
        ]

        person_in_content = claim.person_name.lower() in html_lower
        event_in_content = claim.event_type.lower() in html_lower

        if person_in_content and event_in_content:
            return any(indicator in html_lower for indicator in contradiction_indicators)

        return False

    def _contains_confirmation_indicators(self, html: str, claim: PoliticalEventClaim) -> bool:
        """
        Check if HTML contains confirmation of the claim.
        
        Args:
            html: HTML content to analyze
            claim: Claim being checked
            
        Returns:
            True if confirmation indicators found
        """
        html_lower = html.lower()

        confirmation_indicators = [
            'confirmado', 'confirma', 'ha sido', 'se ha producido',
            'según fuentes', 'oficialmente', 'anuncia'
        ]

        person_in_content = claim.person_name.lower() in html_lower
        event_in_content = claim.event_type.lower() in html_lower

        if person_in_content and event_in_content:
            return any(indicator in html_lower for indicator in confirmation_indicators)

        return False

    def _extract_judicial_claim_info(self, claim_text: str) -> Tuple[str, str]:
        """
        Extract person name and event type from judicial claim text.
        
        Args:
            claim_text: Text containing judicial claim
            
        Returns:
            Tuple of (person_name, event_type)
        """
        # Simple extraction - in production would use NLP
        text_lower = claim_text.lower()

        # Common political figures
        figures = ['ábalos', 'sánchez', 'iglesias', 'montero', 'casado', 'abascal', 'rivera']

        person_name = "desconocido"
        for figure in figures:
            if figure in text_lower:
                person_name = figure
                break

        # Determine event type
        if 'juez' in text_lower or 'jueza' in text_lower:
            event_type = 'judicial_order'
        elif 'procesado' in text_lower or 'procesamiento' in text_lower:
            event_type = 'prosecution'
        elif 'condenado' in text_lower or 'condena' in text_lower:
            event_type = 'conviction'
        else:
            event_type = 'judicial_event'

        return person_name, event_type

    def _parse_event_description(self, description: str) -> PoliticalEventClaim:
        """
        Parse a political event description into a PoliticalEventClaim.
        
        Args:
            description: Event description text
            
        Returns:
            Parsed claim object
        """
        # Simple parsing - in production would use NLP
        text_lower = description.lower()

        # Extract person (simplified)
        person_patterns = [
            r'([A-ZÁ-Ú][a-zá-úñ]+(?:\s+[A-ZÁ-Ú][a-zá-úñ]+)*)',
        ]

        person_name = "desconocido"
        for pattern in person_patterns:
            matches = re.findall(pattern, description)
            if matches:
                person_name = matches[0]
                break

        # Determine event type
        if 'detenido' in text_lower or 'arrestado' in text_lower:
            event_type = 'arrest'
        elif 'procesado' in text_lower:
            event_type = 'prosecution'
        elif 'condenado' in text_lower:
            event_type = 'conviction'
        elif 'dimitido' in text_lower or 'dimisión' in text_lower:
            event_type = 'resignation'
        else:
            event_type = 'political_event'

        return PoliticalEventClaim(
            person_name=person_name,
            event_type=event_type,
            context=description
        )