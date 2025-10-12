"""
Query building functionality for evidence retrieval.
Generates optimized search queries from verification targets.
"""

import re
from typing import List, Dict, Optional, Set
from urllib.parse import quote_plus

from .claim_extractor import VerificationTarget, ClaimType


class QueryBuilder:
    """
    Builds optimized search queries for different types of claims.
    """

    def __init__(self):
        # Stopwords to filter out
        self.stopwords = {
            'que', 'de', 'la', 'el', 'y', 'a', 'en', 'un', 'ser', 'se', 'no', 'haber',
            'por', 'con', 'su', 'para', 'como', 'más', 'o', 'pero', 'sus', 'le', 'ya',
            'o', 'este', 'esta', 'son', 'entre', 'cuando', 'todo', 'nos', 'ya', 'los',
            'las', 'del', 'al', 'lo', 'ha', 'es', 'son', 'era', 'eran', 'fue', 'fueron'
        }

        # Query templates for different claim types
        self.query_templates = {
            ClaimType.NUMERICAL: [
                '"{value}" {context}',  # Exact value with context
                '{context} datos estadísticas',  # Statistical context
                '{value} verificación comprobación',  # Verification terms
            ],
            ClaimType.TEMPORAL: [
                '"{value}" {context}',  # Exact date with context
                '{context} fecha verificación',  # Date verification
                '{value} acontecimiento histórico',  # Historical context
            ],
            ClaimType.ATTRIBUTION: [
                '{context}',  # Attribution context
                'afirmación verificación {context}',  # Verification context
                'dicho por fuente oficial',  # Official source
            ],
            ClaimType.CAUSAL: [
                '{context}',  # Causal relationship
                'relación causa efecto {context}',  # Cause-effect
                'evidencia científica {context}',  # Scientific evidence
            ]
        }

    def build_queries_for_targets(self, targets: List[VerificationTarget],
                                max_queries_per_target: int = 3) -> List[str]:
        """
        Build search queries for verification targets.

        Args:
            targets: List of verification targets
            max_queries_per_target: Maximum queries per target

        Returns:
            List of search query strings
        """
        queries = []

        for target in targets:
            target_queries = self._build_queries_for_target(target, max_queries_per_target)
            queries.extend(target_queries)

        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query in queries:
            if query not in seen:
                seen.add(query)
                unique_queries.append(query)

        return unique_queries

    def _build_queries_for_target(self, target: VerificationTarget,
                                max_queries: int) -> List[str]:
        """Build queries for a single verification target."""
        queries = []

        # Get appropriate templates for this claim type
        templates = self.query_templates.get(target.claim_type, ['{context}'])

        # Extract clean context terms
        context_terms = self._extract_context_terms(target.context)

        # Build queries using templates
        for template in templates[:max_queries]:
            try:
                if '{value}' in template and target.extracted_value:
                    query = template.format(
                        value=target.extracted_value,
                        context=' '.join(context_terms[:5])  # Limit context terms
                    )
                else:
                    query = template.format(context=' '.join(context_terms[:5]))

                # Clean and validate query
                query = self._clean_query(query)
                if query and len(query.split()) >= 2:  # At least 2 words
                    queries.append(query)

            except (KeyError, ValueError):
                continue

        return queries

    def _extract_context_terms(self, context: str) -> List[str]:
        """Extract meaningful terms from context text."""
        # Remove URLs and special characters
        context = re.sub(r'https?://\S+', ' ', context)
        context = re.sub(r'[^\w\s]', ' ', context)

        # Tokenize and filter
        tokens = re.findall(r'\w{3,}', context.lower())  # Words of 3+ characters

        # Remove stopwords and duplicates
        meaningful_terms = []
        seen = set()

        for token in tokens:
            if token not in self.stopwords and token not in seen:
                meaningful_terms.append(token)
                seen.add(token)

        return meaningful_terms

    def _clean_query(self, query: str) -> str:
        """Clean and optimize search query."""
        # Remove extra whitespace
        query = ' '.join(query.split())

        # Remove very short queries
        if len(query) < 5:
            return ''

        # Limit query length
        words = query.split()
        if len(words) > 10:
            words = words[:10]

        return ' '.join(words)

    def build_fact_checking_queries(self, text: str, max_queries: int = 5) -> List[str]:
        """
        Build queries specifically for fact-checking the given text.
        This is a fallback method when no specific claims are extracted.
        """
        # Extract key terms using frequency analysis
        terms = self._extract_key_terms(text)

        queries = []

        # Build various query combinations
        if len(terms) >= 2:
            # Two-term combinations
            for i in range(len(terms)):
                for j in range(i + 1, len(terms)):
                    query = f'"{terms[i]}" "{terms[j]}" verificación'
                    queries.append(query)

        # Add general fact-checking queries
        if terms:
            queries.append(f'{" ".join(terms[:3])} desmentido')
            queries.append(f'{" ".join(terms[:3])} comprobación')
            queries.append(f'{" ".join(terms[:3])} evidencia')

        # Clean and deduplicate
        clean_queries = []
        seen = set()
        for query in queries[:max_queries]:
            clean_query = self._clean_query(query)
            if clean_query and clean_query not in seen:
                clean_queries.append(clean_query)
                seen.add(clean_query)

        return clean_queries

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms using frequency analysis."""
        # Remove URLs and clean text
        text = re.sub(r'https?://\S+', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)

        # Tokenize
        tokens = re.findall(r'\w{4,}', text.lower())

        # Count frequencies
        freq = {}
        for token in tokens:
            if token not in self.stopwords:
                freq[token] = freq.get(token, 0) + 1

        # Sort by frequency and return top terms
        sorted_terms = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        return [term for term, _ in sorted_terms[:8]]  # Top 8 terms


def build_search_queries(targets: List[VerificationTarget],
                        max_queries_per_target: int = 3) -> List[str]:
    """
    Convenience function to build search queries from verification targets.

    Args:
        targets: List of verification targets
        max_queries_per_target: Maximum queries per target

    Returns:
        List of search query strings
    """
    builder = QueryBuilder()
    return builder.build_queries_for_targets(targets, max_queries_per_target)