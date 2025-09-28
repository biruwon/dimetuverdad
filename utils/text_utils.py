"""Text utilities: Unicode normalization helpers used across the analysis pipeline.

This module centralizes Unicode normalization so we consistently handle
diacritics, combining marks and invisible characters while preserving
emoji and symbol characters.
"""
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize text for safe pattern matching and LLM prompting.

    Steps:
    - If input is falsy, return empty string
    - Normalize to NFKD to decompose combined characters
    - Lowercase using Unicode-aware lower()
    - Recompose to NFC for stable representation

    This preserves emoji and symbol characters while making letter
    comparisons robust across composed/decomposed forms.
    """
    if not text:
        return ""

    # Decompose combined characters (separates accents)
    try:
        decomposed = unicodedata.normalize('NFKD', text)
    except Exception:
        decomposed = text

    # Lowercase with Unicode awareness
    lowered = decomposed.lower()

    # Recompose to NFC for stability
    try:
        recomposed = unicodedata.normalize('NFC', lowered)
    except Exception:
        recomposed = lowered

    return recomposed
