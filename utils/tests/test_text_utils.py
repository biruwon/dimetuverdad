"""Unit tests for text_utils module."""

import unittest
from utils.text_utils import normalize_text


class TestNormalizeText(unittest.TestCase):
    """Test cases for the normalize_text function."""

    def test_normalize_text_empty_string(self):
        """Test normalize_text with empty string."""
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(None), "")

    def test_normalize_text_basic_lowercasing(self):
        """Test basic lowercasing functionality."""
        self.assertEqual(normalize_text("HELLO WORLD"), "hello world")
        self.assertEqual(normalize_text("Hello World"), "hello world")

    def test_normalize_text_unicode_normalization(self):
        """Test Unicode normalization with combining characters."""
        # Test NFC normalization (combining characters)
        self.assertEqual(normalize_text("café"), "café")  # Already NFC
        self.assertEqual(normalize_text("naïve"), "naïve")  # Already NFC

        # Test NFKD decomposition and recomposition
        decomposed = "café"  # This might be decomposed in some inputs
        self.assertEqual(normalize_text(decomposed), "café")

    def test_normalize_text_diacritics_preserved(self):
        """Test that diacritics and accented characters are preserved."""
        spanish_text = "español, México, naïve, résumé"
        normalized = normalize_text(spanish_text)
        self.assertEqual(normalized, "español, méxico, naïve, résumé")

        # Test specific Spanish characters
        self.assertEqual(normalize_text("ESPAÑOL"), "español")
        self.assertEqual(normalize_text("MÉXICO"), "méxico")
        self.assertEqual(normalize_text("ÑANDÚ"), "ñandú")

    def test_normalize_text_emoji_preserved(self):
        """Test that emoji characters are preserved."""
        text_with_emoji = "Hello 😀 world 🌍 test"
        normalized = normalize_text(text_with_emoji)
        self.assertEqual(normalized, "hello 😀 world 🌍 test")

        # Test with uppercase text containing emoji
        self.assertEqual(normalize_text("HELLO 😀 WORLD"), "hello 😀 world")

    def test_normalize_text_symbols_preserved(self):
        """Test that mathematical and other symbols are preserved."""
        text_with_symbols = "Price: €50, ½ + ¼ = ¾, © 2024"
        normalized = normalize_text(text_with_symbols)
        # NFKD normalization decomposes fraction characters
        self.assertEqual(normalized, "price: €50, 1⁄2 + 1⁄4 = 3⁄4, © 2024")

    def test_normalize_text_combining_marks(self):
        """Test handling of combining marks and diacritics."""
        # Test various combining sequences
        self.assertEqual(normalize_text("naïve"), "naïve")
        self.assertEqual(normalize_text("résumé"), "résumé")
        self.assertEqual(normalize_text("Müller"), "müller")

    def test_normalize_text_whitespace_preserved(self):
        """Test that whitespace is preserved."""
        text_with_whitespace = "  hello   world  \t\n  test  "
        normalized = normalize_text(text_with_whitespace)
        self.assertEqual(normalized, "  hello   world  \t\n  test  ")

    def test_normalize_text_mixed_content(self):
        """Test mixed content with text, diacritics, emoji, and symbols."""
        mixed_text = "¡Hola ESPAÑOL! 😀 Cómo estás? €50 ½ + ¼ = ¾ © 2024 naïve résumé"
        expected = "¡hola español! 😀 cómo estás? €50 1⁄2 + 1⁄4 = 3⁄4 © 2024 naïve résumé"
        self.assertEqual(normalize_text(mixed_text), expected)

    def test_normalize_text_spanish_specific(self):
        """Test Spanish-specific text normalization."""
        spanish_content = "ESPAÑOL MÉXICO ESPAÑA FRANCOFONÍA"
        expected = "español méxico españa francofonía"
        self.assertEqual(normalize_text(spanish_content), expected)

    def test_normalize_text_case_preservation_verification(self):
        """Verify that the function correctly handles case conversion."""
        # Test that it's consistently lowercase
        inputs = ["HELLO", "Hello", "hello", "HeLLo WoRLd"]
        for input_text in inputs:
            with self.subTest(input_text=input_text):
                result = normalize_text(input_text)
                self.assertEqual(result, result.lower())


if __name__ == '__main__':
    unittest.main()