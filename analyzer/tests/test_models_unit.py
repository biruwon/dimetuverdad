"""Unit tests for analyzer/models.py ContentAnalysis dataclass."""

import unittest
from datetime import datetime
from analyzer.models import ContentAnalysis


class TestContentAnalysis(unittest.TestCase):
    """Test cases for ContentAnalysis dataclass."""

    def test_init_basic_fields(self):
        """Test ContentAnalysis initialization with basic required fields."""
        analysis = ContentAnalysis(
            post_id="123",
            post_url="https://example.com/post/123",
            author_username="testuser",
            post_content="Test content",
            analysis_timestamp="2024-01-01T00:00:00Z",
            category="general"
        )

        self.assertEqual(analysis.post_id, "123")
        self.assertEqual(analysis.post_url, "https://example.com/post/123")
        self.assertEqual(analysis.author_username, "testuser")
        self.assertEqual(analysis.post_content, "Test content")
        self.assertEqual(analysis.analysis_timestamp, "2024-01-01T00:00:00Z")
        self.assertEqual(analysis.category, "general")

    def test_init_default_values(self):
        """Test ContentAnalysis initialization sets default values correctly."""
        analysis = ContentAnalysis(
            post_id="123",
            post_url="https://example.com/post/123",
            author_username="testuser",
            post_content="Test content",
            analysis_timestamp="2024-01-01T00:00:00Z",
            category="general"
        )

        # Check default values
        self.assertEqual(analysis.categories_detected, [])
        self.assertEqual(analysis.local_explanation, "")
        self.assertEqual(analysis.analysis_stages, "")  # Empty by default, set during analysis
        self.assertEqual(analysis.media_urls, [])
        self.assertEqual(analysis.media_type, "")
        self.assertFalse(analysis.multimodal_analysis)
        self.assertEqual(analysis.pattern_matches, [])
        self.assertIsNone(analysis.topic_classification)
        self.assertEqual(analysis.analysis_json, "")
        self.assertEqual(analysis.analysis_time_seconds, 0.0)
        self.assertEqual(analysis.model_used, "")
        self.assertEqual(analysis.tokens_used, 0)
        self.assertIsNone(analysis.verification_data)
        self.assertEqual(analysis.verification_confidence, 0.0)

    def test_post_init_initializes_lists(self):
        """Test __post_init__ properly initializes list fields."""
        analysis = ContentAnalysis(
            post_id="123",
            post_url="https://example.com/post/123",
            author_username="testuser",
            post_content="Test content",
            analysis_timestamp="2024-01-01T00:00:00Z",
            category="general",
            categories_detected=None,
            pattern_matches=None,
            media_urls=None
        )

        # __post_init__ should initialize None lists to empty lists
        self.assertEqual(analysis.categories_detected, [])
        self.assertEqual(analysis.pattern_matches, [])
        self.assertEqual(analysis.media_urls, [])

    def test_has_multiple_categories_single_category(self):
        """Test has_multiple_categories returns False for single category."""
        analysis = ContentAnalysis(
            post_id="123",
            post_url="https://example.com/post/123",
            author_username="testuser",
            post_content="Test content",
            analysis_timestamp="2024-01-01T00:00:00Z",
            category="general",
            categories_detected=["general"]
        )

        self.assertFalse(analysis.has_multiple_categories)

    def test_has_multiple_categories_multiple_categories(self):
        """Test has_multiple_categories returns True for multiple categories."""
        analysis = ContentAnalysis(
            post_id="123",
            post_url="https://example.com/post/123",
            author_username="testuser",
            post_content="Test content",
            analysis_timestamp="2024-01-01T00:00:00Z",
            category="hate_speech",
            categories_detected=["hate_speech", "far_right_bias"]
        )

        self.assertTrue(analysis.has_multiple_categories)

    def test_get_secondary_categories_no_secondary(self):
        """Test get_secondary_categories returns empty list when no secondary categories."""
        analysis = ContentAnalysis(
            post_id="123",
            post_url="https://example.com/post/123",
            author_username="testuser",
            post_content="Test content",
            analysis_timestamp="2024-01-01T00:00:00Z",
            category="general",
            categories_detected=["general"]
        )

        secondary = analysis.get_secondary_categories()
        self.assertEqual(secondary, [])

    def test_get_secondary_categories_with_secondary(self):
        """Test get_secondary_categories returns secondary categories excluding primary."""
        analysis = ContentAnalysis(
            post_id="123",
            post_url="https://example.com/post/123",
            author_username="testuser",
            post_content="Test content",
            analysis_timestamp="2024-01-01T00:00:00Z",
            category="hate_speech",
            categories_detected=["hate_speech", "far_right_bias", "call_to_action"]
        )

        secondary = analysis.get_secondary_categories()
        self.assertEqual(set(secondary), {"far_right_bias", "call_to_action"})
        self.assertNotIn("hate_speech", secondary)

    def test_get_secondary_categories_primary_not_in_list(self):
        """Test get_secondary_categories when primary category is not in detected list."""
        analysis = ContentAnalysis(
            post_id="123",
            post_url="https://example.com/post/123",
            author_username="testuser",
            post_content="Test content",
            analysis_timestamp="2024-01-01T00:00:00Z",
            category="hate_speech",
            categories_detected=["far_right_bias", "call_to_action"]
        )

        secondary = analysis.get_secondary_categories()
        self.assertEqual(set(secondary), {"far_right_bias", "call_to_action"})

    def test_init_with_all_fields(self):
        """Test ContentAnalysis initialization with all fields provided."""
        analysis = ContentAnalysis(
            post_id="123",
            post_url="https://example.com/post/123",
            author_username="testuser",
            post_content="Test content",
            analysis_timestamp="2024-01-01T00:00:00Z",
            category="hate_speech",
            categories_detected=["hate_speech", "far_right_bias"],
            local_explanation="This content shows hate speech",
            analysis_stages="pattern->local_llm",
            media_urls=["https://example.com/image.jpg"],
            media_type="image",
            pattern_matches=[{"pattern": "hate", "score": 0.9}],
            topic_classification={"topic": "politics"},
            analysis_json='{"result": "test"}',
            analysis_time_seconds=2.5,
            model_used="gpt-oss:20b",
            tokens_used=150,
            verification_data={"verified": True},
            verification_confidence=0.85
        )

        self.assertEqual(analysis.post_id, "123")
        self.assertEqual(analysis.category, "hate_speech")
        self.assertEqual(analysis.categories_detected, ["hate_speech", "far_right_bias"])
        self.assertEqual(analysis.local_explanation, "This content shows hate speech")
        self.assertEqual(analysis.analysis_stages, "pattern->local_llm")  # Test actual value passed in
        self.assertEqual(analysis.media_urls, ["https://example.com/image.jpg"])
        self.assertTrue(analysis.multimodal_analysis)  # Boolean flag
        self.assertEqual(analysis.media_type, "image")
        self.assertTrue(analysis.multimodal_analysis)
        self.assertEqual(analysis.pattern_matches, [{"pattern": "hate", "score": 0.9}])
        self.assertEqual(analysis.topic_classification, {"topic": "politics"})
        self.assertEqual(analysis.analysis_json, '{"result": "test"}')
        self.assertEqual(analysis.analysis_time_seconds, 2.5)
        self.assertEqual(analysis.model_used, "gpt-oss:20b")
        self.assertEqual(analysis.tokens_used, 150)
        self.assertEqual(analysis.verification_data, {"verified": True})
        self.assertEqual(analysis.verification_confidence, 0.85)


if __name__ == '__main__':
    unittest.main()