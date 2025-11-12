"""
Unit tests for AnalysisFlowManager component.
Tests orchestration of the 3-stage analysis pipeline.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch

from analyzer.flow_manager import AnalysisFlowManager, AnalysisStages
from analyzer.external_analyzer import ExternalAnalysisResult
from analyzer.categories import Categories
from analyzer.pattern_analyzer import AnalysisResult, PatternMatch


@pytest.fixture
def mock_pattern_analyzer():
    """Mock PatternAnalyzer."""
    mock = Mock()
    # Default: pattern detection finds hate_speech
    mock.analyze_content.return_value = AnalysisResult(
        categories=[Categories.HATE_SPEECH],
        pattern_matches=[PatternMatch(
            category=Categories.HATE_SPEECH,
            matched_text="inmigrantes",
            description="Hate speech, xenophobia, and violent threats targeting specific groups",
            context="Los inmigrantes destruyen"
        )],
        primary_category=Categories.HATE_SPEECH,
        political_context=["extremist rhetoric"],
        keywords=["inmigrantes", "destruir"]
    )
    return mock


@pytest.fixture
def mock_text_llm():
    """Mock text LLM analyzer."""
    mock = Mock()
    mock.detect_category_only = AsyncMock(return_value=Categories.HATE_SPEECH)
    mock.generate_explanation_with_context = AsyncMock(return_value="Explicación para categoría detectada por patrones.")
    return mock


@pytest.fixture
def mock_vision_llm():
    """Mock vision LLM analyzer."""
    mock = Mock()
    mock.describe_media = AsyncMock(return_value="Descripción de la imagen del medio.")
    return mock


@pytest.fixture
def mock_external():
    """Mock ExternalAnalyzer."""
    mock = Mock()
    mock.analyze = AsyncMock(return_value=ExternalAnalysisResult(
        category=None,  # Don't override local category by default
        explanation="Análisis externo detallado usando Gemini."
    ))
    return mock


@pytest.fixture
def flow_manager(mock_pattern_analyzer, mock_text_llm, mock_vision_llm, mock_external):
    """Create AnalysisFlowManager with mocked components."""
    with patch('analyzer.flow_manager.PatternAnalyzer', return_value=mock_pattern_analyzer), \
         patch('analyzer.flow_manager.OllamaAnalyzer') as mock_ollama_class, \
         patch('analyzer.flow_manager.ExternalAnalyzer', return_value=mock_external):
        
        # Configure OllamaAnalyzer to return different instances for text and vision
        mock_ollama_class.side_effect = [mock_text_llm, mock_vision_llm]
        
        manager = AnalysisFlowManager(verbose=False)
        return manager


class TestAnalysisStages:
    """Test AnalysisStages dataclass."""
    
    def test_default_initialization(self):
        """Test default stages are all False."""
        stages = AnalysisStages()
        
        assert stages.pattern is False
        assert stages.category_detection is False
        assert stages.media_analysis is False
        assert stages.explanation is False
        assert stages.external is False
    
    def test_to_string_all_stages(self):
        """Test converting stages to string."""
        stages = AnalysisStages(pattern=True, category_detection=True, media_analysis=True, explanation=True, external=True)
        
        result = stages.to_string()
        
        assert result == "pattern,category_detection,media_analysis,explanation,external"
    
    def test_to_string_partial_stages(self):
        """Test string conversion with partial stages."""
        stages = AnalysisStages(pattern=True, category_detection=True, explanation=True, external=False)
        
        result = stages.to_string()
        
        assert result == "pattern,category_detection,explanation"
    
    def test_to_string_no_stages(self):
        """Test string conversion with no stages."""
        stages = AnalysisStages()
        
        result = stages.to_string()
        
        assert result == ""
    
    def test_from_string_all_stages(self):
        """Test parsing stages from string."""
        stages = AnalysisStages.from_string("pattern,category_detection,media_analysis,explanation,external")
        
        assert stages.pattern is True
        assert stages.category_detection is True
        assert stages.media_analysis is True
        assert stages.explanation is True
        assert stages.external is True
    
    def test_from_string_partial(self):
        """Test parsing partial stages."""
        stages = AnalysisStages.from_string("pattern,category_detection,explanation")
        
        assert stages.pattern is True
        assert stages.category_detection is True
        assert stages.media_analysis is False
        assert stages.explanation is True
        assert stages.external is False
    
    def test_from_string_empty(self):
        """Test parsing empty string."""
        stages = AnalysisStages.from_string("")
        
        assert stages.pattern is False
        assert stages.category_detection is False
        assert stages.media_analysis is False
        assert stages.explanation is False
        assert stages.external is False
    
    def test_round_trip_conversion(self):
        """Test converting to string and back preserves state."""
        original = AnalysisStages(pattern=True, category_detection=True, explanation=True, external=False)
        
        string_repr = original.to_string()
        restored = AnalysisStages.from_string(string_repr)
        
        assert restored.pattern == original.pattern
        assert restored.category_detection == original.category_detection
        assert restored.media_analysis == original.media_analysis
        assert restored.explanation == original.explanation
        assert restored.external == original.external


class TestFlowManagerInitialization:
    """Test flow manager initialization."""
    
    def test_default_initialization(self):
        """Test default initialization creates all analyzers."""
        with patch('analyzer.flow_manager.PatternAnalyzer'), \
             patch('analyzer.flow_manager.OllamaAnalyzer'), \
             patch('analyzer.flow_manager.ExternalAnalyzer'):
            manager = AnalysisFlowManager()
            
            assert manager.verbose is False
            assert manager.pattern_analyzer is not None
            assert manager.text_llm is not None
            assert manager.vision_llm is not None
            assert manager.external is not None
    
    def test_verbose_initialization(self):
        """Test verbose initialization passes to components."""
        with patch('analyzer.flow_manager.PatternAnalyzer'), \
             patch('analyzer.flow_manager.OllamaAnalyzer') as mock_llm, \
             patch('analyzer.flow_manager.ExternalAnalyzer') as mock_ext:
            manager = AnalysisFlowManager(verbose=True)
            
            assert manager.verbose is True
            # Verify verbose passed to components - should be called twice (text_llm and vision_llm)
            assert mock_llm.call_count == 2
            mock_llm.assert_any_call(model='gemma3:27b-it-q4_K_M', verbose=True, fast_mode=False)
            mock_ext.assert_called_once_with(verbose=True)


class TestAnalyzeLocal:
    """Test local analysis flow (pattern + local LLM)."""
    
    @pytest.mark.asyncio
    async def test_local_with_successful_patterns(self, flow_manager, mock_pattern_analyzer, mock_text_llm):
        """Test local flow when pattern detection succeeds."""
        content = "Los inmigrantes destruyen nuestra cultura"
        
        result = await flow_manager.analyze_local(content)
        
        assert result.category == Categories.HATE_SPEECH
        assert result.local_explanation == "Explicación para categoría detectada por patrones."
        assert result.stages.pattern is True
        assert result.stages.category_detection is True
        assert result.stages.explanation is True
        assert result.stages.external is False
        
        # Verify pattern data is returned
        assert 'pattern_matches' in result.pattern_data
        assert 'topic_classification' in result.pattern_data
        assert len(result.pattern_data['pattern_matches']) == 1
        assert result.pattern_data['pattern_matches'][0]['matched_text'] == "inmigrantes"
        
        # Verify pattern analyzer was called
        mock_pattern_analyzer.analyze_content.assert_called_once_with(content)
        
        # Verify text LLM methods were called
        mock_text_llm.detect_category_only.assert_called_once()
        mock_text_llm.generate_explanation_with_context.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_local_with_failed_patterns(self, flow_manager, mock_pattern_analyzer, mock_text_llm):
        """Test local flow when pattern detection returns general category."""
        # Mock pattern detection finding only general category
        mock_pattern_analyzer.analyze_content.return_value = AnalysisResult(
            categories=[Categories.GENERAL],
            pattern_matches=[],
            primary_category=Categories.GENERAL,
            political_context=[],
            keywords=[]
        )
        
        content = "Contenido neutral sin patrones"
        
        result = await flow_manager.analyze_local(content)
        
        assert result.category == Categories.HATE_SPEECH  # From LLM mock
        assert result.local_explanation == "Explicación para categoría detectada por patrones."
        assert result.stages.pattern is True
        assert result.stages.category_detection is True
        assert result.stages.explanation is True
        
        # Verify pattern data is returned
        assert 'pattern_matches' in result.pattern_data
        assert 'topic_classification' in result.pattern_data
        assert len(result.pattern_data['pattern_matches']) == 0
        
        # Verify text LLM methods were called
        mock_text_llm.detect_category_only.assert_called_once()
        mock_text_llm.generate_explanation_with_context.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_local_with_no_categories(self, flow_manager, mock_pattern_analyzer, mock_text_llm):
        """Test local flow when pattern detection returns empty categories."""
        mock_pattern_analyzer.analyze_content.return_value = AnalysisResult(
            categories=[],
            pattern_matches=[],
            primary_category=Categories.GENERAL,
            political_context=[],
            keywords=[]
        )
        
        content = "Test content"
        
        result = await flow_manager.analyze_local(content)
        
        # Should use LLM for both category and explanation
        mock_text_llm.detect_category_only.assert_called_once()
        mock_text_llm.generate_explanation_with_context.assert_called_once()
        
        # Verify pattern data is returned
        assert 'pattern_matches' in result.pattern_data
        assert 'topic_classification' in result.pattern_data
    
    @pytest.mark.asyncio
    async def test_local_with_multiple_pattern_categories(self, flow_manager, mock_pattern_analyzer, mock_text_llm):
        """Test local flow uses first non-general category from patterns."""
        mock_pattern_analyzer.analyze_content.return_value = AnalysisResult(
            categories=[Categories.DISINFORMATION, Categories.ANTI_GOVERNMENT],
            pattern_matches=[PatternMatch(
                category=Categories.DISINFORMATION,
                matched_text="fake news",
                description="False information including health, statistical, and factual disinformation",
                context="fake news content"
            )],
            primary_category=Categories.DISINFORMATION,
            political_context=["misinformation"],
            keywords=["fake", "news"]
        )
        
        content = "Fake news content"
        
        result = await flow_manager.analyze_local(content)
        
        assert result.category == Categories.HATE_SPEECH  # From LLM mock
        mock_text_llm.detect_category_only.assert_called_once()
        mock_text_llm.generate_explanation_with_context.assert_called_once()
        
        # Verify pattern data is returned
        assert 'pattern_matches' in result.pattern_data
        assert 'topic_classification' in result.pattern_data
        assert len(result.pattern_data['pattern_matches']) == 1


class TestAnalyzeExternal:
    """Test external analysis flow."""
    
    @pytest.mark.asyncio
    async def test_external_with_text_only(self, flow_manager, mock_external):
        """Test external analysis with text only."""
        content = "Test content for external analysis"
        
        result = await flow_manager.analyze_external(content)
        
        assert isinstance(result, ExternalAnalysisResult)
        assert result.category is None
        assert result.explanation == "Análisis externo detallado usando Gemini."
        mock_external.analyze.assert_called_once_with(content, None)
    
    @pytest.mark.asyncio
    async def test_external_with_media(self, flow_manager, mock_external):
        """Test external analysis with media URLs."""
        content = "Content with images"
        media_urls = ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]
        
        result = await flow_manager.analyze_external(content, media_urls)
        
        assert isinstance(result, ExternalAnalysisResult)
        assert result.category is None
        assert result.explanation == "Análisis externo detallado usando Gemini."
        mock_external.analyze.assert_called_once_with(content, media_urls)


class TestAnalyzeFull:
    """Test complete analysis flow with conditional external analysis."""
    
    @pytest.mark.asyncio
    async def test_full_triggers_external_for_hate_speech(
        self, flow_manager, mock_pattern_analyzer, mock_text_llm, mock_external
    ):
        """Test full flow triggers external for hate_speech category."""
        content = "Hate speech content"
        
        result = await flow_manager.analyze_full(content)
        
        assert result.category == Categories.HATE_SPEECH
        assert result.local_explanation == "Explicación para categoría detectada por patrones."
        assert result.external_explanation == "Análisis externo detallado usando Gemini."
        assert result.stages.pattern is True
        assert result.stages.category_detection is True
        assert result.stages.explanation is True
        assert result.stages.external is True  # Should run external
        
        # Verify pattern data is returned
        assert 'pattern_matches' in result.pattern_data
        assert 'topic_classification' in result.pattern_data
        
        # Verify external was called
        mock_external.analyze.assert_called_once_with(content, None)
    
    @pytest.mark.asyncio
    async def test_full_skips_external_for_general(
        self, flow_manager, mock_pattern_analyzer, mock_text_llm, mock_external
    ):
        """Test full flow skips external for general category."""
        # Mock pattern detection finding general category
        mock_pattern_analyzer.analyze_content.return_value = AnalysisResult(
            categories=[Categories.GENERAL],
            pattern_matches=[],
            primary_category=Categories.GENERAL,
            political_context=[],
            keywords=[]
        )
        
        # Mock text LLM to also return GENERAL
        mock_text_llm.detect_category_only.return_value = Categories.GENERAL
        mock_text_llm.generate_explanation_with_context.return_value = "Contenido general sin patrones problemáticos."
        
        content = "Normal neutral content"
        
        result = await flow_manager.analyze_full(content)
        
        assert result.category == Categories.GENERAL
        assert result.external_explanation is None  # Should NOT run external
        assert result.stages.external is False
        
        # Verify pattern data is returned
        assert 'pattern_matches' in result.pattern_data
        assert 'topic_classification' in result.pattern_data
        
        # Verify external was NOT called
        mock_external.analyze.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_full_skips_external_for_political_general(
        self, flow_manager, mock_pattern_analyzer, mock_text_llm, mock_external
    ):
        """Test full flow skips external for political_general category."""
        mock_pattern_analyzer.analyze_content.return_value = AnalysisResult(
            categories=[Categories.POLITICAL_GENERAL],
            pattern_matches=[PatternMatch(
                category=Categories.POLITICAL_GENERAL,
                matched_text="political term",
                description="General political content without extremist elements",
                context="political discussion"
            )],
            primary_category=Categories.POLITICAL_GENERAL,
            political_context=["general politics"],
            keywords=["política"]
        )
        
        # Mock text LLM to return POLITICAL_GENERAL
        mock_text_llm.detect_category_only.return_value = Categories.POLITICAL_GENERAL
        mock_text_llm.generate_explanation_with_context.return_value = "Contenido político general."
        
        content = "General political discussion"
        
        result = await flow_manager.analyze_full(content)
        
        assert result.category == Categories.POLITICAL_GENERAL
        assert result.external_explanation is None
        assert result.stages.external is False
        
        # Verify pattern data is returned
        assert 'pattern_matches' in result.pattern_data
        assert 'topic_classification' in result.pattern_data
        
        mock_external.analyze.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_full_admin_override_triggers_external(
        self, flow_manager, mock_pattern_analyzer, mock_text_llm, mock_external
    ):
        """Test admin override triggers external even for general category."""
        mock_pattern_analyzer.analyze_content.return_value = AnalysisResult(
            categories=[Categories.GENERAL],
            pattern_matches=[],
            primary_category=Categories.GENERAL,
            political_context=[],
            keywords=[]
        )
        
        # Mock text LLM to return GENERAL
        mock_text_llm.detect_category_only.return_value = Categories.GENERAL
        mock_text_llm.generate_explanation_with_context.return_value = "Contenido general."
        
        content = "Content requiring admin review"
        
        result = await flow_manager.analyze_full(content, admin_override=True)
        
        assert result.category == Categories.GENERAL
        assert result.external_explanation == "Análisis externo detallado usando Gemini."
        assert result.stages.external is True  # Should run due to admin override
        
        # Verify pattern data is returned
        assert 'pattern_matches' in result.pattern_data
        assert 'topic_classification' in result.pattern_data
        
        mock_external.analyze.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_full_with_media_urls(
        self, flow_manager, mock_pattern_analyzer, mock_text_llm, mock_external
    ):
        """Test full flow passes media URLs to external analyzer."""
        content = "Content with images"
        media_urls = ["https://example.com/image.jpg"]
        
        result = await flow_manager.analyze_full(content, media_urls=media_urls)
        
        # Verify pattern data is returned
        assert 'pattern_matches' in result.pattern_data
        assert 'topic_classification' in result.pattern_data
        
        # Verify media URLs passed to external
        mock_external.analyze.assert_called_once_with(content, media_urls)
    
    @pytest.mark.asyncio
    async def test_full_triggers_external_for_all_non_general_categories(
        self, flow_manager, mock_pattern_analyzer, mock_text_llm, mock_external
    ):
        """Test external analysis triggers for all problematic categories."""
        problematic_categories = [
            Categories.HATE_SPEECH,
            Categories.DISINFORMATION,
            Categories.CONSPIRACY_THEORY,
            Categories.ANTI_IMMIGRATION,
            Categories.ANTI_LGBTQ,
            Categories.ANTI_FEMINISM,
            Categories.NATIONALISM,
            Categories.ANTI_GOVERNMENT,
            Categories.HISTORICAL_REVISIONISM,
            Categories.CALL_TO_ACTION
        ]
        
        for category in problematic_categories:
            mock_pattern_analyzer.analyze_content.return_value = AnalysisResult(
                categories=[category],
                pattern_matches=[PatternMatch(
                    category=category,
                    matched_text="test",
                    description=f"Test description for {category}",
                    context="test content"
                )],
                primary_category=category,
                political_context=["test"],
                keywords=["test"]
            )
            mock_external.analyze.reset_mock()
            
            result = await flow_manager.analyze_full("Test content")
            
            assert result.external_explanation is not None, f"External should run for {category}"
            assert result.stages.external is True, f"External stage should be True for {category}"
            
            # Verify pattern data is returned
            assert 'pattern_matches' in result.pattern_data
            assert 'topic_classification' in result.pattern_data
            
    @pytest.mark.asyncio
    async def test_full_external_overrides_category_any_category(
        self, flow_manager, mock_pattern_analyzer, mock_text_llm, mock_external
    ):
        """Test that external analysis can override category for any category, not just disinformation."""
        # Mock local analysis returns hate_speech
        mock_pattern_analyzer.analyze_content.return_value = AnalysisResult(
            categories=[Categories.HATE_SPEECH],
            pattern_matches=[],
            primary_category=Categories.HATE_SPEECH,
            political_context=[],
            keywords=[]
        )
        
        # Mock external analysis to return conspiracy_theory category
        mock_external.analyze.return_value = ExternalAnalysisResult(
            category=Categories.CONSPIRACY_THEORY,
            explanation="Este contenido promueve teorías conspirativas sobre el control global."
        )
        
        content = "Content that external analysis reclassifies"
        
        result = await flow_manager.analyze_full(content)
        
        # Should be overridden to conspiracy_theory
        assert result.category == Categories.CONSPIRACY_THEORY
        assert result.external_explanation == "Este contenido promueve teorías conspirativas sobre el control global."
        assert result.stages.external is True
    """Test verbose logging output."""
    
    @pytest.mark.asyncio
    async def test_local_verbose_output(self, mock_pattern_analyzer, mock_text_llm, mock_vision_llm, mock_external, capsys):
        """Test verbose logging in local analysis."""
        with patch('analyzer.flow_manager.PatternAnalyzer', return_value=mock_pattern_analyzer), \
             patch('analyzer.flow_manager.OllamaAnalyzer') as mock_ollama_class, \
             patch('analyzer.flow_manager.ExternalAnalyzer', return_value=mock_external):
            
            mock_ollama_class.side_effect = [mock_text_llm, mock_vision_llm]
            manager = AnalysisFlowManager(verbose=True)
            
            await manager.analyze_local("Test content")
            
            captured = capsys.readouterr()
            assert "🔄 Starting LOCAL analysis flow" in captured.out
            assert "📊 Stage 1: Pattern Detection" in captured.out
            assert "🤖 Stage 2:" in captured.out
    
    @pytest.mark.asyncio
    async def test_external_verbose_output(self, mock_pattern_analyzer, mock_text_llm, mock_vision_llm, mock_external, capsys):
        """Test verbose logging in external analysis."""
        with patch('analyzer.flow_manager.PatternAnalyzer', return_value=mock_pattern_analyzer), \
             patch('analyzer.flow_manager.OllamaAnalyzer') as mock_ollama_class, \
             patch('analyzer.flow_manager.ExternalAnalyzer', return_value=mock_external):
            
            mock_ollama_class.side_effect = [mock_text_llm, mock_vision_llm]
            manager = AnalysisFlowManager(verbose=True)
            
            await manager.analyze_external("Test content", media_urls=["https://example.com/image.jpg"])
            
            captured = capsys.readouterr()
            assert "🌐 Starting EXTERNAL analysis flow" in captured.out
            assert "🖼️  Media:" in captured.out
    
    @pytest.mark.asyncio
    async def test_full_verbose_output_with_external(
        self, mock_pattern_analyzer, mock_text_llm, mock_vision_llm, mock_external, capsys
    ):
        """Test verbose logging in full analysis with external."""
        with patch('analyzer.flow_manager.PatternAnalyzer', return_value=mock_pattern_analyzer), \
             patch('analyzer.flow_manager.OllamaAnalyzer') as mock_ollama_class, \
             patch('analyzer.flow_manager.ExternalAnalyzer', return_value=mock_external):
            
            mock_ollama_class.side_effect = [mock_text_llm, mock_vision_llm]
            manager = AnalysisFlowManager(verbose=True)
            
            await manager.analyze_full("Test content")
            
            captured = capsys.readouterr()
            assert "🌐 External analysis triggered" in captured.out
            assert "✅ Analysis complete:" in captured.out
            assert "Stages: pattern,category_detection,explanation,external" in captured.out


class TestVerificationTimeout:
    """Test verification timeout functionality."""
    
    @pytest.mark.asyncio
    async def test_verification_timeout_enforced(self):
        """Test that verification has timeout enforced."""
        with patch('analyzer.flow_manager.PatternAnalyzer') as mock_pattern_class, \
             patch('analyzer.flow_manager.OllamaAnalyzer') as mock_ollama_class, \
             patch('analyzer.flow_manager.ExternalAnalyzer') as mock_external_class, \
             patch('analyzer.flow_manager.create_analyzer_hooks') as mock_hooks_factory, \
             patch('analyzer.flow_manager.ConfigDefaults.VERIFICATION_TIMEOUT', 2.0):  # Short timeout for tests
            
            # Setup mocks
            mock_pattern = Mock()
            mock_pattern.analyze_content.return_value = AnalysisResult(
                categories=[Categories.ANTI_GOVERNMENT],
                pattern_matches=[],
                primary_category=Categories.ANTI_GOVERNMENT,
                political_context=["anti-government"],
                keywords=["gobierno"]
            )
            mock_pattern_class.return_value = mock_pattern
            
            mock_text_llm = Mock()
            mock_text_llm.detect_category_only = AsyncMock(return_value=Categories.ANTI_GOVERNMENT)
            mock_text_llm.generate_explanation_with_context = AsyncMock(return_value="Explanation")
            
            mock_vision_llm = Mock()
            mock_vision_llm.describe_media = AsyncMock(return_value="")
            
            mock_ollama_class.side_effect = [mock_text_llm, mock_vision_llm]
            
            mock_external = Mock()
            mock_external.analyze = AsyncMock(return_value=ExternalAnalysisResult(
                category=None,
                explanation="External explanation"
            ))
            mock_external_class.return_value = mock_external
            
            # Mock analyzer hooks with slow verification
            mock_hooks = Mock()
            
            async def slow_verification(*args, **kwargs):
                """Simulate slow verification that times out."""
                await asyncio.sleep(5)  # Longer than 2s timeout but much faster than 200s
                return Mock()
            
            mock_hooks.analyze_with_verification = slow_verification
            mock_hooks.should_trigger_verification = Mock(return_value=(True, "test trigger"))
            mock_hooks._extract_political_event_claim = Mock(return_value=None)
            mock_hooks_factory.return_value = mock_hooks
            
            manager = AnalysisFlowManager(verbose=False)
            
            # Should not hang, should timeout after 2s
            start_time = asyncio.get_event_loop().time()
            result = await manager.analyze_full(
                "Test content about gobierno"
            )
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # Should complete much faster than 5s (the mock delay)
            assert elapsed < 3  # Allow small buffer over 2s timeout
            assert result.category is not None  # Should still return results
            assert result.local_explanation is not None  # Should have local explanation
    
    @pytest.mark.asyncio
    async def test_verification_timeout_continues_with_local_results(self):
        """Test that timeout allows analysis to continue with local results."""
        with patch('analyzer.flow_manager.PatternAnalyzer') as mock_pattern_class, \
             patch('analyzer.flow_manager.OllamaAnalyzer') as mock_ollama_class, \
             patch('analyzer.flow_manager.ExternalAnalyzer') as mock_external_class, \
             patch('analyzer.flow_manager.create_analyzer_hooks') as mock_hooks_factory:
            
            # Setup mocks
            mock_pattern = Mock()
            mock_pattern.analyze_content.return_value = AnalysisResult(
                categories=[Categories.DISINFORMATION],
                pattern_matches=[],
                primary_category=Categories.DISINFORMATION,
                political_context=[],
                keywords=[]
            )
            mock_pattern_class.return_value = mock_pattern
            
            mock_text_llm = Mock()
            mock_text_llm.detect_category_only = AsyncMock(return_value=Categories.DISINFORMATION)
            mock_text_llm.generate_explanation_with_context = AsyncMock(return_value="Local explanation")
            
            mock_vision_llm = Mock()
            mock_ollama_class.side_effect = [mock_text_llm, mock_vision_llm]
            
            mock_external = Mock()
            mock_external_class.return_value = mock_external
            
            # Mock analyzer hooks with timeout
            mock_hooks = Mock()
            mock_hooks.analyze_with_verification = AsyncMock(side_effect=asyncio.TimeoutError)
            mock_hooks.should_trigger_verification = Mock(return_value=(True, "test"))
            mock_hooks._extract_political_event_claim = Mock(return_value=None)
            mock_hooks_factory.return_value = mock_hooks
            
            manager = AnalysisFlowManager(verbose=False)
            
            result = await manager.analyze_full("Test content")
            
            # Should still return local results
            assert result.category == Categories.DISINFORMATION
            assert result.local_explanation == "Local explanation"
            assert result.external_explanation is None  # No external analysis
