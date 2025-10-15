"""
Unit tests for AnalysisFlowManager component.
Tests orchestration of the 3-stage analysis pipeline.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch

from analyzer.flow_manager import AnalysisFlowManager, AnalysisStages
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
def mock_local_llm():
    """Mock LocalLLMAnalyzer."""
    mock = Mock()
    mock.categorize_and_explain = AsyncMock(return_value=(
        Categories.HATE_SPEECH,
        "Este contenido contiene discurso de odio xenófobo."
    ))
    mock.explain_only = AsyncMock(return_value="Explicación para categoría detectada por patrones.")
    return mock


@pytest.fixture
def mock_external():
    """Mock ExternalAnalyzer."""
    mock = Mock()
    mock.analyze = AsyncMock(return_value="Análisis externo detallado usando Gemini.")
    return mock


@pytest.fixture
def flow_manager(mock_pattern_analyzer, mock_local_llm, mock_external):
    """Create AnalysisFlowManager with mocked components."""
    with patch('analyzer.flow_manager.PatternAnalyzer', return_value=mock_pattern_analyzer), \
         patch('analyzer.flow_manager.LocalLLMAnalyzer', return_value=mock_local_llm), \
         patch('analyzer.flow_manager.ExternalAnalyzer', return_value=mock_external):
        manager = AnalysisFlowManager(verbose=False)
        return manager


class TestAnalysisStages:
    """Test AnalysisStages dataclass."""
    
    def test_default_initialization(self):
        """Test default stages are all False."""
        stages = AnalysisStages()
        
        assert stages.pattern is False
        assert stages.local_llm is False
        assert stages.external is False
    
    def test_to_string_all_stages(self):
        """Test converting stages to string."""
        stages = AnalysisStages(pattern=True, local_llm=True, external=True)
        
        result = stages.to_string()
        
        assert result == "pattern,local_llm,external"
    
    def test_to_string_partial_stages(self):
        """Test string conversion with partial stages."""
        stages = AnalysisStages(pattern=True, local_llm=True, external=False)
        
        result = stages.to_string()
        
        assert result == "pattern,local_llm"
    
    def test_to_string_no_stages(self):
        """Test string conversion with no stages."""
        stages = AnalysisStages()
        
        result = stages.to_string()
        
        assert result == ""
    
    def test_from_string_all_stages(self):
        """Test parsing stages from string."""
        stages = AnalysisStages.from_string("pattern,local_llm,external")
        
        assert stages.pattern is True
        assert stages.local_llm is True
        assert stages.external is True
    
    def test_from_string_partial(self):
        """Test parsing partial stages."""
        stages = AnalysisStages.from_string("pattern,local_llm")
        
        assert stages.pattern is True
        assert stages.local_llm is True
        assert stages.external is False
    
    def test_from_string_empty(self):
        """Test parsing empty string."""
        stages = AnalysisStages.from_string("")
        
        assert stages.pattern is False
        assert stages.local_llm is False
        assert stages.external is False
    
    def test_round_trip_conversion(self):
        """Test converting to string and back preserves state."""
        original = AnalysisStages(pattern=True, local_llm=True, external=False)
        
        string_repr = original.to_string()
        restored = AnalysisStages.from_string(string_repr)
        
        assert restored.pattern == original.pattern
        assert restored.local_llm == original.local_llm
        assert restored.external == original.external


class TestFlowManagerInitialization:
    """Test flow manager initialization."""
    
    def test_default_initialization(self):
        """Test default initialization creates all analyzers."""
        with patch('analyzer.flow_manager.PatternAnalyzer'), \
             patch('analyzer.flow_manager.LocalLLMAnalyzer'), \
             patch('analyzer.flow_manager.ExternalAnalyzer'):
            manager = AnalysisFlowManager()
            
            assert manager.verbose is False
            assert manager.pattern_analyzer is not None
            assert manager.local_llm is not None
            assert manager.external is not None
    
    def test_verbose_initialization(self):
        """Test verbose initialization passes to components."""
        with patch('analyzer.flow_manager.PatternAnalyzer'), \
             patch('analyzer.flow_manager.LocalLLMAnalyzer') as mock_llm, \
             patch('analyzer.flow_manager.ExternalAnalyzer') as mock_ext:
            manager = AnalysisFlowManager(verbose=True)
            
            assert manager.verbose is True
            # Verify verbose passed to components
            mock_llm.assert_called_once_with(verbose=True)
            mock_ext.assert_called_once_with(verbose=True)


class TestAnalyzeLocal:
    """Test local analysis flow (pattern + local LLM)."""
    
    @pytest.mark.asyncio
    async def test_local_with_successful_patterns(self, flow_manager, mock_pattern_analyzer, mock_local_llm):
        """Test local flow when pattern detection succeeds."""
        content = "Los inmigrantes destruyen nuestra cultura"
        
        category, explanation, stages, pattern_data = await flow_manager.analyze_local(content)
        
        assert category == Categories.HATE_SPEECH
        assert explanation == "Explicación para categoría detectada por patrones."
        assert stages.pattern is True
        assert stages.local_llm is True
        assert stages.external is False
        
        # Verify pattern data is returned
        assert 'pattern_matches' in pattern_data
        assert 'topic_classification' in pattern_data
        assert len(pattern_data['pattern_matches']) == 1
        assert pattern_data['pattern_matches'][0]['matched_text'] == "inmigrantes"
        
        # Verify pattern analyzer was called
        mock_pattern_analyzer.analyze_content.assert_called_once_with(content)
        
        # Verify local LLM explain_only was called (not categorize_and_explain)
        mock_local_llm.explain_only.assert_called_once_with(content, Categories.HATE_SPEECH)
        mock_local_llm.categorize_and_explain.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_local_with_failed_patterns(self, flow_manager, mock_pattern_analyzer, mock_local_llm):
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
        
        category, explanation, stages, pattern_data = await flow_manager.analyze_local(content)
        
        assert category == Categories.HATE_SPEECH  # From LLM mock
        assert explanation == "Este contenido contiene discurso de odio xenófobo."
        assert stages.pattern is True
        assert stages.local_llm is True
        
        # Verify pattern data is returned
        assert 'pattern_matches' in pattern_data
        assert 'topic_classification' in pattern_data
        assert len(pattern_data['pattern_matches']) == 0
        
        # Verify local LLM categorize_and_explain was called (not explain_only)
        mock_local_llm.categorize_and_explain.assert_called_once_with(content)
        mock_local_llm.explain_only.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_local_with_no_categories(self, flow_manager, mock_pattern_analyzer, mock_local_llm):
        """Test local flow when pattern detection returns empty categories."""
        mock_pattern_analyzer.analyze_content.return_value = AnalysisResult(
            categories=[],
            pattern_matches=[],
            primary_category=Categories.GENERAL,
            political_context=[],
            keywords=[]
        )
        
        content = "Test content"
        
        category, explanation, stages, pattern_data = await flow_manager.analyze_local(content)
        
        # Should use LLM for both category and explanation
        mock_local_llm.categorize_and_explain.assert_called_once()
        mock_local_llm.explain_only.assert_not_called()
        
        # Verify pattern data is returned
        assert 'pattern_matches' in pattern_data
        assert 'topic_classification' in pattern_data
    
    @pytest.mark.asyncio
    async def test_local_with_multiple_pattern_categories(self, flow_manager, mock_pattern_analyzer, mock_local_llm):
        """Test local flow uses first non-general category from patterns."""
        mock_pattern_analyzer.analyze_content.return_value = AnalysisResult(
            categories=[Categories.DISINFORMATION, Categories.FAR_RIGHT_BIAS],
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
        
        category, explanation, stages, pattern_data = await flow_manager.analyze_local(content)
        
        assert category == Categories.DISINFORMATION  # First category
        mock_local_llm.explain_only.assert_called_once_with(content, Categories.DISINFORMATION)
        
        # Verify pattern data is returned
        assert 'pattern_matches' in pattern_data
        assert 'topic_classification' in pattern_data
        assert len(pattern_data['pattern_matches']) == 1


class TestAnalyzeExternal:
    """Test external analysis flow."""
    
    @pytest.mark.asyncio
    async def test_external_with_text_only(self, flow_manager, mock_external):
        """Test external analysis with text only."""
        content = "Test content for external analysis"
        
        result = await flow_manager.analyze_external(content)
        
        assert result == "Análisis externo detallado usando Gemini."
        mock_external.analyze.assert_called_once_with(content, None)
    
    @pytest.mark.asyncio
    async def test_external_with_media(self, flow_manager, mock_external):
        """Test external analysis with media URLs."""
        content = "Content with images"
        media_urls = ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]
        
        result = await flow_manager.analyze_external(content, media_urls)
        
        assert result == "Análisis externo detallado usando Gemini."
        mock_external.analyze.assert_called_once_with(content, media_urls)


class TestAnalyzeFull:
    """Test complete analysis flow with conditional external analysis."""
    
    @pytest.mark.asyncio
    async def test_full_triggers_external_for_hate_speech(
        self, flow_manager, mock_pattern_analyzer, mock_local_llm, mock_external
    ):
        """Test full flow triggers external for hate_speech category."""
        content = "Hate speech content"
        
        category, local_exp, external_exp, stages, pattern_data = await flow_manager.analyze_full(content)
        
        assert category == Categories.HATE_SPEECH
        assert local_exp == "Explicación para categoría detectada por patrones."
        assert external_exp == "Análisis externo detallado usando Gemini."
        assert stages.pattern is True
        assert stages.local_llm is True
        assert stages.external is True  # Should run external
        
        # Verify pattern data is returned
        assert 'pattern_matches' in pattern_data
        assert 'topic_classification' in pattern_data
        
        # Verify external was called
        mock_external.analyze.assert_called_once_with(content, None)
    
    @pytest.mark.asyncio
    async def test_full_skips_external_for_general(
        self, flow_manager, mock_pattern_analyzer, mock_local_llm, mock_external
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
        mock_local_llm.categorize_and_explain.return_value = (
            Categories.GENERAL,
            "Contenido general sin patrones problemáticos."
        )
        
        content = "Normal neutral content"
        
        category, local_exp, external_exp, stages, pattern_data = await flow_manager.analyze_full(content)
        
        assert category == Categories.GENERAL
        assert external_exp is None  # Should NOT run external
        assert stages.external is False
        
        # Verify pattern data is returned
        assert 'pattern_matches' in pattern_data
        assert 'topic_classification' in pattern_data
        
        # Verify external was NOT called
        mock_external.analyze.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_full_skips_external_for_political_general(
        self, flow_manager, mock_pattern_analyzer, mock_local_llm, mock_external
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
        
        content = "General political discussion"
        
        category, local_exp, external_exp, stages, pattern_data = await flow_manager.analyze_full(content)
        
        assert category == Categories.POLITICAL_GENERAL
        assert external_exp is None
        assert stages.external is False
        
        # Verify pattern data is returned
        assert 'pattern_matches' in pattern_data
        assert 'topic_classification' in pattern_data
        
        mock_external.analyze.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_full_admin_override_triggers_external(
        self, flow_manager, mock_pattern_analyzer, mock_local_llm, mock_external
    ):
        """Test admin override triggers external even for general category."""
        mock_pattern_analyzer.analyze_content.return_value = AnalysisResult(
            categories=[Categories.GENERAL],
            pattern_matches=[],
            primary_category=Categories.GENERAL,
            political_context=[],
            keywords=[]
        )
        mock_local_llm.categorize_and_explain.return_value = (
            Categories.GENERAL,
            "Contenido general."
        )
        
        content = "Content requiring admin review"
        
        category, local_exp, external_exp, stages, pattern_data = await flow_manager.analyze_full(
            content,
            admin_override=True
        )
        
        assert category == Categories.GENERAL
        assert external_exp == "Análisis externo detallado usando Gemini."
        assert stages.external is True  # Should run due to admin override
        
        # Verify pattern data is returned
        assert 'pattern_matches' in pattern_data
        assert 'topic_classification' in pattern_data
        
        mock_external.analyze.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_full_with_media_urls(
        self, flow_manager, mock_pattern_analyzer, mock_local_llm, mock_external
    ):
        """Test full flow passes media URLs to external analyzer."""
        content = "Content with images"
        media_urls = ["https://example.com/image.jpg"]
        
        category, local_exp, external_exp, stages, pattern_data = await flow_manager.analyze_full(
            content,
            media_urls=media_urls
        )
        
        # Verify pattern data is returned
        assert 'pattern_matches' in pattern_data
        assert 'topic_classification' in pattern_data
        
        # Verify media URLs passed to external
        mock_external.analyze.assert_called_once_with(content, media_urls)
    
    @pytest.mark.asyncio
    async def test_full_triggers_external_for_all_non_general_categories(
        self, flow_manager, mock_pattern_analyzer, mock_local_llm, mock_external
    ):
        """Test external analysis triggers for all problematic categories."""
        problematic_categories = [
            Categories.HATE_SPEECH,
            Categories.DISINFORMATION,
            Categories.CONSPIRACY_THEORY,
            Categories.FAR_RIGHT_BIAS,
            Categories.CALL_TO_ACTION,
            Categories.NATIONALISM,
            Categories.ANTI_GOVERNMENT,
            Categories.HISTORICAL_REVISIONISM
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
            
            _, _, external_exp, stages, pattern_data = await flow_manager.analyze_full("Test content")
            
            assert external_exp is not None, f"External should run for {category}"
            assert stages.external is True, f"External stage should be True for {category}"
            
            # Verify pattern data is returned
            assert 'pattern_matches' in pattern_data
            assert 'topic_classification' in pattern_data
            
            mock_external.analyze.assert_called_once()


class TestVerboseLogging:
    """Test verbose logging output."""
    
    @pytest.mark.asyncio
    async def test_local_verbose_output(self, mock_pattern_analyzer, mock_local_llm, mock_external, capsys):
        """Test verbose logging in local analysis."""
        with patch('analyzer.flow_manager.PatternAnalyzer', return_value=mock_pattern_analyzer), \
             patch('analyzer.flow_manager.LocalLLMAnalyzer', return_value=mock_local_llm), \
             patch('analyzer.flow_manager.ExternalAnalyzer', return_value=mock_external):
            manager = AnalysisFlowManager(verbose=True)
            
            await manager.analyze_local("Test content")
            
            captured = capsys.readouterr()
            assert "🔄 Starting LOCAL analysis flow" in captured.out
            assert "📊 Stage 1: Pattern Detection" in captured.out
            assert "🤖 Stage 2:" in captured.out
    
    @pytest.mark.asyncio
    async def test_external_verbose_output(self, mock_pattern_analyzer, mock_local_llm, mock_external, capsys):
        """Test verbose logging in external analysis."""
        with patch('analyzer.flow_manager.PatternAnalyzer', return_value=mock_pattern_analyzer), \
             patch('analyzer.flow_manager.LocalLLMAnalyzer', return_value=mock_local_llm), \
             patch('analyzer.flow_manager.ExternalAnalyzer', return_value=mock_external):
            manager = AnalysisFlowManager(verbose=True)
            
            await manager.analyze_external("Test content", media_urls=["https://example.com/image.jpg"])
            
            captured = capsys.readouterr()
            assert "🌐 Starting EXTERNAL analysis flow" in captured.out
            assert "🖼️  Media:" in captured.out
    
    @pytest.mark.asyncio
    async def test_full_verbose_output_with_external(
        self, mock_pattern_analyzer, mock_local_llm, mock_external, capsys
    ):
        """Test verbose logging in full analysis with external."""
        with patch('analyzer.flow_manager.PatternAnalyzer', return_value=mock_pattern_analyzer), \
             patch('analyzer.flow_manager.LocalLLMAnalyzer', return_value=mock_local_llm), \
             patch('analyzer.flow_manager.ExternalAnalyzer', return_value=mock_external):
            manager = AnalysisFlowManager(verbose=True)
            
            await manager.analyze_full("Test content")
            
            captured = capsys.readouterr()
            assert "🌐 External analysis triggered" in captured.out
            assert "✅ Analysis complete:" in captured.out
            assert "Stages: pattern,local_llm,external" in captured.out
