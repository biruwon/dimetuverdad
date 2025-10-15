"""
Analysis Flow Manager - Orchestrates the 3-stage pipeline.

Stage 1: Pattern Detection
Stage 2: Local LLM Analysis (gpt-oss:20b)
Stage 3: External Analysis (Gemini, admin-triggered only)
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass

from .pattern_analyzer import PatternAnalyzer
from .local_llm_analyzer import LocalLLMAnalyzer
from .external_analyzer import ExternalAnalyzer
from .categories import Categories


@dataclass
class AnalysisStages:
    """Records which stages were executed"""
    pattern: bool = False
    local_llm: bool = False
    external: bool = False
    
    def to_string(self) -> str:
        """Convert to comma-separated string for database storage"""
        stages = []
        if self.pattern:
            stages.append("pattern")
        if self.local_llm:
            stages.append("local_llm")
        if self.external:
            stages.append("external")
        return ",".join(stages)
    
    @classmethod
    def from_string(cls, stages_str: str) -> "AnalysisStages":
        """Parse from database string"""
        stages = AnalysisStages()
        if stages_str:
            parts = stages_str.split(",")
            stages.pattern = "pattern" in parts
            stages.local_llm = "local_llm" in parts
            stages.external = "external" in parts
        return stages


class AnalysisFlowManager:
    """
    Orchestrates the multi-stage content analysis pipeline.
    
    Flow:
    1. Try pattern detection first (fast, rule-based)
    2. If patterns insufficient → Local LLM analysis (gpt-oss:20b)
    3. External analysis only when explicitly triggered (admin action)
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize flow manager with all analyzers.
        
        Args:
            verbose: Enable detailed logging
        """
        self.pattern_analyzer = PatternAnalyzer()
        self.local_llm = LocalLLMAnalyzer(verbose=verbose)
        self.external = ExternalAnalyzer(verbose=verbose)
        self.verbose = verbose
        
        if self.verbose:
            print("🔄 AnalysisFlowManager initialized")
    
    async def analyze_local(
        self,
        content: str,
        media_urls: Optional[List[str]] = None
    ) -> Tuple[str, str, AnalysisStages, dict]:
        """
        Run local analysis flow (pattern + local LLM).
        
        Args:
            content: Text content to analyze
            media_urls: Optional media URLs (not used in local flow)
        
        Returns:
            Tuple of (category, local_explanation, stages, pattern_data)
            where pattern_data contains pattern_matches and topic_classification
        """
        stages = AnalysisStages()
        
        if self.verbose:
            print("=" * 80)
            print("🔄 Starting LOCAL analysis flow")
            print(f"📝 Content: {content[:100]}...")
        
        # Stage 1: Pattern Detection
        if self.verbose:
            print("\n📊 Stage 1: Pattern Detection")
        
        pattern_result = self.pattern_analyzer.analyze_content(content)
        stages.pattern = True
        
        if self.verbose:
            print(f"   Categories found: {pattern_result.categories}")
            print(f"   Pattern matches: {len(pattern_result.pattern_matches)}")
        
        # Check if pattern detection succeeded
        if pattern_result.categories and Categories.GENERAL not in pattern_result.categories:
            # Patterns found a non-general category
            primary_category = pattern_result.categories[0]
            
            if self.verbose:
                print(f"   ✅ Pattern detection succeeded: {primary_category}")
                print(f"\n🤖 Stage 2: Local LLM (explanation only)")
            
            # Stage 2: Get explanation from local LLM
            local_explanation = await self.local_llm.explain_only(content, primary_category)
            stages.local_llm = True
            
            if self.verbose:
                print(f"   ✅ Local explanation: {local_explanation[:100]}...")
            
            # Return pattern data
            pattern_data = {
                'pattern_matches': [match.__dict__ for match in pattern_result.pattern_matches],
                'topic_classification': {
                    'categories': pattern_result.categories,
                    'primary_category': pattern_result.primary_category,
                    'political_context': pattern_result.political_context,
                    'keywords': pattern_result.keywords
                }
            }
            
            return primary_category, local_explanation, stages, pattern_data
        
        # Stage 2: Pattern detection failed or returned general → Use local LLM for both
        if self.verbose:
            print("   ⚠️  Pattern detection insufficient")
            print("\n🤖 Stage 2: Local LLM (categorization + explanation)")
        
        category, local_explanation = await self.local_llm.categorize_and_explain(content)
        stages.local_llm = True
        
        if self.verbose:
            print(f"   ✅ Local LLM category: {category}")
            print(f"   ✅ Local LLM explanation: {local_explanation[:100]}...")
        
        # Return pattern data (even if patterns didn't find categories)
        pattern_data = {
            'pattern_matches': [match.__dict__ for match in pattern_result.pattern_matches],
            'topic_classification': {
                'categories': pattern_result.categories,
                'primary_category': pattern_result.primary_category,
                'political_context': pattern_result.political_context,
                'keywords': pattern_result.keywords
            }
        }
        
        return category, local_explanation, stages, pattern_data
    
    async def analyze_external(
        self,
        content: str,
        media_urls: Optional[List[str]] = None
    ) -> str:
        """
        Run external analysis (Gemini).
        
        This is a separate flow, triggered only by admin action.
        Does NOT depend on local analysis results.
        
        Args:
            content: Text content to analyze
            media_urls: Optional media URLs for multimodal analysis
        
        Returns:
            External explanation (independent from local)
        """
        if self.verbose:
            print("=" * 80)
            print("🌐 Starting EXTERNAL analysis flow")
            print(f"📝 Content: {content[:100]}...")
            if media_urls:
                print(f"🖼️  Media: {len(media_urls)} URLs")
        
        # Run independent external analysis
        external_explanation = await self.external.analyze(content, media_urls)
        
        if self.verbose:
            print(f"✅ External explanation: {external_explanation[:100]}...")
        
        return external_explanation
    
    async def analyze_full(
        self,
        content: str,
        media_urls: Optional[List[str]] = None,
        admin_override: bool = False
    ) -> Tuple[str, str, Optional[str], AnalysisStages, dict]:
        """
        Run complete analysis flow (local + conditional external).
        
        External analysis is triggered when:
        - Category is NOT general AND NOT political_general
        - OR admin_override is True (admin-triggered analysis)
        
        Args:
            content: Text content to analyze
            media_urls: Optional media URLs
            admin_override: Force external analysis regardless of category (admin action)
        
        Returns:
            Tuple of (category, local_explanation, external_explanation, stages, pattern_data)
            where pattern_data contains pattern_matches and topic_classification
        """
        # Run local analysis
        category, local_explanation, stages, pattern_data = await self.analyze_local(content, media_urls)
        
        # Determine if external analysis should run
        should_run_external = (
            admin_override or 
            (category not in [Categories.GENERAL, Categories.POLITICAL_GENERAL])
        )
        
        # Run external analysis if conditions met
        external_explanation = None
        if should_run_external:
            if self.verbose:
                reason = "admin override" if admin_override else f"category {category}"
                print(f"\n🌐 External analysis triggered ({reason})")
            
            external_explanation = await self.analyze_external(content, media_urls)
            stages.external = True
        elif self.verbose:
            print(f"   ℹ️  Skipping external analysis for category: {category}")
        
        if self.verbose:
            print("\n" + "=" * 80)
            print(f"✅ Analysis complete:")
            print(f"   Category: {category}")
            print(f"   Local: {local_explanation[:100]}...")
            if external_explanation:
                print(f"   External: {external_explanation[:100]}...")
            print(f"   Stages: {stages.to_string()}")
        
        return category, local_explanation, external_explanation, stages, pattern_data
