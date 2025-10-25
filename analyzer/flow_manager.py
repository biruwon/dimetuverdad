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
from retrieval.integration.analyzer_hooks import create_analyzer_hooks


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
        self.analyzer_hooks = create_analyzer_hooks(verbose=verbose)
        self.verbose = verbose
        
        if self.verbose:
            print("🔄 AnalysisFlowManager initialized")
    
    async def analyze_local(
        self,
        content: str,
        media_urls: Optional[List[str]] = None
    ) -> Tuple[str, str, AnalysisStages, dict, dict]:
        """
        Run local analysis flow (pattern + local LLM).
        
        Args:
            content: Text content to analyze
            media_urls: Optional media URLs (not used in local flow)
        
        Returns:
            Tuple of (category, local_explanation, stages, pattern_data, verification_data)
            where pattern_data contains pattern_matches and topic_classification
            and verification_data contains verification results if triggered
        """
        stages = AnalysisStages()
        
        if self.verbose:
            print("=" * 80)
            print("🔄 Starting LOCAL analysis flow")
            print(f"📝 Content: {content[:100]}...")
        
        # Handle empty content as special case
        if not content or len(content.strip()) < 3:
            if self.verbose:
                print("⚠️  Empty or very short content detected")
            return Categories.GENERAL, "Contenido vacío o muy corto para analizar.", stages, {}, {}
        
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
            # Patterns found a specific category (not general or political_general)
            primary_category = pattern_result.categories[0]
            
            if self.verbose:
                print(f"   ✅ Pattern detection succeeded: {primary_category}")
                print(f"\n🤖 Stage 2: Local LLM (explanation only)")
            
            # Stage 2: Get explanation from local LLM
            local_explanation = await self.local_llm.explain_only(content, primary_category)
            stages.local_llm = True
            
            if self.verbose:
                print(f"   ✅ Local explanation: {local_explanation[:100]}...")
            
            # Check if LLM explanation indicates this should be disinformation
            if self.analyzer_hooks.explanation_indicates_disinformation(local_explanation):
                if self.verbose:
                    print("   🔄 LLM explanation indicates disinformation - overriding pattern category")
                primary_category = Categories.DISINFORMATION
            
            # Phase 2: Verification feedback enhancement
            if self.verbose:
                print("\n🔍 Phase 2: Verification Feedback Enhancement")
            
            primary_category, local_explanation, verification_data = await self._enhance_with_verification_feedback(
                content, primary_category, local_explanation, stages
            )
            
            if self.verbose:
                print(f"   ✅ Final category after verification: {primary_category}")
                print(f"   ✅ Enhanced explanation: {local_explanation[:100]}...")
            
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
            
            return primary_category, local_explanation, stages, pattern_data, verification_data
        
        # Stage 2: Pattern detection failed or returned general → Use local LLM for both
        if self.verbose:
            print("   ⚠️  Pattern detection insufficient")
            print("\n🤖 Stage 2: Local LLM (categorization + explanation)")
        
        category, local_explanation = await self.local_llm.categorize_and_explain(content)
        stages.local_llm = True
        
        if self.verbose:
            print(f"   ✅ Local LLM category: {category}")
            print(f"   ✅ Local LLM explanation: {local_explanation[:100]}...")
        
        # Phase 2: Verification feedback enhancement
        if self.verbose:
            print("\n🔍 Phase 2: Verification Feedback Enhancement")
        
        category, local_explanation, verification_data = await self._enhance_with_verification_feedback(
            content, category, local_explanation, stages
        )
        
        if self.verbose:
            print(f"   ✅ Final category after verification: {category}")
            print(f"   ✅ Enhanced explanation: {local_explanation[:100]}...")
        
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
        
        return category, local_explanation, stages, pattern_data, verification_data
    
    async def _enhance_with_verification_feedback(
        self,
        content: str,
        initial_category: str,
        initial_explanation: str,
        stages: AnalysisStages
    ) -> Tuple[str, str, dict]:
        """
        Check for verifiable claims and enhance categorization with verification feedback.
        
        Args:
            content: Original content
            initial_category: Category from initial analysis
            initial_explanation: Explanation from initial analysis
            stages: Analysis stages tracking
            
        Returns:
            Tuple of (updated_category, enhanced_explanation, verification_data)
        """
        # Skip verification for categories that don't need it
        if initial_category in [Categories.GENERAL]:
            return initial_category, initial_explanation, {}
        
        # Check if verification should be triggered
        analyzer_result = {'category': initial_category, 'confidence': 0.8}
        should_trigger, reason = self.analyzer_hooks.should_trigger_verification(content, analyzer_result)
        
        if not should_trigger:
            return initial_category, initial_explanation, {}
        
        if self.verbose:
            print(f"🔍 Verification triggered: {reason}")
        
        try:
            # Run verification
            analysis_result = await self.analyzer_hooks.analyze_with_verification(
                content, 
                original_result=analyzer_result
            )
            
            verification_data = analysis_result.verification_data
            
            # Check if verification found contradictions
            if verification_data and verification_data.get('contradictions_detected'):
                if self.verbose:
                    print("⚠️  Verification found contradictions - updating category to disinformation")
                
                # Update category to disinformation
                updated_category = Categories.DISINFORMATION
                
                # Use the already enhanced explanation from analyzer_hooks
                return updated_category, analysis_result.explanation_with_verification, verification_data
            
            # No contradictions found, but add verification context if sources were checked
            elif verification_data and verification_data.get('sources_cited'):
                # Use the already enhanced explanation from analyzer_hooks
                return initial_category, analysis_result.explanation_with_verification, verification_data
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Verification failed: {e}")
            # Continue with original results if verification fails
        
        return initial_category, initial_explanation, {}
    
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
        admin_override: bool = False,
        force_disable_external: bool = False
    ) -> Tuple[str, str, Optional[str], AnalysisStages, dict, dict]:
        """
        Run complete analysis flow (local + conditional external).
        
        External analysis is triggered when:
        - Category is NOT general AND NOT political_general
        - OR admin_override is True (admin-triggered analysis)
        
        For disinformation detection, external analysis takes precedence over local analysis.
        
        Args:
            content: Text content to analyze
            media_urls: Optional media URLs
            admin_override: Force external analysis regardless of category (admin action)
        
        Returns:
            Tuple of (category, local_explanation, external_explanation, stages, pattern_data, verification_data)
            where pattern_data contains pattern_matches and topic_classification
            and verification_data contains verification results if triggered
        """
        # Run local analysis
        category, local_explanation, stages, pattern_data, verification_data = await self.analyze_local(content, media_urls)
        
        # Determine if external analysis should run
        should_run_external = (
            not force_disable_external and
            (admin_override or category not in [Categories.GENERAL, Categories.POLITICAL_GENERAL])
        )
        
        # Run external analysis if conditions met
        external_explanation = None
        if should_run_external:
            if self.verbose:
                reason = "admin override" if admin_override else f"category {category}"
                print(f"\n🌐 External analysis triggered ({reason})")
            
            external_explanation = await self.analyze_external(content, media_urls)
            stages.external = True
            
            # Check if external analysis detects disinformation - if so, override local category
            if self.analyzer_hooks.external_analysis_indicates_disinformation(external_explanation):
                if self.verbose:
                    print("🔄 External analysis detected disinformation - overriding local category")
                category = Categories.DISINFORMATION
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
        
        return category, local_explanation, external_explanation, stages, pattern_data, verification_data
