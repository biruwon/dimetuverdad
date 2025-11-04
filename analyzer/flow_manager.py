"""
Analysis Flow Manager - Orchestrates the 3-stage pipeline.

Stage 1: Pattern Detection
Stage 2: Local LLM Analysis
Stage 3: External Analysis (Gemini, admin-triggered only)
"""

from typing import List, Optional
from dataclasses import dataclass
import time
from .pattern_analyzer import PatternAnalyzer
from .ollama_analyzer import OllamaAnalyzer
from .external_analyzer import ExternalAnalyzer, ExternalAnalysisResult
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


@dataclass
class AnalysisResult:
    """Complete analysis result with all data"""
    category: str
    local_explanation: str
    stages: AnalysisStages
    pattern_data: dict
    verification_data: dict
    external_explanation: Optional[str] = None


class AnalysisFlowManager:
    """
    Orchestrates the multi-stage content analysis pipeline.
    
    Flow:
    1. Try pattern detection first (fast, rule-based)
    2. If patterns insufficient → Local LLM analysis (gemma3:4b)
    3. External analysis only when explicitly triggered (admin action)
    """
    
    def __init__(self, verbose: bool = False, fast_mode: bool = False):
        """
        Initialize flow manager with all analyzers.
        
        Args:
            verbose: Enable detailed logging
            fast_mode: Use simplified prompts for faster bulk processing
        """
        # Initialize analyzers
        self.pattern_analyzer = PatternAnalyzer()
        self.local_llm = OllamaAnalyzer(verbose=verbose, fast_mode=fast_mode)
        self.external = ExternalAnalyzer(verbose=verbose)
        self.analyzer_hooks = create_analyzer_hooks(verbose=verbose)
        self.verbose = verbose
        
        if self.verbose:
            print("🔄 AnalysisFlowManager initialized")
    
    async def analyze_local(
        self,
        content: str,
        media_urls: Optional[List[str]] = None
    ) -> AnalysisResult:
        """
        Run local analysis flow (pattern + local LLM).
        
        Always runs both pattern detection and LLM analysis, then intelligently
        combines results based on pattern confidence.
        
        Args:
            content: Text content to analyze
            media_urls: Optional media URLs for multimodal analysis
        
        Returns:
            AnalysisResult with category, local_explanation, stages, pattern_data, and verification_data
        """
        import time
        
        stages = AnalysisStages()
        stage_timings = {}  # Track timing for each stage
        
        if self.verbose:
            print("=" * 80)
            print("🔄 Starting LOCAL analysis flow")
            print(f"📝 Content: {content[:100] if content else '(empty)'}...")
            if media_urls:
                print(f"🖼️  Media URLs: {len(media_urls)} items")
        
        # Handle empty content as special case
        if not content or len(content.strip()) < 3:
            if self.verbose:
                print("⚠️  Empty or very short content detected")
            
            # If we have media URLs but no text, run multimodal analysis
            if media_urls and len(media_urls) > 0:
                if self.verbose:
                    print("🖼️  Media-only post detected - running multimodal analysis")
                
                # Skip pattern detection (no text to analyze)
                # Go directly to LLM multimodal analysis
                start_time = time.time()
                primary_category, local_explanation = await self.local_llm.categorize_and_explain("", media_urls)
                stage_timings['llm_multimodal'] = time.time() - start_time
                stages.local_llm = True
                
                if self.verbose:
                    print(f"   ✅ Final category: {primary_category}")
                    print(f"   ✅ Local explanation: {local_explanation[:100]}...")
                
                return AnalysisResult(
                    category=primary_category,
                    local_explanation=local_explanation,
                    stages=stages,
                    pattern_data={},
                    verification_data={'stage_timings': stage_timings}
                )
            else:
                # No text and no media - truly empty content
                return AnalysisResult(
                    category=Categories.GENERAL,
                    local_explanation="Contenido vacío o muy corto para analizar.",
                    stages=stages,
                    pattern_data={},
                    verification_data={'stage_timings': stage_timings}
                )
        
        # Stage 1: Pattern Detection
        if self.verbose:
            print("\n📊 Stage 1: Pattern Detection")
        
        start_time = time.time()
        pattern_result = self.pattern_analyzer.analyze_content(content)
        stage_timings['pattern_detection'] = time.time() - start_time
        stages.pattern = True
        
        if self.verbose:
            print(f"   Categories found: {pattern_result.categories}")
            print(f"   Pattern matches: {len(pattern_result.pattern_matches)}")
            print(f"   ⏱️  Pattern detection: {stage_timings['pattern_detection']:.3f}s")
        
        # Stage 2: Local LLM Analysis (always run)
        if self.verbose:
            print("\n🤖 Stage 2: Local LLM Analysis")
        
        # Determine if we need categorization or just explanation
        patterns_found_specific_category = (
            pattern_result.categories and 
            Categories.GENERAL not in pattern_result.categories
        )
        
        start_time = time.time()
        if patterns_found_specific_category:
            # Patterns found a specific category - use LLM for explanation only
            primary_category = pattern_result.categories[0]
            if self.verbose:
                print(f"   Using LLM for explanation of pattern-detected category: {primary_category}")
            
            local_explanation = await self.local_llm.explain_only(content, primary_category, media_urls)
        else:
            # No specific patterns found - use LLM for both categorization and explanation
            if self.verbose:
                print("   Using LLM for full categorization + explanation")
            
            primary_category, local_explanation = await self.local_llm.categorize_and_explain(content, media_urls)
        
        stage_timings['llm_analysis'] = time.time() - start_time
        stages.local_llm = True
        
        if self.verbose:
            print(f"   ✅ Final category: {primary_category}")
            print(f"   ✅ Local explanation: {local_explanation[:100]}...")
            print(f"   ⏱️  LLM analysis: {stage_timings['llm_analysis']:.3f}s")
        
        # Check if LLM explanation indicates this should be disinformation
        if self.analyzer_hooks.explanation_indicates_disinformation(local_explanation):
            if self.verbose:
                print("   🔄 LLM explanation indicates disinformation - overriding category")
            primary_category = Categories.DISINFORMATION
        
        # Apply verification feedback enhancement
        if self.verbose:
            print("\n🔍 Phase 2: Verification Feedback Enhancement")
        
        # Run verification only for disinformation category (most critical for fact-checking)
        if primary_category == Categories.DISINFORMATION:
            # Check if verification should be triggered
            analyzer_result = {'category': primary_category, 'confidence': 0.8}
            should_trigger, reason = self.analyzer_hooks.should_trigger_verification(content, analyzer_result)
            
            if should_trigger:
                if self.verbose:
                    print(f"🔍 Verification triggered: {reason}")
                
                start_time = time.time()
                try:
                    # Run verification
                    analysis_result = await self.analyzer_hooks.analyze_with_verification(
                        content, 
                        original_result=analyzer_result
                    )
                    
                    stage_timings['verification'] = time.time() - start_time
                    
                    verification_data = analysis_result.verification_data
                    
                    # Check if verification found contradictions
                    if verification_data and verification_data.get('contradictions_detected'):
                        if self.verbose:
                            print("⚠️  Verification found contradictions - updating category to disinformation")
                        
                        # Update category to disinformation
                        primary_category = Categories.DISINFORMATION
                        local_explanation = analysis_result.explanation_with_verification
                    
                    # No contradictions found, but add verification context if sources were checked
                    elif verification_data and verification_data.get('sources_cited'):
                        local_explanation = analysis_result.explanation_with_verification
                        verification_data = analysis_result.verification_data
                    
                except Exception as e:
                    stage_timings['verification'] = time.time() - start_time
                    if self.verbose:
                        print(f"⚠️  Verification failed: {e}")
                    verification_data = {}
            else:
                verification_data = {}
        else:
            verification_data = {}
        
        if self.verbose:
            print(f"   ✅ Final category after verification: {primary_category}")
            print(f"   ✅ Enhanced explanation: {local_explanation[:100]}...")
            if 'verification' in stage_timings:
                print(f"   ⏱️  Verification: {stage_timings['verification']:.3f}s")
            else:
                print("   ⏱️  Verification: skipped")
        
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
        
        # Add stage timings to verification data
        if verification_data:
            verification_data['stage_timings'] = stage_timings
        else:
            verification_data = {'stage_timings': stage_timings}
        
        return AnalysisResult(
            category=primary_category,
            local_explanation=local_explanation,
            stages=stages,
            pattern_data=pattern_data,
            verification_data=verification_data
        )
    
    async def analyze_external(
        self,
        content: str,
        media_urls: Optional[List[str]] = None
    ) -> ExternalAnalysisResult:
        """
        Run external analysis (Gemini).
        
        This is a separate flow, triggered only by admin action.
        Does NOT depend on local analysis results.
        
        Args:
            content: Text content to analyze
            media_urls: Optional media URLs for multimodal analysis
        
        Returns:
            ExternalAnalysisResult with category and explanation (independent from local)
        """        
        if self.verbose:
            print("=" * 80)
            print("🌐 Starting EXTERNAL analysis flow")
            print(f"📝 Content: {content[:300]}...")
            if media_urls:
                print(f"🖼️  Media: {len(media_urls)} URLs")
        
        start_time = time.time()
        # Run independent external analysis
        external_result = await self.external.analyze(content, media_urls)
        external_timing = time.time() - start_time
        
        if self.verbose:
            print(f"✅ External analysis complete: {external_result.category} - {external_result.explanation[:300]}...")
            print(f"   ⏱️  External analysis: {external_timing:.3f}s")
        
        # Add timing to the result if it has a dict structure
        if hasattr(external_result, 'timing'):
            external_result.timing = external_timing
        elif hasattr(external_result, '__dict__'):
            external_result.__dict__['timing'] = external_timing
        
        return external_result
    
    async def analyze_full(
        self,
        content: str,
        media_urls: Optional[List[str]] = None,
        admin_override: bool = False,
        force_disable_external: bool = False
    ) -> AnalysisResult:
        """
        Run complete analysis flow (local + conditional external).
        
        External analysis is triggered when:
        - Category is NOT general AND NOT political_general
        - OR admin_override is True (admin-triggered analysis)
        
        External analysis can override the local category if it detects a different category.
        For disinformation detection, external analysis takes precedence over local analysis.
        
        Args:
            content: Text content to analyze
            media_urls: Optional media URLs
            admin_override: Force external analysis regardless of category (admin action)
        
        Returns:
            AnalysisResult with category, local_explanation, external_explanation, stages, pattern_data, and verification_data
            where pattern_data contains pattern_matches and topic_classification
            and verification_data contains verification results if triggered
        """
        try:
            # Run local analysis
            local_result = await self.analyze_local(content, media_urls)
            
            # If local analysis failed, return error result
            if local_result is None:
                if self.verbose:
                    print("❌ Local analysis returned None, returning error result")
                return AnalysisResult(
                    category="ERROR",
                    local_explanation="Local analysis failed: returned None",
                    stages=AnalysisStages(),
                    pattern_data={},
                    verification_data={}
                )
            
            # Determine if external analysis should run
            should_run_external = (
                not force_disable_external and
                (admin_override or 
                 local_result.category not in [Categories.GENERAL, Categories.POLITICAL_GENERAL])
            )
            
            # Run external analysis if conditions met
            if should_run_external:
                if self.verbose:
                    reason = "admin override" if admin_override else f"category {local_result.category}"
                    print(f"\n🌐 External analysis triggered ({reason})")
                
                try:
                    external_result = await self.analyze_external(content, media_urls)
                    
                    # Normal case: mark external stage as run
                    local_result.stages.external = True
                    local_result.external_explanation = external_result.explanation
                    
                    # Check if external analysis detects a different category - if so, override local category
                    if external_result.category and external_result.category != local_result.category:
                        if self.verbose:
                            print(f"🔄 External analysis detected different category: {external_result.category} (was {local_result.category})")
                        local_result.category = external_result.category
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ External analysis failed: {e}")
                    # Continue with local result only
                    pass
            elif self.verbose and local_result:
                print(f"   ℹ️  Skipping external analysis for category: {local_result.category}")
            
            if self.verbose and local_result:
                print("\n" + "=" * 80)
                print(f"✅ Analysis complete:")
                print(f"   Category: {local_result.category}")
                print(f"   Local: {local_result.local_explanation[:300]}...")
                if local_result.external_explanation:
                    print(f"   External: {local_result.external_explanation[:300]}...")
                print(f"   Stages: {local_result.stages.to_string()}")
            
            return local_result
            
        except Exception as e:
            if self.verbose:
                print(f"💥 Critical error in analyze_full: {e}")
                import traceback
                traceback.print_exc()
            
            # Return error result instead of None
            return AnalysisResult(
                category="ERROR",
                local_explanation=f"Analysis failed: {str(e)}",
                stages=AnalysisStages(),
                pattern_data={},
                verification_data={}
            )
