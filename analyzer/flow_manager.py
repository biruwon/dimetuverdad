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
    category_detection: bool = False  # LLM category detection/validation
    media_analysis: bool = False      # Vision model media description
    explanation: bool = False          # Final explanation generation
    external: bool = False
    
    def to_string(self) -> str:
        """Convert to comma-separated string for database storage"""
        stages = []
        if self.pattern:
            stages.append("pattern")
        if self.category_detection:
            stages.append("category_detection")
        if self.media_analysis:
            stages.append("media_analysis")
        if self.explanation:
            stages.append("explanation")
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
            stages.category_detection = "category_detection" in parts
            stages.media_analysis = "media_analysis" in parts
            stages.explanation = "explanation" in parts
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
    media_description: Optional[str] = None
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
        
        # Text analyzer for category detection and explanation (gemma3:27b)
        self.text_llm = OllamaAnalyzer(model="gemma3:27b-it-q4_K_M", verbose=verbose, fast_mode=fast_mode)
        
        # Vision analyzer using same model (gemma3 supports multimodal!)
        self.vision_llm = OllamaAnalyzer(model="gemma3:27b-it-q4_K_M", verbose=verbose, fast_mode=fast_mode)
        
        self.external = ExternalAnalyzer(verbose=verbose)
        self.analyzer_hooks = create_analyzer_hooks(verbose=verbose)
        self.verbose = verbose
        self.analysis_count = 0  # Track number of analyses performed
        
        if self.verbose:
            print("🔄 AnalysisFlowManager initialized")
            print("   📝 Text LLM: gemma3:27b-it-q4_K_M")
            print("   🖼️  Vision LLM: gemma3:27b-it-q4_K_M (multimodal capable)")
    
    async def analyze_local(
        self,
        content: str,
        media_urls: Optional[List[str]] = None
    ) -> AnalysisResult:
        """
        Run local analysis flow with new optimized stages:
        1. Pattern Detection (fast, rule-based)
        2. Category Detection/Validation (text LLM, category only)
        3. Media Analysis (vision LLM, if media present)
        4. Explanation Generation (text LLM, with full context)
        
        Args:
            content: Text content to analyze
            media_urls: Optional media URLs for multimodal analysis
        
        Returns:
            AnalysisResult with category, explanation, media_description, stages, pattern_data, verification_data
        """
        import time
        
        stages = AnalysisStages()
        stage_timings = {}
        media_description = None
        
        # Reset model context before analysis (except for first analysis)
        if self.analysis_count > 0:
            if self.verbose:
                print(f"🔄 Resetting model context before analysis #{self.analysis_count + 1}")
            try:
                await self.text_llm.reset_model_context()
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Context reset failed: {e}")
        
        self.analysis_count += 1
        
        if self.verbose:
            print("=" * 80)
            print("🔄 Starting LOCAL analysis flow (NEW ARCHITECTURE)")
            print(f"📝 Content: {content[:100] if content else '(empty)'}...")
            if media_urls:
                print(f"🖼️  Media URLs: {len(media_urls)} items")
        
        # Handle empty content
        if not content or len(content.strip()) < 3:
            if self.verbose:
                print("⚠️  Empty or very short content detected")
            
            if media_urls and len(media_urls) > 0:
                # Media-only post - analyze images then explain
                if self.verbose:
                    print("🖼️  Media-only post - analyzing images")
                
                # Stage: Media Analysis
                start_time = time.time()
                media_description = await self.vision_llm.describe_media(media_urls)
                stage_timings['media_analysis'] = time.time() - start_time
                stages.media_analysis = True
                
                # Use media description as content for category detection
                start_time = time.time()
                primary_category = await self.text_llm.detect_category_only(media_description, None)
                stage_timings['category_detection'] = time.time() - start_time
                stages.category_detection = True
                
                # Generate explanation based on media
                start_time = time.time()
                local_explanation = await self.text_llm.generate_explanation_with_context("", primary_category, media_description)
                stage_timings['explanation'] = time.time() - start_time
                stages.explanation = True
                
                return AnalysisResult(
                    category=primary_category,
                    local_explanation=local_explanation,
                    media_description=media_description,
                    stages=stages,
                    pattern_data={},
                    verification_data={'stage_timings': stage_timings}
                )
            else:
                # Truly empty content
                return AnalysisResult(
                    category=Categories.GENERAL,
                    local_explanation="Contenido vacío o muy corto para analizar.",
                    media_description=None,
                    stages=stages,
                    pattern_data={},
                    verification_data={'stage_timings': stage_timings}
                )
        
        # STAGE 1: Pattern Detection
        if self.verbose:
            print("\n📊 Stage 1: Pattern Detection")
        
        start_time = time.time()
        pattern_result = self.pattern_analyzer.analyze_content(content)
        stage_timings['pattern_detection'] = time.time() - start_time
        stages.pattern = True
        
        # Extract pattern-suggested category
        pattern_category = None
        if pattern_result.categories and Categories.GENERAL not in pattern_result.categories:
            pattern_category = pattern_result.categories[0]
        
        if self.verbose:
            print(f"   Pattern category: {pattern_category or 'None'}")
            print(f"   Pattern matches: {len(pattern_result.pattern_matches)}")
            print(f"   ⏱️  Pattern detection: {stage_timings['pattern_detection']:.3f}s")
        
        # STAGE 2: Category Detection/Validation (Text LLM)
        if self.verbose:
            print("\n🤖 Stage 2: Category Detection (Text LLM)")
        
        start_time = time.time()
        primary_category = await self.text_llm.detect_category_only(content, pattern_category)
        stage_timings['category_detection'] = time.time() - start_time
        stages.category_detection = True
        
        if self.verbose:
            print(f"   Detected category: {primary_category}")
            print(f"   ⏱️  Category detection: {stage_timings['category_detection']:.3f}s")
        
        # STAGE 3: Media Analysis (Vision LLM, if media present)
        if media_urls and len(media_urls) > 0:
            if self.verbose:
                print("\n🖼️  Stage 3: Media Analysis (Vision LLM)")
            
            start_time = time.time()
            try:
                media_description = await self.vision_llm.describe_media(media_urls)
                stage_timings['media_analysis'] = time.time() - start_time
                stages.media_analysis = True
                
                if self.verbose:
                    print(f"   Media description: {media_description[:100]}...")
                    print(f"   ⏱️  Media analysis: {stage_timings['media_analysis']:.3f}s")
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️  Media analysis failed: {e}")
                # Continue without media description
                media_description = None
        
        # STAGE 4: Explanation Generation (Text LLM with full context)
        if self.verbose:
            print("\n💭 Stage 4: Explanation Generation (Text LLM)")
        
        start_time = time.time()
        local_explanation = await self.text_llm.generate_explanation_with_context(content, primary_category, media_description)
        stage_timings['explanation'] = time.time() - start_time
        stages.explanation = True
        
        if self.verbose:
            print(f"   Explanation: {local_explanation[:100]}...")
            print(f"   ⏱️  Explanation generation: {stage_timings['explanation']:.3f}s")
        
        # Check if LLM explanation indicates this should be disinformation
        if self.analyzer_hooks.explanation_indicates_disinformation(local_explanation):
            if self.verbose:
                print("   🔄 Explanation indicates disinformation - overriding category")
            primary_category = Categories.DISINFORMATION
        
        # STAGE 5: Verification (for disinformation category only)
        verification_data = {}
        if primary_category == Categories.DISINFORMATION:
            if self.verbose:
                print("\n🔍 Stage 5: Verification Feedback Enhancement")
            
            # Convert enum to string for verification hooks
            analyzer_result = {'category': primary_category.value if hasattr(primary_category, 'value') else primary_category, 'confidence': 0.8}
            should_trigger, reason = self.analyzer_hooks.should_trigger_verification(content, analyzer_result)
            
            if should_trigger:
                if self.verbose:
                    print(f"🔍 Verification triggered: {reason}")
                
                start_time = time.time()
                try:
                    analysis_result = await self.analyzer_hooks.analyze_with_verification(
                        content, 
                        original_result=analyzer_result
                    )
                    
                    stage_timings['verification'] = time.time() - start_time
                    verification_data = analysis_result.verification_data
                    
                    # Update explanation with verification context if contradictions found
                    if verification_data and verification_data.get('contradictions_detected'):
                        if self.verbose:
                            print("⚠️  Verification found contradictions")
                        local_explanation = analysis_result.explanation_with_verification
                    elif verification_data and verification_data.get('sources_cited'):
                        local_explanation = analysis_result.explanation_with_verification
                        
                except Exception as e:
                    stage_timings['verification'] = time.time() - start_time
                    if self.verbose:
                        print(f"⚠️  Verification failed: {e}")
        
        # Prepare pattern data
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
        
        if self.verbose:
            print(f"\n✅ Local analysis complete:")
            print(f"   Final category: {primary_category}")
            print(f"   Explanation: {local_explanation[:100]}...")
            if media_description:
                print(f"   Media: {media_description[:80]}...")
            total_time = sum(stage_timings.values())
            print(f"   ⏱️  Total time: {total_time:.3f}s")
        
        return AnalysisResult(
            category=primary_category,
            local_explanation=local_explanation,
            media_description=media_description,
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
            
            # Re-raise the exception to stop the analysis pipeline
            raise RuntimeError(f"Analysis failed: {str(e)}") from e
