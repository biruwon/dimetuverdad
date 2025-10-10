#!/usr/bin/env python3
"""
Gemini Multimodal Analysis Module for dimetuverdad.

Provides unified analysis for images and video            # Verify file was downloaded and has content
            file_size = os.path.getsize(temp_file.name)
            if file_size == 0:
                print(f"\r  ❌ Downloaded file is empty ({file_size} bytes)")
                os.unlink(temp_file.name)
                continue

            print(f"\r  ✅ Successfully downloaded: {temp_file.name} ({file_size} bytes)")Google Gemini 2.5 Flash.
Handles media download, upload to Gemini, and comprehensive political content analysis.
"""

import os
import tempfile
import time
import requests
from typing import Optional, Tuple, List

# Simple warning suppression for Google Cloud libraries
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

import google.generativeai as genai
from dotenv import load_dotenv

# Import prompt generation
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyzer.prompts import EnhancedPromptGenerator

# Load environment variables
load_dotenv()

def extract_media_type(media_urls: List[str]) -> str:
    """
    Extract media type from a list of media URLs.
    
    Args:
        media_urls: List of media URLs to analyze
        
    Returns:
        Media type: "image", "video", "mixed", "unknown", or "" for empty list
    """
    if not media_urls:
        return ""
    
    has_image = False
    has_video = False
    
    for url in media_urls:
        url_lower = url.lower()
        # Check for video extensions or video in URL
        if any(ext in url_lower for ext in ['.mp4', '.m3u8', '.mov', '.avi', '.webm']):
            has_video = True
        elif 'video' in url_lower:
            has_video = True
        # Check for image extensions or format parameters
        elif any(ext in url_lower for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
            has_image = True
        elif 'format=jpg' in url_lower or 'format=jpeg' in url_lower or 'format=png' in url_lower:
            has_image = True
    
    if has_image and has_video:
        return "mixed"
    elif has_video:
        return "video"
    elif has_image:
        return "image"
    else:
        return "unknown"

def download_media_to_temp_file(media_url: str, is_video: bool = False, max_retries: int = 3) -> Optional[str]:
    """
    Download media from URL to a temporary file for analysis.

    This function downloads media directly from Twitter URLs during analysis.
    If download fails, it provides detailed error information.

    Args:
        media_url: URL of the media to download
        is_video: Whether the media is a video (affects file extension)
        max_retries: Maximum number of download attempts

    Returns:
        Path to the temporary file, or None if download failed
    """
    print(f"🔗 Attempting to download media: {media_url}")

    for attempt in range(max_retries):
        try:
            # Add headers to mimic a browser and handle Twitter's requirements
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8,video/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'image',
                'Sec-Fetch-Mode': 'no-cors',
                'Sec-Fetch-Site': 'cross-site',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }

            print(f"  📡 Download attempt {attempt + 1}/{max_retries}...")
            response = requests.get(media_url, stream=True, timeout=30, headers=headers, allow_redirects=True)

            # Check for common Twitter error responses
            if response.status_code == 404:
                print(f"\r  ❌ Media not found (404): {media_url}")
                print("     This URL may have expired or the media was deleted")
                return None
            elif response.status_code == 403:
                print(f"\r  ❌ Access forbidden (403): {media_url}")
                print("     This media may be restricted or require authentication")
                return None
            elif response.status_code == 410:
                print(f"\r  ❌ Media gone (410): {media_url}")
                print("     This media has been permanently removed")
                return None
            elif response.status_code >= 500:
                print(f"\r  ❌ Server error ({response.status_code}): {media_url}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"     Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return None

            response.raise_for_status()

            # Determine file extension based on content type or URL
            content_type = response.headers.get('content-type', '').lower()
            if is_video or 'video' in content_type or 'mp4' in media_url.lower():
                suffix = '.mp4'
            elif 'image' in content_type:
                if 'png' in content_type:
                    suffix = '.png'
                elif 'gif' in content_type:
                    suffix = '.gif'
                else:
                    suffix = '.jpg'
            else:
                # Fallback based on URL
                if 'mp4' in media_url.lower() or 'video' in media_url.lower():
                    suffix = '.mp4'
                else:
                    suffix = '.jpg'

            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

            # Download with progress indication for large files
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(temp_file.name, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 1024 * 1024:  # Show progress for files > 1MB
                            progress = (downloaded / total_size) * 100 if total_size > 0 else 0
                            print(f"\rDownloading... {progress:.1f}%", end='', flush=True)

            # Verify file was downloaded and has content
            file_size = os.path.getsize(temp_file.name)
            if file_size == 0:
                print(f"  ❌ Downloaded file is empty ({file_size} bytes)")
                os.unlink(temp_file.name)
                continue

            print(f"\r  ✅ Successfully downloaded: {temp_file.name} ({file_size} bytes)")
            return temp_file.name

        except requests.exceptions.Timeout:
            print(f"\r  ⏰ Timeout on attempt {attempt + 1}: {media_url}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"     Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print("     All timeout attempts failed")
        except requests.exceptions.ConnectionError as e:
            print(f"\r  🌐 Connection error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"     Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print("     All connection attempts failed")
        except requests.exceptions.HTTPError as e:
            print(f"\r  🔴 HTTP error on attempt {attempt + 1}: {e}")
            # Don't retry on HTTP errors (4xx, 5xx) as they're likely permanent
            return None
        except Exception as e:
            print(f"\r  💥 Unexpected error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"     Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print("     All attempts failed")

    print(f"\r❌ Failed to download media after {max_retries} attempts: {media_url}")
    return None

def _get_gemini_client(model_name: str = 'gemini-2.0-flash-exp') -> Tuple[Optional[genai.GenerativeModel], Optional[str]]:
    """
    Get initialized Gemini client with specified model.
    
    Args:
        model_name: Name of the Gemini model to use
        
    Returns:
        Tuple of (model_instance, error_message)
        Returns (None, error_message) if initialization failed
    """
    # Try GEMINI_API_KEY first, then fall back to GOOGLE_API_KEY for compatibility
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        error_msg = "Neither GEMINI_API_KEY nor GOOGLE_API_KEY found in environment variables"
        print(f"Error: {error_msg}")
        return None, error_msg
    
    try:
        genai.configure(api_key=api_key)
        # Return a model instance with the specified model name
        model = genai.GenerativeModel(model_name)
        return model, None
    except Exception as e:
        error_msg = f"Error initializing Gemini client with {model_name}: {e}"
        print(error_msg)
        return None, error_msg

def _upload_media_to_gemini(client: genai.GenerativeModel, media_path: str, media_url: str) -> Optional[genai.types.File]:
    """
    Upload media file to Gemini.

    Args:
        client: Initialized Gemini model
        media_path: Local path to media file
        media_url: Original URL (for logging only)

    Returns:
        Uploaded Gemini file object, or None if upload failed
    """
    try:
        print("Uploading media to Gemini...")
        # For google.generativeai, we upload the file
        media_file = genai.upload_file(media_path)
        
        # Wait for processing to complete with timeout
        print("Waiting for media processing...")
        max_wait_time = 60  # 60 second timeout for media processing
        wait_start = time.time()
        
        while media_file.state.name == "PROCESSING" and (time.time() - wait_start) < max_wait_time:
            time.sleep(1)
            media_file = genai.get_file(media_file.name)

        if media_file.state.name == "PROCESSING":
            print(f"Media processing timed out after {max_wait_time} seconds")
            return None
        elif media_file.state.name == "FAILED":
            print(f"Media processing failed: {media_file.state.name}")
            return None

        print("Media uploaded and processed successfully")
        return media_file

    except Exception as e:
        print(f"Error uploading media to Gemini: {e}")
        return None

def analyze_multimodal_content(media_urls: List[str], text_content: str) -> Tuple[Optional[str], float]:
    """
    Analyze multimodal content (images/videos + text) using Gemini models with fallback.

    Tries multiple Gemini models in priority order, only retrying on recoverable errors
    (quota exceeded or model not available). Falls back to text-only analysis on unrecoverable errors.

    Args:
        media_urls: List of media URLs to analyze
        text_content: Text content accompanying the media

    Returns:
        Tuple of (analysis_result, analysis_time_seconds)
        Returns (None, error_time) if analysis failed
    """
    start_time = time.time()

    if not media_urls:
        print("No media URLs provided")
        return None, 0.0

    # Model priority order - only retry on quota exceeded or model not available
    MODEL_PRIORITY = [
        'gemini-2.0-flash-exp',    # Current experimental model
        'gemini-2.0-flash-lite',   # Lighter 2.0 model
        'gemini-2.0-flash',        # Standard 2.0 model
        'gemini-2.5-flash-lite',   # Lighter 2.5 model
        'gemini-2.5-flash',        # Standard 2.5 model
        'gemini-2.5-pro'           # Most capable model
    ]

    # Filter out unwanted media (profile images, card previews)
    filtered_urls = []
    for url in media_urls:
        if 'profile_images' in url or 'card_img' in url:
            continue
        filtered_urls.append(url)

    if not filtered_urls:
        print("No valid media URLs found after filtering")
        return None, time.time() - start_time

    # Process the first media URL (for now, we handle one media per analysis)
    # But prioritize video URLs over image URLs if available
    has_video = any('video' in url.lower() or '.mp4' in url.lower() or '.m3u8' in url.lower() for url in filtered_urls)

    # Priority order: MP4 > M3U8 > other video URLs > images
    if has_video:
        # Find MP4 files first (highest priority)
        mp4_url = next((url for url in filtered_urls if '.mp4' in url.lower()), None)
        if mp4_url:
            media_url = mp4_url
        else:
            # No MP4, check for M3U8
            m3u8_url = next((url for url in filtered_urls if '.m3u8' in url.lower()), None)
            if m3u8_url:
                media_url = m3u8_url
            else:
                # No MP4/M3U8, use first video URL
                media_url = next((url for url in filtered_urls if 'video' in url.lower()), filtered_urls[0])
    else:
        media_url = filtered_urls[0]

    is_video = has_video

    # Try each model in priority order
    for model_name in MODEL_PRIORITY:
        model_start_time = time.time()
        print(f"🔄 Trying Gemini model: {model_name}")

        # Get Gemini client for this model
        model, client_error = _get_gemini_client(model_name)
        if not model:
            print(f"❌ Failed to initialize {model_name}: {client_error}")

            # Check if this is a recoverable error (quota or model availability)
            if _is_recoverable_error(client_error):
                print(f"🔄 Error is recoverable, trying next model...")
                continue
            else:
                print(f"💥 Unrecoverable error, falling back to text-only analysis")
                return _fallback_to_text_only(text_content), time.time() - start_time

        media_path = None
        try:
            # Download media
            media_path = download_media_to_temp_file(media_url, is_video)
            if not media_path:
                print(f"❌ Failed to download media for {model_name}")
                continue

            try:
                # Upload to Gemini
                media_file = _upload_media_to_gemini(model, media_path, media_url)
                if not media_file:
                    print(f"❌ Failed to upload media to {model_name}")
                    continue

                # Create analysis prompt using centralized prompt generator
                prompt = EnhancedPromptGenerator.build_gemini_analysis_prompt(text_content, is_video)

                # Generate analysis with timeout
                print(f"🧠 Analyzing with {model_name}...")

                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError("Gemini API call timed out")

                # Set up timeout signal
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)

                try:
                    response = model.generate_content([media_file, prompt])

                    # Cancel the timeout
                    signal.alarm(0)

                    analysis_time = time.time() - model_start_time
                    print(f"✅ {model_name} analysis completed in {analysis_time:.2f}s")

                    return response.text, time.time() - start_time

                except TimeoutError:
                    print(f"⏰ {model_name} analysis timed out after 60 seconds")
                    signal.alarm(0)
                    continue
                except Exception as api_error:
                    error_str = str(api_error).lower()
                    print(f"❌ {model_name} API error: {api_error}")

                    # Check if this is a recoverable error
                    if _is_recoverable_error(error_str):
                        print(f"🔄 API error is recoverable, trying next model...")
                        signal.alarm(0)
                        continue
                    else:
                        print(f"💥 Unrecoverable API error, falling back to text-only analysis")
                        signal.alarm(0)
                        return _fallback_to_text_only(text_content), time.time() - start_time

            finally:
                # Clean up temporary file after analysis attempt
                if media_path and os.path.exists(media_path):
                    try:
                        os.unlink(media_path)
                        print(f"🧹 Cleaned up temporary media file: {media_path}")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not clean up temporary file {media_path}: {e}")

        except Exception as e:
            model_error_time = time.time() - model_start_time
            print(f"❌ Error with {model_name} after {model_error_time:.2f}s: {e}")

            # Check if this is a recoverable error
            if _is_recoverable_error(str(e)):
                print(f"🔄 Model error is recoverable, trying next model...")
                continue
            else:
                print(f"💥 Unrecoverable model error, falling back to text-only analysis")
                return _fallback_to_text_only(text_content), time.time() - start_time

    # All models failed with recoverable errors
    print(f"💔 All {len(MODEL_PRIORITY)} Gemini models failed, falling back to text-only analysis")
    return _fallback_to_text_only(text_content), time.time() - start_time

def _is_recoverable_error(error_message: str) -> bool:
    """
    Check if an error is recoverable (should try next model) or unrecoverable (should fall back to text-only).

    Recoverable errors: quota exceeded, model not available
    Unrecoverable errors: authentication, network, invalid requests, etc.

    Args:
        error_message: Error message string to check

    Returns:
        True if error is recoverable, False if unrecoverable
    """
    if not error_message:
        return False

    error_lower = error_message.lower()

    # Recoverable errors - try next model
    recoverable_patterns = [
        'quota exceeded',
        'quota_exceeded',
        'rate limit',
        'rate_limit',
        'model not available',
        'model_not_available',
        'model not found',
        'model_not_found',
        'resource exhausted',
        'resource_exhausted'
    ]

    return any(pattern in error_lower for pattern in recoverable_patterns)


def _fallback_to_text_only(text_content: str) -> Optional[str]:
    """
    Fallback to text-only analysis when multimodal analysis fails.

    Args:
        text_content: Text content to analyze

    Returns:
        Text-only analysis result, or None if that also fails
    """
    print("📝 Falling back to text-only analysis...")

    try:
        # Import here to avoid circular imports
        from analyzer.analyzer import create_analyzer

        analyzer = create_analyzer()
        result = analyzer.analyze_content(text_content, media_urls=[])

        if result and hasattr(result, 'llm_explanation') and result.llm_explanation:
            print("✅ Text-only analysis completed")
            return result.llm_explanation
        else:
            print("❌ Text-only analysis failed to produce result")
            return None

    except Exception as e:
        print(f"❌ Text-only analysis failed: {e}")
        return None