#!/usr/bin/env python3
"""
Gemini Multimodal Analysis Module for dimetuverdad.

Provides unified analysis for images and videos using Google Gemini 2.5 Flash.
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
                print(f"  ❌ Media not found (404): {media_url}")
                print("     This URL may have expired or the media was deleted")
                return None
            elif response.status_code == 403:
                print(f"  ❌ Access forbidden (403): {media_url}")
                print("     This media may be restricted or require authentication")
                return None
            elif response.status_code == 410:
                print(f"  ❌ Media gone (410): {media_url}")
                print("     This media has been permanently removed")
                return None
            elif response.status_code >= 500:
                print(f"  ❌ Server error ({response.status_code}): {media_url}")
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
                            print(f"{progress:.1f}%")

            # Verify file was downloaded and has content
            file_size = os.path.getsize(temp_file.name)
            if file_size == 0:
                print(f"  ❌ Downloaded file is empty ({file_size} bytes)")
                os.unlink(temp_file.name)
                continue

            print(f"  ✅ Successfully downloaded: {temp_file.name} ({file_size} bytes)")
            return temp_file.name

        except requests.exceptions.Timeout:
            print(f"  ⏰ Timeout on attempt {attempt + 1}: {media_url}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"     Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print("     All timeout attempts failed")
        except requests.exceptions.ConnectionError as e:
            print(f"  🌐 Connection error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"     Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print("     All connection attempts failed")
        except requests.exceptions.HTTPError as e:
            print(f"  🔴 HTTP error on attempt {attempt + 1}: {e}")
            # Don't retry on HTTP errors (4xx, 5xx) as they're likely permanent
            return None
        except Exception as e:
            print(f"  💥 Unexpected error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"     Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print("     All attempts failed")

    print(f"❌ Failed to download media after {max_retries} attempts: {media_url}")
    return None

def _get_gemini_client() -> Optional[genai.GenerativeModel]:
    """Get initialized Gemini client with API key."""
    # Try GEMINI_API_KEY first, then fall back to GOOGLE_API_KEY for compatibility
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        print("Error: Neither GEMINI_API_KEY nor GOOGLE_API_KEY found in environment variables")
        return None
    
    try:
        genai.configure(api_key=api_key)
        # Return a model instance - using gemini-2.0-flash-exp (experimental 2.0 model)
        return genai.GenerativeModel('gemini-2.0-flash-exp')
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        return None

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
    Analyze multimodal content (images/videos + text) using Gemini 2.5 Flash.

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

    # Get Gemini client
    model = _get_gemini_client()
    if not model:
        return None, time.time() - start_time

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

    media_path = None
    try:
        # Download media
        media_path = download_media_to_temp_file(media_url, is_video)
        if not media_path:
            return None, time.time() - start_time

        try:
            # Upload to Gemini
            media_file = _upload_media_to_gemini(model, media_path, media_url)
            if not media_file:
                return None, time.time() - start_time

            # Create analysis prompt using centralized prompt generator
            prompt = EnhancedPromptGenerator.build_gemini_analysis_prompt(text_content, is_video)

            # Generate analysis with timeout
            print("Analyzing with Gemini 2.5 Flash...")
            
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
                
                analysis_time = time.time() - start_time
                print(f"Gemini analysis completed in {analysis_time:.2f}s")

                return response.text, analysis_time
                
            except TimeoutError:
                print("Gemini analysis timed out after 30 seconds")
                return None, time.time() - start_time
            except Exception as api_error:
                print(f"Gemini API error: {api_error}")
                signal.alarm(0)  # Cancel timeout on other errors too
                return None, time.time() - start_time

        finally:
            # Clean up temporary file after analysis
            if media_path and os.path.exists(media_path):
                try:
                    os.unlink(media_path)
                    print(f"Cleaned up temporary media file: {media_path}")
                except Exception as e:
                    print(f"Warning: Could not clean up temporary file {media_path}: {e}")

    except Exception as e:
        error_time = time.time() - start_time
        print(f"Error in multimodal analysis after {error_time:.2f}s: {e}")
        return None, error_time

def extract_media_type(media_urls: List[str]) -> str:
    """
    Extract media type from URLs.

    Args:
        media_urls: List of media URLs

    Returns:
        Media type string: "image", "video", or "mixed"
    """
    if not media_urls:
        return ""

    has_image = any('pbs.twimg.com' in url or 'card_img' in url for url in media_urls)
    has_video = any('video.twimg.com' in url for url in media_urls)

    if has_video and has_image:
        return "mixed"
    elif has_video:
        return "video"
    elif has_image:
        return "image"
    else:
        return "unknown"