#!/usr/bin/env python3
"""
Quick test tool for analyzer without loading full models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from analyzer.analyzer import Analyzer
from analyzer.pattern_analyzer import PatternAnalyzer
from analyzer.categories import Categories
import time
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Quickly analyze text content with the Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_test.py "Los chemtrails son reales"                          # Fast mode (patterns only)
  python quick_test.py --llm "Mi vecino dice que las vacunas..."            # Full mode (LLM + patterns)
  python quick_test.py --file input.txt --llm                               # Analyze file with LLM
  python quick_test.py --interactive                                         # Interactive mode

Modes:
  Default: Fast pattern-based analysis only (~1-2 seconds)
  --llm:   Full analysis with LLM when needed (~30+ seconds)

Categories:
  🚫 hate_speech: Direct hate speech, slurs, dehumanization
  ❌ disinformation: False claims, misleading information
  🕵️ conspiracy_theory: Conspiracy theories, hidden agendas
  ⚡ far_right_bias: Far-right political bias, extremist views
  📢 call_to_action: Mobilization calls, organized actions
  🏴 nationalism: Nationalist rhetoric, cultural superiority
  🏛️ anti_government: Anti-institutional, anti-system discourse  
  📜 historical_revisionism: Historical revisionism, denial
  🗳️ political_general: General political discourse
  ✅ general: Neutral, non-problematic content
        """
    )
    
    # Input methods
    group = parser.add_mutually_exclusive_group()
    group.add_argument('text', nargs='?', help='Text to analyze (direct argument)')
    group.add_argument('--text', '-t', help='Text to analyze (flag format)')
    group.add_argument('--file', '-f', help='Read text from file')
    
    # Options
    parser.add_argument('--llm', action='store_true', help='Enable LLM analysis (slower but more accurate)')
    parser.add_argument('--json', action='store_true', help='Output results in JSON format')
    
    args = parser.parse_args()
    
    # Determine text input
    text = None
    if args.text:
        text = args.text
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except FileNotFoundError:
            print(f"❌ Error: File '{args.file}' not found")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            sys.exit(1)
    elif args.text is None and len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        # Direct positional argument
        text = sys.argv[1]
    else:
        parser.print_help()
        sys.exit(1)
    
    if not text:
        print("❌ Error: No text provided")
        parser.print_help()
        sys.exit(1)
    
    # Analyze text
    analyze_text(text, use_llm=args.llm, json_output=args.json)

def interactive_mode(use_llm=False, json_output=False):
    """Interactive mode for continuous testing."""
    print("🚀 Analyzer - Interactive Mode")
    print("=" * 50)
    mode = "LLM + Patterns" if use_llm else "Patterns Only"
    print(f"🔧 Analysis Mode: {mode}")
    print("Enter text to analyze (or 'quit' to exit)")
    print("Commands: 'help', 'toggle-llm', 'quit'")
    print()
    
    # Initialize analyzer once and keep it loaded
    try:
        analyzer = Analyzer(use_llm=use_llm)
        print("✅ Models loaded and ready!")
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        return
    
    while True:
        try:
            text = input("📝 Text: ").strip()
            
            if not text:
                continue
            elif text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif text.lower() == 'help':
                print_help()
                continue
            elif text.lower() == 'toggle-llm':
                use_llm = not use_llm
                print(f"🧠 LLM mode: {'ON' if use_llm else 'OFF'}")
                print("⚠️  Note: Restart interactive mode to apply LLM toggle")
                continue
            
            # Analyze the text (models already loaded)
            analyze_single_text(analyzer, text, json_output=json_output)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def analyze_text(text, use_llm=False, json_output=False):
    """Analyze a single text input."""
    try:
        # Initialize analyzer with verbose output by default
        analyzer = Analyzer(use_llm=use_llm)
        
        # Show mode info
        mode = "LLM + Patterns" if use_llm else "Patterns Only"
        print(f"🔧 Analysis Mode: {mode}")
        
        # Analyze text
        analyze_single_text(analyzer, text, json_output=json_output)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def analyze_single_text(analyzer, text, json_output=False):
    """Analyze a single text with the given analyzer."""
    try:
        # Analyze content with verbose output by default
        result = analyzer.analyze_content(
            tweet_id='quick_test',
            tweet_url='https://test.com/quick_test', 
            username='test_user',
            content=text
        )
        
        if json_output:
            import json
            output = {
                'content': text,
                'category': result.category,
                'confidence': getattr(result, 'confidence', None),
                'method_used': getattr(result, 'method_used', 'unknown'),
                'explanation': result.explanation if hasattr(result, 'explanation') else None
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            # Clean output format
            print(f"📝 Text: {text}")
            print(f"🎯 Category: {result.category}")
            
            # Show explanation from LLM analysis
            if result.llm_explanation and result.llm_explanation.strip():
                print(f"💭 Explanation: {result.llm_explanation}")
            
            # Add emoji indicators
            category_emojis = {
                Categories.HATE_SPEECH: '🚫',
                Categories.DISINFORMATION: '❌', 
                Categories.CONSPIRACY_THEORY: '🕵️',
                Categories.FAR_RIGHT_BIAS: '⚡',
                Categories.CALL_TO_ACTION: '📢',
                Categories.NATIONALISM: '🏴',
                Categories.ANTI_GOVERNMENT: '🏛️',
                Categories.HISTORICAL_REVISIONISM: '📜',
                Categories.POLITICAL_GENERAL: '🗳️',
                Categories.GENERAL: '✅'
            }
            
            emoji = category_emojis.get(result.category, '❓')
            print(f"{emoji} {result.category}")
            
            # Show analysis method used
            method = getattr(result, 'analysis_method', 'unknown')
            if method == 'pattern':
                print("⚡ Method: Pattern-based (fast)")
            elif method == 'llm':
                print("🧠 Method: LLM analysis")
            else:
                print(f"🔧 Method: {method}")
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")

def print_help():
    """Print help information."""
    print("""
🔧 Interactive Commands:
  help           - Show this help
  toggle-llm     - Enable/disable LLM analysis  
  toggle-verbose - Enable/disable verbose output
  quit/exit/q    - Exit interactive mode
  
📊 Categories:
  🚫 hate_speech      - Direct hate speech, slurs
  ❌ disinformation   - False claims, misleading info
  🕵️ conspiracy_theory - Conspiracy theories, hidden agendas  
  ⚡ far_right_bias   - Far-right political bias
  📢 call_to_action   - Mobilization calls, actions
  🏴 nationalism      - Nationalist rhetoric, superiority
  🏛️ anti_government   - Anti-institutional discourse
  📜 historical_revisionism - Historical revisionism, denial  
  🗳️ political_general - General political discourse
  ✅ general          - Neutral, non-problematic content
    """)

if __name__ == '__main__':
    main()
