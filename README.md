# Ollama Configuration for Quick Analysis

## Performance Results (M1 Pro MacBook)

**Best Configuration**: Original `gpt-oss:20b` with default settings
- **Time**: ~46 seconds per LLM analysis ⚡
- **Context**: 8192 tokens (full context)
- **Processing**: 100% GPU (optimal for M1 Pro)
- **Memory**: 14GB fits well in 32GB RAM

## Keep Models in Memory

To avoid loading delays, keep the model loaded in Ollama memory:

```bash
# Load original model with optimal settings
ollama run gpt-oss:20b --keepalive 24h "Ready for fast analysis"

# Check loaded models
ollama ps

## Performance Comparison

- **Without preloaded model**: 3+ minutes per analysis
- **With original model (gpt-oss:20b) - BEST**: ~46 seconds per LLM analysis ⚡
- **With "optimized" model (gpt-oss-fast)**: ~2+ minutes per LLM analysis
- **Pattern-only analysis**: ~5 seconds

*Key insight: M1 Pro's GPU is much faster than CPU for this workload!*

# Check loaded models
ollama ps
```

## Quick Test Usage

With model preloaded, analysis is much faster:

```bash
# Activate environment
source venv/bin/activate

# Pattern-only analysis (fast, ~2-5 seconds)
python quick_test.py "Your text here"

# LLM analysis (medium, ~20-30 seconds with preloaded model)  
python quick_test.py --llm "Your text here"

# JSON output
python quick_test.py --llm --json "Your text here"
```

## Performance Comparison

- **Without preloaded model**: 3+ minutes per analysis
- **With optimized preloaded model (gpt-oss-fast)**: ~2-3 minutes per LLM analysis  
- **With original preloaded model (gpt-oss:20b)**: ~3+ minutes per LLM analysis
- **Pattern-only analysis**: ~5 seconds

*Note: The 2-3 minute time includes Python startup, model loading, and analysis. The actual LLM inference is much faster (~10-20 seconds) but system initialization adds overhead.*

## System Optimization for M1 Pro

Your MacBook Pro M1 Pro with 32GB RAM works best with:
- **GPU Processing**: 100% GPU utilization (much faster than CPU)
- **Full Context**: 8192 tokens (no performance penalty for short texts)
- **Memory**: 14GB model size fits comfortably in your 32GB RAM
- **Temperature**: Default settings work optimally

## Why "CPU optimization" failed:
- M1 Pro's GPU cores are specifically designed for ML workloads
- Forcing CPU processing created an unnecessary bottleneck  
- The unified memory architecture works better with GPU processing

## Model Memory Usage

The `gpt-oss-fast` model uses approximately 14GB of memory when loaded.
Check with `ollama ps` to see current memory usage and processing distribution.

## Additional M1 Pro Optimizations

For even better performance on your Mac:

```bash
# Monitor system resources
ollama ps  # Check model status
htop       # Monitor CPU/memory usage

# For sustained high-performance work, consider:
# 1. Close other memory-intensive applications
# 2. Ensure good ventilation (thermal throttling can slow performance)
# 3. Use Activity Monitor to check for background processes

# Check current optimization status:
system_profiler SPHardwareDataType | grep -E "(Model|Memory)"
```

## Advanced Configuration

Create custom model variants with different optimization levels:

```bash
# Create even faster model with minimal context
cat > Modelfile.ultrafast << EOF
FROM gpt-oss:20b
PARAMETER num_ctx 1024
PARAMETER temperature 0.05
SYSTEM "Classify as: hate_speech, disinformation, conspiracy_theory, far_right_bias, call_to_action, general. One word only."
EOF

ollama create gpt-oss-ultrafast -f Modelfile.ultrafast
```
