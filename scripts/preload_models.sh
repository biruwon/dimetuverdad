#!/bin/bash
# Preload models for multi-model analysis with extended TTL (48 hours)
# 
# IMPORTANT: With 32GB unified memory, all 4 models (~61GB total) cannot fit simultaneously.
# Ollama will dynamically load/unload models as needed during multi-model analysis.
# 
# This script sets extended keep-alive (48h) so models stay cached longer when not in active use,
# reducing reload times during sequential analysis.

echo "🚀 Configuring models for multi-model analysis..."
echo "⏰ Keep-alive set to 48 hours (models load on-demand during analysis)"
echo ""
echo "💡 System: 32GB unified memory - models will load sequentially as needed"
echo ""

# Array of models used in multi-model analysis
models=(
    "gemma3:4b"
    "gemma3:12b"
    "gemma3:27b-it-qat"
    "gpt-oss:20b"
)

# Function to touch a model and set its keep-alive
touch_model() {
    local model=$1
    echo "⚡ Setting keep-alive for $model..."
    
    response=$(curl -s http://localhost:11434/api/generate -d "{
        \"model\": \"$model\",
        \"prompt\": \"OK\",
        \"stream\": false,
        \"keep_alive\": \"48h\",
        \"options\": {
            \"num_predict\": 1
        }
    }")
    
    if echo "$response" | grep -q "response"; then
        echo "✅ $model configured (keep-alive: 48h)"
        return 0
    else
        echo "❌ Failed to configure $model"
        return 1
    fi
}

# Touch each model to set keep-alive
# Models will be evicted from memory but keep-alive setting persists
for model in "${models[@]}"; do
    touch_model "$model"
    echo ""
done

echo "🔍 Currently loaded model:"
ollama ps
echo ""
echo "✅ All models configured with 48h keep-alive!"
echo ""
echo "📊 Model sizes (will load on-demand during analysis):"
echo "   gemma3:4b:        ~3.3 GB"
echo "   gemma3:12b:       ~8.1 GB"
echo "   gemma3:27b-it-qat: ~18 GB"
echo "   gpt-oss:20b:      ~13 GB"
echo ""
echo "🎯 Multi-model analysis workflow:"
echo "   1. Run: ./run_in_venv.sh analyze-twitter-multi --post-id <id>"
echo "   2. Models load sequentially as needed (~30-60s each)"
echo "   3. Keep-alive ensures faster reload for subsequent analyses"
echo ""
echo "💡 To reduce analysis time, use single-model mode:"
echo "   ./run_in_venv.sh analyze-twitter --post-id <id>"
