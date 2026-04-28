#!/bin/bash
# Run Email Classifier using Kedro Pipeline
# 
# IMPORTANT: Run with 'source' to inherit the venv activation:
#   source run_classifier.sh
# 
# Or activate venv first, then run normally:
#   source .venv/bin/activate
#   kedro run --pipeline training

# Check if venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ ERROR: Virtual environment not activated"
    echo ""
    echo "📁 Solution 1 - Use with source (recommended):"
    echo "   source run_classifier.sh"
    echo ""
    echo "📁 Solution 2 - Activate venv first:"
    echo "   source .venv/bin/activate"
    echo "   kedro run --pipeline training"
    exit 1
fi

echo "🤖 Email Classification Training with Kedro"
echo "==========================================="
echo "✅ Virtual environment: $VIRTUAL_ENV"

# Check if data file exists
if [ ! -f "data/01_raw/n8n-classifier-tracker.csv" ]; then
    echo "❌ ERROR: Training data not found at data/01_raw/n8n-classifier-tracker.csv"
    echo "📁 Please place your CSV file at: data/01_raw/n8n-classifier-tracker.csv"
    exit 1
fi

echo "📂 Training data found: data/01_raw/n8n-classifier-tracker.csv"
echo "🚀 Starting Kedro training pipeline..."
echo ""

# Run the Kedro training pipeline
kedro run --pipeline training

echo ""
echo "==========================================="
echo "✅ Training complete!"
echo ""
echo "📊 Outputs:"
echo "  • Model: data/06_models/email_classifier_finetuned/"
echo "  • Metrics: data/07_model_output/evaluation_metrics.json"
echo "  • Logs: logs/"

echo ""
echo "==========================================="
echo "✅ Training complete!"
echo ""
echo "📊 Outputs:"
echo "  • Model: data/06_models/email_classifier_finetuned/"
echo "  • Metrics: data/07_model_output/evaluation_metrics.json"
echo "  • Logs: logs/"
