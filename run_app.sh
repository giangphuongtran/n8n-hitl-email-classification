#!/bin/bash
# Run FastAPI Email Classification Service
# This starts the production API server

set -e

PORT=${1:-8001}
VENV_PYTHON=".venv/bin/python"

echo "🚀 Email Classification API Server"
echo "=================================="

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "❌ ERROR: Virtual environment not found at .venv"
    echo "📁 Please run: source .venv/bin/activate first"
    exit 1
fi

echo "✅ Using venv: $VENV_PYTHON"
echo "📍 Python: $($VENV_PYTHON --version)"
echo "📍 Starting FastAPI server on port $PORT..."
echo ""

# Run the FastAPI app
$VENV_PYTHON -m uvicorn app:app --host 0.0.0.0 --port $PORT --reload

echo ""
echo "=================================="
echo "API Documentation:"
echo "  • Swagger UI: http://localhost:$PORT/docs"
echo "  • ReDoc: http://localhost:$PORT/redoc"
