 source /Users/apple/online-teacher/.venv/bin/activate
#!/bin/bash

echo "========================================"
echo "  Online Teacher Platform - Starting"
echo "========================================"

# Step 0 — Start Docker Desktop if not running
echo "→ Starting Docker Desktop..."
open -a Docker
echo "   Waiting for Docker to start (30s)..."
sleep 30

# Wait until Docker is actually ready
ATTEMPTS=0
until docker info > /dev/null 2>&1; do
    ATTEMPTS=$((ATTEMPTS+1))
    if [ $ATTEMPTS -gt 20 ]; then
        echo "   ❌ Docker failed to start. Please open Docker Desktop manually."
        exit 1
    fi
    echo "   Waiting for Docker daemon... (${ATTEMPTS}/20)"
    sleep 5
done
echo "   ✅ Docker is ready"

# Step 1 — Kill any processes on our ports
echo "→ Clearing port 8000..."
kill -9 $(lsof -t -i:8000) 2>/dev/null
sleep 2

# Step 2 — Start Qdrant
echo "→ Starting Qdrant..."
docker start qdrant 2>/dev/null
if [ $? -ne 0 ]; then
    docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
fi
sleep 3

# Step 3 — Verify Qdrant
echo "→ Verifying Qdrant..."
until curl -s http://localhost:6333 > /dev/null; do
    echo "   Waiting for Qdrant..."
    sleep 3
done
echo "   ✅ Qdrant is ready"

# Step 4 — Activate venv
echo "→ Activating virtual environment..."
source ~/online-teacher/.venv/bin/activate

# Step 5 — Preload MiniLM model before first request
echo "→ Preloading MiniLM model (one-time, ~5 seconds)..."
python3 -c "
from sentence_transformers import SentenceTransformer
print('Loading all-MiniLM-L6-v2...')
model = SentenceTransformer('all-MiniLM-L6-v2')
print('MiniLM ready.')
" && echo "   ✅ MiniLM preloaded"

# Step 6 — Start main app
echo "→ Starting main app on port 8000..."
echo "========================================"
echo "  Platform ready!"
echo "  API:  http://localhost:8000"
echo "  Docs: http://localhost:8000/docs"
echo "========================================"
uvicorn app.main:app --reload --port 8000

