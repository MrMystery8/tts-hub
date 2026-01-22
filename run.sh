#!/bin/bash
set -e

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}==> TTS Hub Launcher <==${NC}"

# 1. Check/Create Virtual Environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "Installing dependencies..."
    .venv/bin/pip install -r requirements.txt
else
    echo "Virtual environment found."
fi

# 2. Run the Server
echo -e "${GREEN}Starting TTS Hub on http://localhost:7891 ...${NC}"
echo "Press Ctrl+C to stop."
echo ""

# Run webui.py using the venv python
.venv/bin/python3 webui.py --port 7891
