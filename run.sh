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

# 2. Configure Tailscale Serve for private mobile access.
if command -v tailscale >/dev/null 2>&1; then
    if tailscale serve --bg 7896 >/dev/null 2>&1; then
        TAILSCALE_HOST=$(tailscale status --json 2>/dev/null | .venv/bin/python3 -c \
            'import json, sys; print(json.load(sys.stdin).get("Self", {}).get("DNSName", "").rstrip("."))' \
            2>/dev/null || true)

        echo -e "${GREEN}Tailscale Serve enabled.${NC}"
        if [ -n "${TAILSCALE_HOST}" ]; then
            echo -e "${GREEN}Open on mobile: https://${TAILSCALE_HOST}/mobile/${NC}"
        else
            echo "Tailscale is enabled, but its MagicDNS name could not be detected."
            echo "Run: tailscale serve status"
        fi
    else
        echo "Warning: Tailscale Serve could not be enabled. Continuing with local access only."
        echo "Check that Tailscale is running and signed in."
    fi
else
    echo "Warning: Tailscale is not installed. Continuing with local access only."
    echo "Install it with: brew install --cask tailscale"
fi

# 3. Run the Server
echo -e "${GREEN}Starting TTS Hub on http://localhost:7896 ...${NC}"
echo "Press Ctrl+C to stop."
echo ""

# Run the supported desktop application using the virtual environment.
.venv/bin/python3 app.py --port 7896
