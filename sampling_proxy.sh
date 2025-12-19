#!/bin/bash

# Sampling Proxy Startup Script for Linux/macOS
# This script activates the virtual environment and starts the proxy server

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/sampling-proxy"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found at: $VENV_PATH"
    echo "Please create it first with:"
    echo "  python -m venv sampling-proxy"
    echo "  source sampling-proxy/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Check if activation was successful
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "Virtual environment activated: $VIRTUAL_ENV"
else
    echo "Failed to activate virtual environment"
    exit 1
fi

# Start the proxy server
echo "Starting Sampling Proxy..."
python sampling_proxy.py "$@"
