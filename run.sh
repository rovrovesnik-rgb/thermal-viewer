#!/bin/zsh
# Run the P2Pro Thermal Camera Viewer

cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Enable rusticl OpenCL for AMD GPUs (Mesa)
export RUSTICL_ENABLE=radeonsi

# Run the application
uv run python main.py
