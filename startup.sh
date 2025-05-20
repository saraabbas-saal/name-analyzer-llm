#!/bin/bash

# ------------------------
# GPU-accelerated startup script for name analyzer service
# This script:
# 1. Verifies GPU availability
# 2. Starts Ollama with GPU support
# 3. Loads the LLM model with GPU acceleration
# 4. Launches the FastAPI application
# ------------------------

# Display GPU information
echo "========== GPU VERIFICATION =========="
nvidia-smi || echo "WARNING: nvidia-smi failed. GPU might not be properly accessible!"

# List NVIDIA devices
echo "Checking NVIDIA device access:"
ls -la /dev/nvidia* || echo "WARNING: No NVIDIA devices found! Container might not have GPU access."

# Check CUDA libraries
echo "Checking CUDA libraries:"
ldconfig -p | grep -i cuda | head -5

# Ensure GPU settings are active
export OLLAMA_USE_GPU=1
export OLLAMA_GPU_LAYERS=1000
export CUDA_VISIBLE_DEVICES=0

echo "Environment variables set:"
env | grep -E "OLLAMA|CUDA|NVIDIA" | sort

# Kill any existing Ollama process
echo "Stopping any existing Ollama processes:"
pkill ollama || echo "No Ollama processes found to kill."
sleep 2

# Start Ollama server with GPU support
echo "========== STARTING OLLAMA WITH GPU =========="
echo "Starting Ollama server with GPU support..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to initialize
echo "Waiting for Ollama server to initialize..."
sleep 5

# Verify Ollama API is responding
echo "Checking Ollama API:"
curl -s http://localhost:11434/api/info || echo "WARNING: Ollama API not responding!"

# Pull the model with explicit GPU settings
echo "========== LOADING MODEL WITH GPU =========="
echo "Pulling qwen2.5:7b model with GPU acceleration..."
OLLAMA_USE_GPU=1 ollama pull qwen2.5:7b

# Verify model status
echo "Checking model status:"
ollama ps

# If model shows "100% CPU", try to force GPU usage
if ollama ps | grep -q "100% CPU"; then
    echo "WARNING: Model still using CPU! Attempting to force GPU usage..."
    
    # Try alternative approach
    pkill ollama
    sleep 2
    
    # Create Ollama config directory if it doesn't exist
    # mkdir -p ~/.ollama
    
    # # Create/update Ollama configuration
    # echo '{
    #   "gpu": true,
    #   "gpu_layers": 100
    # }' > ~/.ollama/config.json
    
    # Restart Ollama with explicit config
    OLLAMA_USE_GPU=1 ollama serve &
    OLLAMA_PID=$!
    sleep 5
    
    # Force model reload
    OLLAMA_USE_GPU=1 ollama pull qwen2.5:7b
    
    # Check again
    echo "Checking model status after forcing GPU:"
    ollama ps
fi

# Run a quick test to see if GPU is utilized
echo "Running quick test to verify GPU utilization..."
echo "Testing GPU acceleration" | OLLAMA_USE_GPU=1 ollama run qwen2.5:7b

# Check GPU utilization after test
echo "GPU utilization after test:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

# Start the FastAPI application
echo "========== STARTING FASTAPI APPLICATION =========="
echo "Starting FastAPI application on port 8071..."
uvicorn app:app --host 0.0.0.0 --port 8071

# Cleanup on exit
trap "kill $OLLAMA_PID" EXIT

