#!/bin/bash
# ------------------------
# This script launches:
# 1. The Ollama LLM server with GPU support
# 2. A specific model (qwen2.5:7b) with GPU acceleration
# 3. A FastAPI/uvicorn application
# ------------------------

# Set environment variable to enable GPU usage
export OLLAMA_USE_GPU=1

# Use CUDA_VISIBLE_DEVICES passed from docker run, or default to all GPUs if not specified
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "No specific GPU selected, using all available GPUs"
else
    echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"
fi

# Check if nvidia-smi is available and display GPU info
if command -v nvidia-smi &> /dev/null; then
    echo "GPU information:"
    nvidia-smi
else
    echo "WARNING: nvidia-smi command not found. This might indicate that the GPU is not properly configured."
    echo "Make sure you're running with --gpus flag and NVIDIA drivers are properly installed."
fi

echo "Starting Ollama server with GPU support..."
# 1. Start the Ollama server in the background with GPU support
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama server to start
echo "Waiting for Ollama server to initialize..."
sleep 5

# Add a verification step to check if Ollama is using GPU
echo "Checking Ollama GPU utilization..."
curl -s http://localhost:11434/api/tags | grep -i gpu || echo "WARNING: GPU may not be detected by Ollama"

# 2. Run the 'qwen2.5:7b' model with GPU acceleration
echo "Loading qwen2.5:7b model with GPU acceleration..."
ollama run qwen2.5:7b &
MODEL_PID=$!

# 3. Wait for model to initialize
echo "Waiting for model to initialize..."
sleep 5

# Check GPU utilization
echo "GPU utilization after loading model:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
fi

# 4. Launch the FastAPI application using uvicorn
echo "Starting FastAPI application..."
uvicorn app:app --host 0.0.0.0 --port 8071

# Handle shutdown gracefully
trap "kill $OLLAMA_PID $MODEL_PID" EXIT