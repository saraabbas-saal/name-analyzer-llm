#!/bin/bash
# ------------------------
# This script launches:
# 1. The Ollama LLM server with GPU support
# 2. A specific model (qwen2.5:7b) with GPU acceleration
# 3. A FastAPI/uvicorn application
# ------------------------

# Set environment variable to enable GPU usage
export OLLAMA_USE_GPU=1
export CUDA_VISIBLE_DEVICES=0

echo "Starting Ollama server with GPU support..."
# 1. Start the Ollama server in the background with GPU support
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama server to start
echo "Waiting for Ollama server to initialize..."
sleep 5

# 2. Run the 'qwen2.5:7b' model with GPU acceleration
echo "Loading qwen2.5:7b model with GPU acceleration..."
ollama run qwen2.5:7b &
MODEL_PID=$!

# 3. Wait for model to initialize
echo "Waiting for model to initialize..."
sleep 5

# 4. Launch the FastAPI application using uvicorn
echo "Starting FastAPI application..."
uvicorn app:app --host 0.0.0.0 --port 8071

# Handle shutdown gracefully
trap "kill $OLLAMA_PID $MODEL_PID" EXIT