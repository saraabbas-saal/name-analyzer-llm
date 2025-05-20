#!/bin/bash

# Install killall if needed
# apt-get update && apt-get install -y psmisc

# # Find and kill the ollama process
# echo "Stopping existing Ollama processes..."
# killall ollama || pkill ollama || echo "No Ollama processes found to kill"

# # Wait to ensure the process is fully stopped and port is released
# sleep 3

# Set environment variables more explicitly
export OLLAMA_USE_GPU=1
export OLLAMA_GPU_LAYERS=1000
export CUDA_VISIBLE_DEVICES=0

# Display NVIDIA devices
echo "Checking NVIDIA devices..."
ls -la /dev/nvidia* || echo "No NVIDIA devices found - this is a problem!"

# Check NVIDIA driver
echo "Checking NVIDIA driver..."
nvidia-smi || echo "nvidia-smi failed - GPU might not be accessible!"

# Start Ollama with GPU
echo "Starting Ollama with GPU settings..."
ollama serve &
OLLAMA_PID=$!

# Wait for it to start
echo "Waiting for Ollama to start..."
sleep 5

# Check if API is responding
echo "Checking Ollama API response:"
curl -s http://localhost:11434/api/info || echo "Ollama API not responding!"

# Pull the model with explicit GPU flag - force restart
echo "Pulling/updating model with GPU flag..."
OLLAMA_USE_GPU=1 ollama pull qwen2.5:7b

# Check Ollama model list with GPU stats
echo "Checking model status:"
ollama ps

# Run a simple inference test
echo "Running a simple inference test to verify GPU usage..."
echo 'Testing GPU with a simple query...' | OLLAMA_USE_GPU=1 ollama run qwen2.5:7b 2>&1 | grep -i gpu

# Check GPU utilization
echo "Checking GPU utilization after test:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

# Final check
echo "Final model status check:"
ollama ps

echo "Setup complete. If you still see '100% CPU', your container might not have proper GPU access."
