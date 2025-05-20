# Build the image
docker build -t name-analyzer-gpu:latest .

# Run with GPU support
docker run --name name-analyzer-gpu \
    --rm \
  --gpus all \
  -v ./:/app \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e OLLAMA_USE_GPU=1 \
  -e OLLAMA_GPU_LAYERS=1000 \
  -p 8071:8071 \
  --ipc=host \
  name-analyzer-gpu:latest