# Name Origin Scoring LLM Service (GPU-Accelerated)

This repository contains a FastAPI-based service that analyzes a given name and predicts the most likely countries of origin. Internally, it uses a locally hosted LLM (via [Ollama](https://github.com/jmorganca/ollama)) with **GPU acceleration** for name analysis and returns a JSON response with the predicted origins.

## Features

- **GPU Acceleration**: Utilizes NVIDIA CUDA for faster inference.
- **Name Analysis**: Receive a name through a GET endpoint (`/name_analyzer`) and get back up to 5 likely ISO alpha-3 country codes with confidence probabilities.
- **Local LLM Integration**: Uses a locally running LLM instance (e.g., `qwen2.5:7b`) loaded through Ollama with GPU support.
- **ISO Validation**: Filters out invalid alpha-3 country codes using a local `iso_3166_countries.json` file.
- **Customizable**: Easily modify the prompt, max origins, or model temperature settings.

## Requirements

- **Python** 3.8+
- **NVIDIA GPU** with CUDA support
- **NVIDIA Drivers** and **CUDA Toolkit** installed on the host system
- **Ollama**: A local LLM server ([Ollama on GitHub](https://github.com/jmorganca/ollama)) with GPU support
- **Docker** with NVIDIA Container Toolkit installed
- **qwen2.5:7b Model**: Or any compatible model served by Ollama

## Host System Setup

1. Install NVIDIA drivers appropriate for your GPU
2. Install NVIDIA Container Toolkit:
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

## Build the Docker Image
```bash
docker build -t name_analyzer_gpu:latest .
```

## Run the service with GPU
```bash
docker run --gpus all -it -p 8071:8071 name_analyzer_gpu:latest
```

## Send a request
```bash
curl -X GET "http://localhost:8071/name_analyzer?name=Sara%20Abbas"
```

### Expected output
```bash
{"name":"Sara Abbas","likely_origins":[{"origin":"SAU","probability":0.45},{"origin":"IRQ","probability":0.25},{"origin":"PAK","probability":0.15},{"origin":"JOR","probability":0.1},{"origin":"BHR","probability":0.05}]}
```

## Performance Benefits of GPU Acceleration

Using a GPU can significantly improve the inference speed of the LLM:
- Faster response times for name analysis requests
- Higher throughput for handling multiple requests
- Better performance with larger LLM models

## Code Explanation

### Dockerfile
- Uses NVIDIA CUDA base image for GPU support
- Sets environment variables for GPU usage
- Installs necessary dependencies

### app.py
- Sets environment variables to enable GPU usage
- Otherwise functions as before but with faster inference

### startup.sh
- Exports OLLAMA_USE_GPU=1 to enable GPU acceleration
- Launches the Ollama server with GPU support
- Runs the model with GPU acceleration
- Starts the FastAPI application

### Note on GPU Memory
The service requires sufficient GPU memory for the chosen model. For `qwen2.5:7b`, you'll need approximately 7-8GB of VRAM. Adjust your model selection based on available GPU resources.