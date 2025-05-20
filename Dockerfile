FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python3 -> python
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /opt/app
COPY . /opt/app

# Install Ollama with GPU support
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Python dependencies
RUN pip install fastapi uvicorn "openai>=1.0.0" instructor

# Download model during build with GPU acceleration
RUN ollama serve & \
    sleep 5 && \
     ollama pull qwen2.5:7b

ENV OLLAMA_USE_GPU=1

EXPOSE 8071
COPY startup.sh .
RUN chmod +x startup.sh
CMD ["./fix_gpu.sh"]