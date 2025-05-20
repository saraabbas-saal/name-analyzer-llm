FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set environment variables for GPU and other settings
ENV DEBIAN_FRONTEND=noninteractive \
    OLLAMA_USE_GPU=1 \
    OLLAMA_GPU_LAYERS=1000 \
    CUDA_VISIBLE_DEVICES=0 \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install required dependencies
# This layer rarely changes, so it will be cached
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    wget \
    git \
    psmisc \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama with explicit GPU support
# This is a separate step that will be cached unless Ollama changes
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Python dependencies directly
RUN pip3 install --no-cache-dir fastapi uvicorn "openai>=1.0.0" instructor requests

# Verify CUDA installation
RUN echo "Verifying CUDA installation..." && \
    ls -la /usr/local/cuda && \
    ldconfig -p | grep -i cuda | wc -l

# Create app directory
WORKDIR /opt/app

# Copy just the startup script first (changes less frequently)
COPY startup.sh /opt/app/
RUN chmod +x /opt/app/startup.sh

# Copy data folder (changes occasionally)
COPY data/ /opt/app/data/

# Copy application code (changes most frequently) - add this last
COPY app.py /opt/app/

# Expose the FastAPI port
EXPOSE 8071

# Set the entrypoint to the startup script
CMD ["/opt/app/startup.sh"]