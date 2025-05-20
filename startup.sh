#!/bin/bash
# ------------------------
# This script launches:
# 1. The Ollama LLM server.
# 2. A specific model (qwen2.5:7b).
# 3. A FastAPI/uvicorn application.
# ------------------------

# 1. Start the Ollama server in the background
ollama serve & 


# 2. Run the 'qwen2.5:7b' model in the background
#    This command will load the model so it’s ready to serve inference requests.s
ollama run qwen2.5:7b &

# 3. Wait 5 seconds to allow the Ollama server and model to initialize
sleep 5


# 4. Launch the FastAPI application using uvicorn
#    - Bind to all network interfaces (0.0.0.0)
#    - Serve on port 8080
uvicorn app:app --host 0.0.0.0 --port 8080 