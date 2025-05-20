FROM python:3.10.11
COPY .  /opt/app
WORKDIR /opt/app

RUN apt-get update && apt-get install -y curl
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Ollama
# Download model during build
RUN ollama serve & \
    sleep 5 && \
    ollama pull qwen2.5:7b

RUN pip install fastapi uvicorn ollama instructor
COPY app.py .

EXPOSE 8070
COPY startup.sh .
RUN chmod +x startup.sh
CMD ["./startup.sh"]
