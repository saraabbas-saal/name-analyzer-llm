docker build -t name-analyzer-llm-gpu .
# docker run -it -p 8071:8071 name-analyzer-llm-gpu bash startup.sh
docker run --gpus all -it -p 8071:8071 name-analyzer-llm-gpu