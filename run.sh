docker build -t name-analyzer-llm-gpu .
# docker run -it -p 8071:8071 name-analyzer-llm-gpu bash startup.sh
docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=1 -it -p 8071:8071 name-analyzer-llm-gpu