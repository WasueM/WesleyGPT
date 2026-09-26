# Wesley wrote this
# WesleyGPT API server (CPU). Build context = repo root with checkpoints staged in
# serve-data/ in nanochat's layout (tokenizer/, chatsft_checkpoints/<tag>/...).
FROM python:3.10-slim
# CPU-only torch from PyTorch's index; everything else from PyPI.
RUN pip install --no-cache-dir "torch==2.9.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "tiktoken>=0.11.0" "rustbpe>=0.1.0" "filelock>=3.19.0" "numpy>=1.26.0" \
      "psutil>=7.1.0" "fastapi>=0.115" "uvicorn>=0.30"
WORKDIR /app
COPY serve-data /app/serve-data
COPY nanochat /app/nanochat
COPY wesleygpt /app/wesleygpt
ENV NANOCHAT_BASE_DIR=/app/serve-data PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
CMD ["python", "-m", "wesleygpt.serve"]
