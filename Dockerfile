# Base image: PyTorch 2.4 with CUDA 12.1 and cuDNN9
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# --- Environment ---
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    LANG=C.UTF-8 LC_ALL=C.UTF-8

# --- System tools ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl vim tini ca-certificates locales && \
    rm -rf /var/lib/apt/lists/* && \
    locale-gen en_US.UTF-8

# --- Python deps: core DL + RLHF + eval + retrieval + logging ---
RUN pip install -U pip setuptools wheel && \
    pip install -U \
        "transformers>=4.44" "accelerate>=0.34" "datasets" "peft>=0.11.1" \
        "trl>=0.9.6" "bitsandbytes>=0.43" \
        numpy scipy pandas scikit-learn einops sympy networkx \
        evaluate rouge-score sacrebleu bert-score \
        "lm-eval==0.4.2" \
        faiss-cpu rank-bm25 rapidfuzz \
        nltk spacy sentencepiece protobuf \
        wandb tensorboard rich loguru \
        "vllm>=0.5.3" fastapi uvicorn openai

# --- OPTIONAL: FlashAttention 2 (speed/memory). Safe to skip if it fails. ---
# Prebuilt wheels often work; fallback may compile (can be slow).
RUN pip install --no-build-isolation "flash-attn>=2.6.1" || true
# Prefer Flash-Attn for Transformers when available.
RUN pip install --no-cache-dir transformers peft bitsandbytes accelerate datasets trl==0.9.6


ENV FLASH_ATTENTION=1 \
    XFORMERS_DISABLED=1

# --- Workspace ---
WORKDIR /workspace
RUN mkdir -p /workspace/data /workspace/ckpt /workspace/app

# --- Your scripts ---
COPY app/ /workspace/app/

# --- Entry ---
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["bash"]
