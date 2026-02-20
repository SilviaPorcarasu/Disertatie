FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/Disertatie/src \
    XDG_CACHE_HOME=/workspace/.cache \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
    HF_HUB_CACHE=/workspace/.cache/huggingface/hub \
    TMPDIR=/workspace/.cache/tmp \
    HF_HUB_DISABLE_XET=1

WORKDIR /workspace/Disertatie

COPY requirements.txt /workspace/Disertatie/requirements.txt

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r /workspace/Disertatie/requirements.txt

COPY . /workspace/Disertatie

RUN mkdir -p /workspace/.cache/tmp /workspace/Disertatie/outputs

CMD ["bash"]
