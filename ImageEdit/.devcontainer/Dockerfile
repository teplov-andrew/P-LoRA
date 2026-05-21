FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/hub \
    ACCELERATE_DISABLE_RICH=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        git \
        ca-certificates \
        wget \
        fonts-dejavu-core \
        tmux \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip wheel setuptools

COPY .devcontainer/requirements.txt /tmp/requirements.txt
RUN pip install \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        -r /tmp/requirements.txt

# The training script pins `check_min_version("0.36.0.dev0")` (see
# train_dreambooth_lora_sdxl.py:83), which is unreleased — install from git.
RUN pip install "git+https://github.com/huggingface/diffusers.git"

# src/metrics.py imports `clip` (OpenAI's repo, not the PyPI package).
RUN pip install "git+https://github.com/openai/CLIP.git"

WORKDIR /workspaces/ImageEdit

CMD ["/bin/bash"]
