# TRELLIS.2 Dockerfile
#
# Prereq:
#   nvidia container toolkit:
#   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
#
#   licenses:
#   https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
#   https://huggingface.co/briaai/RMBG-2.0
#
# Build (requires HuggingFace token for gated models):
#   python3 -m venv .venv
#   source .venv/bin/activate
#   pip install -U "huggingface_hub"
#   hf auth login
#
#   docker build --secret id=hf_token,src=$HOME/.cache/huggingface/token -t trellis2 .
#
# Run:
#   docker run --gpus all -p 7860:7860 trellis2

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Environment variables for the application
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV GRADIO_SERVER_NAME=0.0.0.0

# Hugging Face cache location (models will be stored here)
ENV HF_HOME=/app/hf_cache

# CUDA architecture to compile for (8.9=RTX40xx, +PTX for forward compatibility)
ENV TORCH_CUDA_ARCH_LIST="8.9+PTX"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    libjpeg-dev \
    libpng-dev \
    libopenexr-dev \
    libwebp-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/include/eigen3/Eigen /usr/include/Eigen

# Set working directory
WORKDIR /app

# LAYER 1: Dependencies & CUDA extensions (rarely changes) ---
# Copy only setup script and o-voxel (required by setup.sh)
COPY setup.sh /app/
COPY o-voxel /app/o-voxel

# Run setup.sh to install all dependencies and build CUDA extensions
RUN PLATFORM=cuda bash -c ". ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm"

# LAYER 2: Model weights (rarely changes) ---
# Pre-download all model weights into the image (cached in HF_HOME)
# Some models are gated and require HF_TOKEN (passed via --secret)
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) && \
    huggingface-cli download microsoft/TRELLIS.2-4B --token "$HF_TOKEN" && \
    huggingface-cli download microsoft/TRELLIS-image-large --token "$HF_TOKEN" && \
    huggingface-cli download facebook/dinov3-vitl16-pretrain-lvd1689m --token "$HF_TOKEN" && \
    huggingface-cli download briaai/RMBG-2.0 --token "$HF_TOKEN"

# LAYER 3: Application code (changes frequently) ---
# Copy the rest of the application (this layer rebuilds on code changes)
COPY . /app

# Block runtime downloads - force offline mode (will error if model missing)
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Run the web demo
CMD ["python", "app.py"]
