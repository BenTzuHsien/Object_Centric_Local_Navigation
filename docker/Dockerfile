FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Add host user to the container
ARG HOST_USER
ARG HOST_USER_ID
ARG HOST_GROUP
ARG HOST_GROUP_ID

RUN groupadd -g $HOST_GROUP_ID $HOST_GROUP \
    && useradd -m -u $HOST_USER_ID -g $HOST_GROUP_ID -s /bin/bash $HOST_USER \
    && passwd -d root

ENV HOME=/home/${HOST_USER}

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute,display

# Cuda Setup
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install Packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    lsb-release curl gnupg2 git vim tmux wget \
    python3-dev build-essential iputils-ping \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libglu1-mesa-dev && \
    python3 -m pip install bosdyn-client==5.0.0 bosdyn-mission==5.0.0 bosdyn-choreography-client==5.0.0 bosdyn-orbit==5.0.0 \
    PyOpenGL PyOpenGL_accelerate pygame && \
    python3 -m pip install xformers --index-url https://download.pytorch.org/whl/cu124 && \
    rm -rf /var/lib/apt/lists/*

# Install Grounded SAM2
# TORCH_CUDA_ARCH_LIST tied directly to your GPU, and it's essecial for building grounding_dino
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"
WORKDIR /opt
RUN python3 -m pip install opencv-python supervision pycocotools transformers addict yapf timm && \
    git clone https://github.com/IDEA-Research/Grounded-SAM-2.git && \
    cd /opt/Grounded-SAM-2 && \
    python3 -m pip install -e . && \
    python3 -m pip install --no-build-isolation -e grounding_dino && \
    cd /opt/Grounded-SAM-2/checkpoints && \
    bash download_ckpts.sh && \
    cd /opt/Grounded-SAM-2/gdino_checkpoints && \
    bash download_ckpts.sh
ENV PYTHONPATH="/opt/Grounded-SAM-2:${PYTHONPATH:+:$PYTHONPATH}"

# Reset DEBIAN_FRONTEND
ENV DEBIAN_FRONTEND=dialog

WORKDIR $HOME
USER $HOST_USER
