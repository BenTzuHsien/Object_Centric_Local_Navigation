FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute,display

# Cuda Setup
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# TORCH_CUDA_ARCH_LIST tied directly to your GPU, and it's essecial for building grounding_dino
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

# Install Packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    lsb-release curl gnupg2 git vim tmux wget \
    python3-dev build-essential iputils-ping \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libglu1-mesa-dev && \
    python3 -m pip install bosdyn-client==5.0.0 bosdyn-mission==5.0.0 bosdyn-choreography-client==5.0.0 bosdyn-orbit==5.0.0 \
    PyOpenGL PyOpenGL_accelerate pygame && \
    rm -rf /var/lib/apt/lists/*

# Install Grounded SAM2
WORKDIR /root/lib
RUN python3 -m pip install opencv-python supervision pycocotools transformers addict yapf timm && \
    git clone https://github.com/IDEA-Research/Grounded-SAM-2.git && \
    cd /root/lib/Grounded-SAM-2 && \
    python3 -m pip install -e . && \
    python3 -m pip install --no-build-isolation -e grounding_dino && \
    cd /root/lib/Grounded-SAM-2/checkpoints && \
    bash download_ckpts.sh && \
    cd /root/lib/Grounded-SAM-2/gdino_checkpoints && \
    bash download_ckpts.sh
ENV PYTHONPATH="/root/lib/Grounded-SAM-2:${PYTHONPATH:+:$PYTHONPATH}"

WORKDIR /root

# Accept build arguments
ARG hostuser
ARG hostgroup
ARG hostuid
ARG hostgid

RUN groupadd -g ${hostgid} ${hostgroup} && \
    useradd -m -u ${hostuid} -g ${hostgroup} ${hostuser}

USER ${hostuser}
WORKDIR /home/${hostuser}

# Reset DEBIAN_FRONTEND
ENV DEBIAN_FRONTEND=dialog