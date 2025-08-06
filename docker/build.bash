#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_NAME=$(basename "$(dirname "$SCRIPT_DIR")")
IMAGE_NAME=$(echo "$REPO_NAME" | tr '[:upper:]' '[:lower:]' | tr -d '_')

echo "Building Docker image: ${IMAGE_NAME}:latest"

HOST_USER=$(whoami)
HOST_USER_ID=$(id -u)
HOST_GROUP=$(id -gn)
HOST_GROUP_ID=$(id -g)

docker build .\
        -t $IMAGE_NAME:latest\
        --build-arg HOST_USER="$HOST_USER"\
        --build-arg HOST_USER_ID="$HOST_USER_ID"\
        --build-arg HOST_GROUP="$HOST_GROUP"\
        --build-arg HOST_GROUP_ID="$HOST_GROUP_ID"\
        --rm
