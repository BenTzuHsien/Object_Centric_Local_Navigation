#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_NAME=$(basename "$(dirname "$SCRIPT_DIR")")
REPO_DIRECTORY="$(cd "$SCRIPT_DIR/../" && pwd)"

DATASET_DIRECTORY=''

IMAGE_NAME=$(echo "$REPO_NAME" | tr '[:upper:]' '[:lower:]' | tr -d '_')
IMAGE_TAG="latest"
CONTAINER_HOME=$(docker inspect --format='{{.Config.Env}}' $IMAGE_NAME:$IMAGE_TAG | tr ' ' '\n' | grep '^HOME=' | cut -d= -f2)

# For Display, Connect to XServer
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]; then
   xauth_list=$(xauth nlist $DISPLAY)
   xauth_list=$(sed -e 's/^..../ffff/' <<<"$xauth_list")
   if [ ! -z "$xauth_list" ]; then
       echo "$xauth_list" | xauth -f $XAUTH nmerge -
   else
       touch $XAUTH
   fi
   chmod a+r $XAUTH
fi

# Prevent executing "docker run" when xauth failed.
if [ ! -f $XAUTH ]; then
  echo "[$XAUTH] was not properly created. Exiting..."
  exit 1
fi

# Mount dataset
DOCKER_DATASET_MOUNT=""
if [ -n "$DATASET_DIRECTORY" ]; then
    DATASET_NAME=$(basename "$DATASET_DIRECTORY")
    DOCKER_DATASET_MOUNT="-v $DATASET_DIRECTORY:/data/$DATASET_NAME"
fi

docker run \
       -it \
       --name $REPO_NAME \
       --user $(id -u):$(id -g) \
       -v $REPO_DIRECTORY:$CONTAINER_HOME/$REPO_NAME \
       $DOCKER_DATASET_MOUNT \
       --network host \
       --shm-size=128G \
       --gpus all \
       --env="DISPLAY=$DISPLAY" \
       --env="XAUTHORITY=$XAUTH" \
       --volume="$XAUTH:$XAUTH" \
       --env="QT_X11_NO_MITSHM=1" \
       --volume="$HOME/.vscode:$CONTAINER_HOME/.vscode" \
       $IMAGE_NAME:$IMAGE_TAG