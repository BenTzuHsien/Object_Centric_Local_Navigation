#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="objectcentriclocalnavigation"
IMAGE_TAG="latest"
REPO_NAME="Object_Centric_Local_Navigation"
REPO_DIRECTORY="$SCRIPT_DIR/../../Object_Centric_Local_Navigation"

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

docker run \
       -it \
       --user $(id -u):$(id -g) \
       -v $REPO_DIRECTORY:/root/$REPO_NAME \
       --network host \
       --shm-size=128G \
       --gpus all \
       --runtime=nvidia \
       --env="DISPLAY=$DISPLAY" \
       --env="XAUTHORITY=$XAUTH" \
       --volume="$XAUTH:$XAUTH" \
       --env="QT_X11_NO_MITSHM=1" \
       --volume="$HOME/.vscode:/root/.vscode" \
       $IMAGE_NAME:$IMAGE_TAG