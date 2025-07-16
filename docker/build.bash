#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

hostuser=$USER
hostuid=$UID
hostgroup=$(id -gn "$hostuser")
hostgid=$(id -g "$hostuser")

docker build . -t objectcentriclocalnavigation:latest \
        --build-arg hostuser="$hostuser" \
        --build-arg hostgroup="$hostgroup" \
        --build-arg hostuid="$hostuid" \
        --build-arg hostgid="$hostgid" \
        --rm
