#!/bin/bash

IMAGE_NAME=${IMAGE_NAME:-ghcr.io/iadi-nancy/grics-torch}
IMAGE_VERSION=${IMAGE_VERSION:-1.0.1}
DOCKERFILE=${DOCKERFILE:-build/Dockerfile}

IMAGE="${IMAGE_NAME}:${IMAGE_VERSION}"

CONTAINER_SUFFIX=grics-torch

# Directory containing this script, i.e. the GRICS-torch repository
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parent workspace directory on the host.
# Expected layout:
#   parent/
#   ├── GRICS-torch/
#   ├── project_a/
#   ├── project_b/
#   └── data/
#
# This whole directory will be mounted into the container as:
#   /home/pyuser/wkdir
#
# Therefore project_a, project_b and data do NOT need to be added to EXTRA_MOUNTS
# as long as they are siblings of GRICS-torch inside this parent directory.
WKDIR="$(dirname "$REPO_DIR")"

# Repository folder name, usually "GRICS-torch"
REPO_NAME="$(basename "$REPO_DIR")"

EXTRAS=( )

if [ -n "${EXTRA_MOUNTS:-}" ]; then
 # shellcheck disable=SC2206
 EXTRAS+=( ${EXTRA_MOUNTS} )
fi

GPU_NUMBER=0

#########################################

# Avoid root ownership on these files
mkdir -p "$WKDIR"
mkdir -p ~/.ssh
touch ~/.gitconfig

# Main workspace mount:
# host parent directory  -> /home/pyuser/wkdir in container
#
# With the recommended host layout, this automatically exposes:
#   /home/<user>/wkdir/GRICS-torch
#   /home/<user>/wkdir/project_a
#   /home/<user>/wkdir/project_b
#   /home/<user>/wkdir/data
MOUNTS=(
  -v "$WKDIR":/home/pyuser/wkdir
  -v ~/.ssh:/home/pyuser/.ssh
  -v ~/.gitconfig:/home/pyuser/.gitconfig
)

GPU=( --gpus device=$GPU_NUMBER )
#GPU=( )

# https://man7.org/linux/man-pages/man7/shm_overview.7.html
SHM=( )
#SHM=( --shm-size=256mb )

USERCUT=$(echo $USER | cut -d\@ -f1)
DOCKER_NAME="${USERCUT}-${CONTAINER_SUFFIX}"

case "$1" in

run)
 echo "Run docker " $DOCKER_NAME

 if [ -z "$(docker ps -a | grep "$DOCKER_NAME")" ]; then
  docker run -d ${EXTRAS[@]} ${MOUNTS[@]} ${GPU[@]} ${SHM[@]} \
   --name ${DOCKER_NAME} \
   -w /home/pyuser/wkdir/${REPO_NAME} \
   -e USERCUT=${USERCUT} \
   -e UID=$(id -u) \
   -e GID=$(id -g) \
   $IMAGE
 else
  docker start ${DOCKER_NAME}
 fi

 printf "Waiting for startup..."
 while [ -z "$(docker exec ${DOCKER_NAME} sh -c 'ls -a /home/pyuser | grep .bashrc')" ]; do
  printf "."
  sleep 1
 done

 echo 'OK'
 echo "To use your container in VSCode, please copy/paste following json in the image-config-file of ${IMAGE} :"
 echo '{
 "workspaceFolder": "/home/pyuser/wkdir",
 "updateRemoteUserUID": false,
 "containerUser": "root",
 "remoteUser": "pyuser",
 "extensions": [
 "ms-python.python",
 "ms-python.vscode-pylance",
 "ms-toolsai.jupyter",
 "ms-toolsai.jupyter-keymap",
 "ms-toolsai.jupyter-renderers"
 ]
}'
 ;;

exec)
 PARAMS=${@:2}
 if [ -z "${2}" ]; then
  PARAMS="/bin/bash"
 fi
 docker exec --user=$(id -u) $DOCKER_NAME ${PARAMS}
 ;;

exec-it)
 PARAMS=${@:2}
 if [ -z "${2}" ]; then
  PARAMS="/bin/bash"
 fi
 docker exec --user=$(id -u) -it $DOCKER_NAME ${PARAMS}
 ;;

exec-it-root)
 PARAMS=${@:2}
 if [ -z "${2}" ]; then
  PARAMS="/bin/bash"
 fi
 docker exec --user=0 -it $DOCKER_NAME ${PARAMS}
 ;;

start)
 docker start $DOCKER_NAME
 ;;

stop)
 docker stop $DOCKER_NAME
 ;;

rm)
 docker stop $DOCKER_NAME
 docker rm $DOCKER_NAME
 ;;

build)
 docker build -f $DOCKERFILE -t $IMAGE --build-arg PYTORCH_VER=${IMAGE_VERSION} --network=host --rm=true build/
 ;;

build-nocache)
 docker build -f $DOCKERFILE -t $IMAGE --build-arg PYTORCH_VER=${IMAGE_VERSION} --network=host --no-cache --rm=true build/
 ;;

push)
 docker push -a ${IMAGE_NAME}
 ;;

*)
 echo "Usage : start-docker.sh (run|exec [command]|exec-it [command]|exec-it-root [command]|stop|rm|prune|build|build-nocache)"
 ;;

esac