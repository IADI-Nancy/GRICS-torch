#!/usr/bin/env bash
set -euo pipefail

# Local installation-test settings: Python without Torch or CUDA.
IMAGE_NAME=${IMAGE_NAME:-grics-torch-env-test-iadi}
IMAGE_VERSION=${IMAGE_VERSION:-py311}
DOCKERFILE=${DOCKERFILE:-build/Dockerfile-env-test}
IMAGE="${IMAGE_NAME}:${IMAGE_VERSION}"
CONTAINER_SUFFIX=${CONTAINER_SUFFIX:-grics-torch-env-test-iadi}
# Set IADI_PIP_INDEX='' when building to use public PyPI only.
IADI_PIP_INDEX=${IADI_PIP_INDEX-http://virtus.iadi.lan:8081/repository/virtus-pip/simple/}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WKDIR="$(dirname "$REPO_DIR")"
REPO_NAME="$(basename "$REPO_DIR")"
USERCUT="${USER:-$(id -un)}"
USERCUT="${USERCUT%%@*}"
DOCKER_NAME="${USERCUT}-${CONTAINER_SUFFIX}"

# USE_GPU=0 explicitly selects CPU-only testing; GPU_NUMBER selects the NVIDIA device.
USE_GPU=${USE_GPU:-1}
GPU_NUMBER=${GPU_NUMBER:-0}
GPU=()
case "$USE_GPU" in
    1) GPU=(--gpus "device=$GPU_NUMBER");;
    0) ;;
    *) echo 'USE_GPU must be 0 or 1.' >&2; exit 1;;
esac

MOUNTS=(-v "$WKDIR:/home/pyuser/wkdir")
EXTRAS=()
# Do not source the host shell override: it may introduce Conda or CUDA paths.
mkdir -p "$HOME/.ssh" "$HOME/.cicit-nancy"
MOUNTS+=(-v "$HOME/.ssh:/home/pyuser/.ssh:ro" -v "$HOME/.cicit-nancy:/home/pyuser/.cicit-nancy:ro")
if [ -f "$HOME/.gitconfig" ]; then
    MOUNTS+=(-v "$HOME/.gitconfig:/home/pyuser/.gitconfig:ro")
fi
for NAME in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY; do
    if [ -n "${!NAME:-}" ]; then
        EXTRAS+=(-e "$NAME")
    fi
done

COMMAND=${1:-help}
if [ "$#" -gt 0 ]; then shift; fi
case "$COMMAND" in
    build|build-nocache)
        BUILD_ARGS=()
        if [ "$COMMAND" = build-nocache ]; then BUILD_ARGS+=(--no-cache); fi
        docker build "${BUILD_ARGS[@]}" -f "$REPO_DIR/$DOCKERFILE" -t "$IMAGE" --build-arg "IADI_PIP_INDEX=${IADI_PIP_INDEX:-}" --network=host "$REPO_DIR/build"
        ;;
    run)
        if docker container inspect "$DOCKER_NAME" >/dev/null 2>&1; then
            EXISTING_IMAGE=$(docker inspect --format '{{.Config.Image}}' "$DOCKER_NAME")
            if [ "$EXISTING_IMAGE" != "$IMAGE" ]; then
                echo "Container $DOCKER_NAME uses $EXISTING_IMAGE, expected $IMAGE. Remove the test container with this script's rm command before retrying." >&2
                exit 1
            fi
            docker start "$DOCKER_NAME"
        else
            if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
                echo "Local test image $IMAGE is missing. First run: bash $0 build" >&2
                exit 1
            fi
            docker run --pull=never -d "${EXTRAS[@]}" "${MOUNTS[@]}" "${GPU[@]}" --name "$DOCKER_NAME" -w "/home/pyuser/wkdir/$REPO_NAME" -e "HOST_UID=$(id -u)" -e "HOST_GID=$(id -g)" "$IMAGE"
        fi
        for ((ATTEMPT=0; ATTEMPT<30; ATTEMPT++)); do
            if docker exec "$DOCKER_NAME" test -f /tmp/env-test-ready; then
                echo "Ready: $DOCKER_NAME ($IMAGE). Use exec-it to open a shell."
                exit 0
            fi
            if [ "$(docker inspect --format '{{.State.Running}}' "$DOCKER_NAME")" != true ]; then break; fi
            sleep 1
        done
        docker logs "$DOCKER_NAME"
        echo 'Test container startup failed or timed out.' >&2
        exit 1
        ;;
    exec|exec-it|exec-it-root)
        if ! docker container inspect "$DOCKER_NAME" >/dev/null 2>&1; then
            echo "Test container $DOCKER_NAME is missing. Build the image, then run: bash $0 run" >&2
            exit 1
        fi
        EXEC_ARGS=(--user "$(id -u):$(id -g)" -e HOME=/home/pyuser)
        if [ "$COMMAND" != exec ]; then EXEC_ARGS+=(-it); fi
        if [ "$COMMAND" = exec-it-root ]; then EXEC_ARGS=(--user 0 -e HOME=/root -it); fi
        if [ "$#" -eq 0 ]; then set -- /bin/bash; fi
        docker exec "${EXEC_ARGS[@]}" "$DOCKER_NAME" "$@"
        ;;
    start|stop) docker "$COMMAND" "$DOCKER_NAME";;
    rm)
        if [ "$(docker inspect --format '{{.State.Running}}' "$DOCKER_NAME")" = true ]; then docker stop "$DOCKER_NAME"; fi
        docker rm "$DOCKER_NAME"
        ;;
    *)
        echo "Usage: $0 {build|build-nocache|run|exec [command]|exec-it [command]|exec-it-root [command]|start|stop|rm}"
        echo 'Set USE_GPU=0 for CPU-only testing. Build the image before the first run.'
        if [ "$COMMAND" != help ]; then exit 1; fi
        ;;
esac
