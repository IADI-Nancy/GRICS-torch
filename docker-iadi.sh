#!/usr/bin/env bash
set -euo pipefail

# Internal use only (IADI environment wrapper around docker.sh).

mkdir -p "${HOME}/.cicit-nancy"

export IMAGE_NAME="virtus.iadi.lan:8123/iadi/grics-torch"
export IMAGE_VERSION="2.3.0-cuda12.1-cudnn8-devel"
export DOCKERFILE="build/Dockerfile-iadi"
export EXTRA_MOUNTS="-v ${HOME}/.cicit-nancy:/home/pyuser/.cicit-nancy -v .bashrcoverride:/home/pyuser/.bashrcoverride"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/docker.sh" "$@"
