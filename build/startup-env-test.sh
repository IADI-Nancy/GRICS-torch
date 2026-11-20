#!/usr/bin/env bash
set -euo pipefail

# Match host ownership without recursively changing mounted workspace or SSH files.
if ! id pyuser >/dev/null 2>&1; then
    if ! getent group "$HOST_GID" >/dev/null; then
        groupadd --gid "$HOST_GID" pyuser
    fi
    useradd --non-unique --uid "$HOST_UID" --gid "$HOST_GID" --home-dir /home/pyuser --no-create-home --shell /bin/bash pyuser
fi
if [ ! -f /home/pyuser/.bashrc ]; then
    cp /etc/skel/.bashrc /home/pyuser/.bashrc
    cp /etc/skel/.profile /home/pyuser/.profile
fi
chown "$HOST_UID:$HOST_GID" /home/pyuser /home/pyuser/.bashrc /home/pyuser/.profile
printf 'Clean Python environment ready; Torch, CuPy and CUDA toolkit are not preinstalled.\n'
touch /tmp/env-test-ready
exec sleep infinity
