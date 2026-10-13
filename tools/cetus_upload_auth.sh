#!/usr/bin/env bash
# Unlock once in a local terminal; the master connection continues in the background.
set -euo pipefail
socket_dir=/tmp/cetus-upload-rxliu
mkdir -p "$socket_dir"
chmod 700 "$socket_dir"
printf 'Unlock the Cetus SSH key in this local window.\n'
printf 'The upload process will reuse this connection after authentication.\n'
ssh -f -M -N -o ControlMaster=yes -o ControlPersist=12h -o "ControlPath=$socket_dir/control" \
    -o ConnectTimeout=15 -o ServerAliveInterval=60 -o ServerAliveCountMax=3 cetus
printf '\nCetus SSH connection is ready in the background. You can leave this window alone.\n'
