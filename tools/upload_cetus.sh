#!/usr/bin/env bash
# Run on the local WSL machine. The SSH key passphrase is entered in its terminal.
# Usage: bash tools/upload_cetus.sh [dataset_code]
set -Eeuo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_code="${1:-cactus_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220}"
[[ "$dataset_code" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid dataset code'; exit 2; }
metadata="$repo_dir/data/upload_configs/$dataset_code/parameters.json"
if [[ ! -f "$metadata" ]]; then
    metadata="$repo_dir/configs/upload/$dataset_code.example.json"
fi
[[ -f "$metadata" ]] || { echo "Missing parameter record: $metadata"; exit 2; }
command -v rsync >/dev/null
command -v ssh >/dev/null

mapfile -t fields < <(python3 - "$metadata" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for key in ('local_source', 'cetus_destination', 'original_bytes'):
    print(m[key])
PY
)
[[ ${#fields[@]} -eq 3 ]] || { echo 'Invalid parameter record'; exit 2; }
source_dir="${fields[0]}"
remote_dir="${fields[1]%/}"
source_bytes="${fields[2]}"
[[ -d "$source_dir" ]] || { echo "Missing source data: $source_dir"; exit 2; }
[[ "$remote_dir" == "/shared/homes/u24615736/git/DeformingPointTransformer/data/$dataset_code" ||
   "$remote_dir" == "/shared/homes/u24615736/datasets/houdini/$dataset_code" ]] || {
    echo 'Destination does not match the authorized dataset directory'; exit 2;
}
for child in 'bones' 'final skin' 'separated muscles' 'entire muscles'; do
    [[ -d "$source_dir/$child" ]] || { echo "Missing source folder: $child"; exit 2; }
done

record_dir="$repo_dir/results/cetus_upload/$dataset_code"
mkdir -p -- "$record_dir"
log_file="$record_dir/upload.log"
status_file="$record_dir/status.json"
exec > >(tee -a "$log_file") 2>&1

set_status() {
    python3 - "$status_file" "$dataset_code" "$1" "${2:-0}" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json, sys
p = Path(sys.argv[1])
value = dict(dataset_code=sys.argv[2], phase=sys.argv[3], exit_code=int(sys.argv[4]),
             updated_at_utc=datetime.now(timezone.utc).isoformat())
tmp = p.with_suffix('.tmp')
tmp.write_text(json.dumps(value, indent=2) + '\n')
tmp.replace(p)
PY
}

finish() {
    result=$?
    trap - EXIT
    if (( result == 0 )) && [[ "${upload_completed:-0}" != 1 ]]; then
        result=130
    fi
    if (( result != 0 )); then
        set_status failed "$result"
        printf '\nUpload did not complete (exit %s). Rerun this script to resume.\n' "$result"
    fi
    printf '\nLog: %s\n' "$log_file"
    if [[ -t 0 ]]; then
        read -r -p 'Press Enter to close this upload terminal...' || true
    fi
    exit "$result"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

socket_dir="$(mktemp -d /tmp/cetus-upload.XXXXXX)"
control_path="${CETUS_UPLOAD_CONTROL_PATH:-$socket_dir/%C}"
ssh_options=(-o ConnectTimeout=15 -o ServerAliveInterval=60 -o ServerAliveCountMax=3
             -o ControlMaster=auto -o ControlPersist=600 -o "ControlPath=$control_path")
printf -v ssh_transport '%q ' ssh "${ssh_options[@]}"
quoted_remote="$(python3 -c 'import shlex,sys; print(shlex.quote(sys.argv[1]))' "$remote_dir")"

printf '\nDataset: %s\nSource: %s\nDestination: cetus:%s/\n' "$dataset_code" "$source_dir" "$remote_dir"
printf 'Source size recorded: %s bytes (about 23 GiB).\n' "$source_bytes"
printf 'Enter your existing Cetus SSH key passphrase if prompted.\n\n'
set_status waiting_for_authentication

# This first connection asks for the key passphrase; subsequent commands reuse it.
ssh "${ssh_options[@]}" cetus "mkdir -p -- $quoted_remote && command -v rsync && df -Pk -- $quoted_remote" > "$record_dir/remote_preflight.txt"
remote_info="$(cat "$record_dir/remote_preflight.txt")"
printf '%s\n' "$remote_info"
free_kib="$(printf '%s\n' "$remote_info" | awk '$2 ~ /^[0-9]+$/ && $4 ~ /^[0-9]+$/ { free=$4 } END { print free }')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || { echo 'Could not determine destination filesystem space'; exit 2; }
if (( free_kib * 1024 < source_bytes )); then
    echo 'Destination filesystem has less free space than the full dataset. Check available space before resuming.'
    exit 2
fi

set_status transferring
printf '\nUploading raw data...\n'
rsync -rtvh --partial-dir=.rsync-partial --info=progress2 \
    -e "$ssh_transport" -- "$source_dir/" "cetus:$remote_dir/"

printf '\nUploading parameter record...\n'
rsync -tv -e "$ssh_transport" -- "$metadata" "cetus:$remote_dir/"

set_status verifying
printf '\nChecking for files still needing synchronization...\n'
verification="$record_dir/remaining_changes.txt"
# Ignore directory timestamps: adding parameters.json changes the destination root mtime.
rsync -rtni --omit-dir-times -e "$ssh_transport" -- "$source_dir/" "cetus:$remote_dir/" > "$verification"
rsync -tni -e "$ssh_transport" -- "$metadata" "cetus:$remote_dir/" >> "$verification"
if [[ -s "$verification" ]]; then
    cat "$verification"
    echo 'Files still need synchronization. Rerun this script.'
    exit 3
fi

printf '\nRemote dataset check:\n'
ssh "${ssh_options[@]}" cetus "du -sh -- $quoted_remote; find $quoted_remote -type f | wc -l"
set_status completed
upload_completed=1
printf '\nUPLOAD COMPLETE: raw data and parameters.json synchronized.\n'
