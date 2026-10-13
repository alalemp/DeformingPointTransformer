"""Move a verified Cetus upload outside the code repository without overwriting data."""
import argparse
import json
import shlex
import subprocess
from pathlib import Path


REMOTE_MOVE = r'''
import json, os, sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:3])
code = sys.argv[3]
if source.is_symlink() or destination.is_symlink():
    raise SystemExit("Refusing a symlink as the source or destination")
already_moved = not source.exists() and destination.is_dir()
if source.exists() and destination.exists():
    raise SystemExit("Destination already exists; source left untouched")
root = destination if already_moved else source
if not root.is_dir():
    raise SystemExit("Dataset directory is missing")
record = json.loads((root / "parameters.json").read_text())
if record["dataset_code"] != code:
    raise SystemExit("Dataset code does not match")
for name in ("bones", "final skin", "separated muscles", "entire muscles"):
    if not (root / name).is_dir():
        raise SystemExit("Missing required dataset folder: " + name)
files = []
for directory, _, names in os.walk(root):
    for name in names:
        p = Path(directory) / name
        if p.is_symlink():
            raise SystemExit("Unexpected symlink in dataset")
        files.append(p)
if len(files) != record["original_file_count"] + 1:
    raise SystemExit("File count differs from the verified upload record")
total = sum(p.stat().st_size for p in files)
expected = record["original_bytes"] + (root / "parameters.json").stat().st_size
if total != expected:
    raise SystemExit("File sizes differ from the verified upload record")
if not already_moved:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A rename on the same filesystem is atomic. Cross-filesystem moves fail safely.
    os.rename(source, destination)
print(json.dumps(dict(destination=str(destination), file_count=len(files),
                      total_bytes=total, already_moved=already_moved)))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upload-record', type=Path, required=True)
    parser.add_argument('--upload-status', type=Path, required=True)
    parser.add_argument('--destination-parent', required=True)
    parser.add_argument('--control-path', required=True)
    args = parser.parse_args()
    record = json.loads(args.upload_record.read_text())
    status = json.loads(args.upload_status.read_text())
    code = record['dataset_code']
    if status.get('dataset_code') != code or status.get('phase') != 'completed' or status.get('exit_code') != 0:
        raise SystemExit('Upload has not completed successfully; no data moved')
    source = record['cetus_destination'].rstrip('/')
    destination = str(Path(args.destination_parent) / code)
    if source != '/shared/homes/u24615736/git/DeformingPointTransformer/data/' + code:
        raise SystemExit('Source is not the authorized staging directory')
    if args.destination_parent.rstrip('/') != '/shared/homes/u24615736/datasets/houdini':
        raise SystemExit('Destination is not the user-selected directory')
    command = 'python3 -c ' + shlex.quote(REMOTE_MOVE) + ' ' + ' '.join(
        shlex.quote(value) for value in (source, destination, code))
    result = subprocess.run(['ssh', '-S', args.control_path, '-o', 'BatchMode=yes',
                             '-o', 'ConnectTimeout=15', 'cetus', command],
                            check=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(result.stdout.strip())


if __name__ == '__main__':
    main()
