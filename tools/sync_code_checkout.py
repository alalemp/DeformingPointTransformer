"""Fast-forward an approved code snapshot while preserving existing untracked files."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sync_checkout(repo, base, commit, manifest):
    def git(*args):
        return subprocess.run(['git', '-C', str(repo), *args], check=True,
                              universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()

    if git('branch', '--show-current') != 'ranxi-dev':
        raise RuntimeError('Checkout is no longer on ranxi-dev')
    head = git('rev-parse', 'HEAD')
    if head not in (base, commit):
        raise RuntimeError('Checkout changed since the approved snapshot')
    if git('diff', '--cached', '--name-only'):
        raise RuntimeError('Existing staged changes require review')
    for name, entry in manifest.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Invalid snapshot path')
        p = repo / relative
        if p.is_symlink():
            raise RuntimeError('Refusing a symlink: ' + name)
        if p.exists() and digest(p) not in entry['allowed_existing_hashes']:
            raise RuntimeError('File changed since review: ' + name)
        if not p.exists() and entry.get('must_exist', False) and head == base:
            raise RuntimeError('Reviewed file was removed: ' + name)
    if head == commit:
        return {'commit': commit, 'already_synced': True}
    git('fetch', 'origin', 'ranxi-dev')
    if git('rev-parse', 'origin/ranxi-dev') != commit:
        raise RuntimeError('Remote branch changed; refusing to sync a stale snapshot')
    git('merge-base', '--is-ancestor', base, commit)
    changed = set(git('diff', '--name-only', base, commit).splitlines())
    if changed != set(manifest):
        raise RuntimeError('Incoming changes do not match the reviewed file list')

    backup = Path(tempfile.mkdtemp(prefix='dpt-code-sync-'))
    saved = []
    try:
        for name in manifest:
            p = repo / name
            if not p.exists():
                continue
            target = backup / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            saved.append(name)
            tracked = subprocess.run(['git', '-C', str(repo), 'ls-files',
                                      '--error-unmatch', '--', name],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0
            if tracked:
                # Only restore the reviewed path. Its modified copy remains in backup.
                git('restore', '--source=HEAD', '--worktree', '--', name)
            else:
                p.unlink()
        git('merge', '--ff-only', 'origin/ranxi-dev')
        if git('rev-parse', 'HEAD') != commit:
            raise RuntimeError('Unexpected checkout commit')
        for name, entry in manifest.items():
            if digest(repo / name) != entry['new_hash']:
                raise RuntimeError('Synced file hash mismatch: ' + name)
    except Exception:
        # Git usually refuses before changing HEAD. Recover reviewed files in that case.
        if git('rev-parse', 'HEAD') == base:
            for name in saved:
                p = repo / name
                p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup / name, p)
        raise
    return {'commit': commit, 'backup_directory': str(backup)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--manifest-json', required=True)
    args = parser.parse_args()
    print(json.dumps(sync_checkout(args.repo, args.base, args.commit,
                                   json.loads(args.manifest_json))))


if __name__ == '__main__':
    main()
