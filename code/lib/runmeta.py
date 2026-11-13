"""Метадані прогону: час, коміт git (і чи є незакомічені зміни), sha256 скрипта, версії, параметри, сіди."""
import datetime
import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

PKGS = ("numpy", "scipy", "networkx", "numba", "torch", "torch_geometric", "scikit-learn")


def git_state(path):
    """(коміт, dirty) робочої копії, що містить path; (None, None) поза git."""
    try:
        run = lambda *a: subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True,
                                        check=True).stdout.strip()
        return run("rev-parse", "HEAD"), bool(run("status", "--porcelain", "--untracked-files=no"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def runmeta(script, params=None, seeds=None):
    vers = {}
    for p in PKGS:
        try:
            vers[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    src = Path(script).resolve()
    commit, dirty = git_state(src.parent)
    return dict(time=datetime.datetime.now().isoformat(timespec="seconds"), script=src.name,
                git_commit=commit, git_dirty=dirty,
                script_sha256=hashlib.sha256(src.read_bytes()).hexdigest(), python=sys.version.split()[0],
                machine=platform.platform(), versions=vers, params=params or {}, seeds=seeds)


def save(path, meta, **results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(meta=meta, **results), ensure_ascii=False, indent=1), encoding="utf-8")
    return path
