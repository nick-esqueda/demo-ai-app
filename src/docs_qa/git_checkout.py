"""Fetch a corpus's docs from git: a sparse, shallow checkout at the pinned ref."""

import subprocess
from pathlib import Path

from docs_qa.corpora import Corpus


def run_git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def init_repo(repo_dir: Path, repo_url: str) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    run_git("init", "--quiet", cwd=repo_dir)
    run_git("remote", "add", "origin", repo_url, cwd=repo_dir)


def fetch_corpus(corpus: Corpus, data_dir: Path) -> Path:
    """Check out just the docs folders at the pinned ref. Safe to re-run."""
    repo_dir = data_dir / corpus.name
    has_repo = (repo_dir / ".git").exists()
    if not has_repo:
        init_repo(repo_dir, corpus.repo_url)

    run_git("sparse-checkout", "set", corpus.docs_dir, *corpus.extra_dirs, cwd=repo_dir)
    run_git("fetch", "--quiet", "--depth", "1", "origin", corpus.ref, cwd=repo_dir)
    run_git("checkout", "--quiet", "--force", "FETCH_HEAD", cwd=repo_dir)
    return repo_dir
