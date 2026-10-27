import subprocess
from pathlib import Path

from docs_qa.corpora import Corpus
from docs_qa.git_checkout import fetch_corpus

BASE_URL = "https://fastapi.tiangolo.com/"


def test_fetch_corpus_checks_out_pinned_ref(tmp_path: Path):
    """Uses a local git repo as the "remote", so no network is needed."""

    def run_git_in_remote(*args: str) -> None:
        subprocess.run(["git", *args], cwd=remote, check=True, capture_output=True)

    remote = tmp_path / "remote"
    (remote / "docs" / "en" / "docs").mkdir(parents=True)
    (remote / "other").mkdir()
    (remote / "docs" / "en" / "docs" / "index.md").write_text("# v1\n")
    (remote / "other" / "big.bin").write_text("not needed")
    run_git_in_remote("init", "--quiet")
    run_git_in_remote("add", ".")
    run_git_in_remote("-c", "user.name=t", "-c", "user.email=t@t", "commit", "--quiet", "-m", "v1")
    run_git_in_remote("tag", "v1")
    (remote / "docs" / "en" / "docs" / "index.md").write_text("# v2\n")
    run_git_in_remote("-c", "user.name=t", "-c", "user.email=t@t", "commit", "--quiet", "-am", "v2")

    corpus = Corpus(
        name="local",
        repo_url=remote.as_uri(),
        ref="v1",
        docs_dir="docs/en/docs",
        base_url=BASE_URL,
    )
    repo_dir = fetch_corpus(corpus, tmp_path / "data")

    assert (repo_dir / "docs/en/docs/index.md").read_text() == "# v1\n"
    assert not (repo_dir / "other").exists()  # sparse checkout skips other folders
    # Re-running is safe.
    assert fetch_corpus(corpus, tmp_path / "data") == repo_dir
