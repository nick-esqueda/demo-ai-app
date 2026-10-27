import subprocess
from pathlib import Path

from docs_qa.corpora import Corpus
from docs_qa.ingest import fetch_corpus, iter_pages, page_url, resolve_includes

BASE = "https://fastapi.tiangolo.com/"


def test_page_url():
    assert page_url(BASE, "index.md", None) == "https://fastapi.tiangolo.com/"
    assert page_url(BASE, "tutorial/index.md", None) == "https://fastapi.tiangolo.com/tutorial/"
    assert (
        page_url(BASE, "tutorial/first-steps.md", "recap")
        == "https://fastapi.tiangolo.com/tutorial/first-steps/#recap"
    )


def test_resolve_includes(tmp_path: Path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs_src").mkdir()
    (tmp_path / "docs_src" / "app.py").write_text("app = FastAPI()\n")

    markdown = (
        "Example:\n\n{* ../docs_src/app.py hl[1] *}\n\n"
        "```Python\n{!../docs_src/app.py!}\n```\n\n"
        "{* ../docs_src/missing.py *}"
    )
    resolved = resolve_includes(markdown, tmp_path / "docs")

    assert "```python\napp = FastAPI()\n```" in resolved
    assert "```Python\napp = FastAPI()\n```" in resolved
    assert "[code sample: ../docs_src/missing.py]" in resolved


def test_iter_pages_respects_exclude(tmp_path: Path):
    docs = tmp_path / "docs" / "en" / "docs"
    (docs / "tutorial").mkdir(parents=True)
    (docs / "index.md").write_text("# Home\n")
    (docs / "tutorial" / "intro.md").write_text("# Intro\n")
    (docs / "release-notes.md").write_text("# Release Notes\n")
    corpus = Corpus(
        name="test",
        repo_url="unused",
        ref="unused",
        docs_dir="docs/en/docs",
        base_url=BASE,
        exclude=("release-notes.md",),
    )

    paths = [path for path, _ in iter_pages(corpus, tmp_path)]
    assert paths == ["index.md", "tutorial/intro.md"]


def test_fetch_corpus_checks_out_pinned_ref(tmp_path: Path):
    """Uses a local git repo as the "remote", so no network is needed."""

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=remote, check=True, capture_output=True)

    remote = tmp_path / "remote"
    (remote / "docs" / "en" / "docs").mkdir(parents=True)
    (remote / "other").mkdir()
    (remote / "docs" / "en" / "docs" / "index.md").write_text("# v1\n")
    (remote / "other" / "big.bin").write_text("not needed")
    git("init", "--quiet")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "--quiet", "-m", "v1")
    git("tag", "v1")
    (remote / "docs" / "en" / "docs" / "index.md").write_text("# v2\n")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "--quiet", "-am", "v2")

    corpus = Corpus(
        name="local",
        repo_url=remote.as_uri(),
        ref="v1",
        docs_dir="docs/en/docs",
        base_url=BASE,
    )
    repo_dir = fetch_corpus(corpus, tmp_path / "data")

    assert (repo_dir / "docs/en/docs/index.md").read_text() == "# v1\n"
    assert not (repo_dir / "other").exists()  # sparse checkout skips other folders
    # Re-running is safe.
    assert fetch_corpus(corpus, tmp_path / "data") == repo_dir
