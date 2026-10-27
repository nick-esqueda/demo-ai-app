from pathlib import Path

from docs_qa.corpora import Corpus
from docs_qa.ingest import build_page_url, read_pages

BASE = "https://fastapi.tiangolo.com/"


def test_build_page_url():
    assert build_page_url(BASE, "index.md", None) == "https://fastapi.tiangolo.com/"
    assert (
        build_page_url(BASE, "tutorial/index.md", None) == "https://fastapi.tiangolo.com/tutorial/"
    )
    assert (
        build_page_url(BASE, "tutorial/first-steps.md", "recap")
        == "https://fastapi.tiangolo.com/tutorial/first-steps/#recap"
    )


def test_read_pages_respects_exclude(tmp_path: Path):
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

    paths = [path for path, _ in read_pages(corpus, tmp_path)]
    assert paths == ["index.md", "tutorial/intro.md"]
