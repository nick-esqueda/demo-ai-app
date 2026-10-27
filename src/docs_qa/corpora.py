"""Document corpora the app can ingest. Each one is a docs folder in a git repo, pinned to a ref."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Corpus:
    name: str
    repo_url: str
    # A tag or commit SHA. Pinning it keeps ingestion (and later, evals) reproducible.
    ref: str
    # Folder inside the repo that holds the Markdown docs.
    docs_dir: str
    # Public site that the docs folder is published to, used to build source URLs.
    base_url: str
    # Extra folders to check out, e.g. code samples that the docs include by reference.
    extra_dirs: tuple[str, ...] = ()
    # Pages (paths inside docs_dir) to leave out of the index.
    exclude: tuple[str, ...] = ()


CORPORA: dict[str, Corpus] = {
    "fastapi": Corpus(
        name="fastapi",
        repo_url="https://github.com/fastapi/fastapi.git",
        ref="0.115.12",
        docs_dir="docs/en/docs",
        base_url="https://fastapi.tiangolo.com/",
        extra_dirs=("docs_src",),
        # The release notes are huge and mostly version bumps, so they drown out real answers.
        exclude=("release-notes.md",),
    ),
    # Pydantic's docs are also Markdown and should slot in the same way, e.g.:
    # "pydantic": Corpus(
    #     name="pydantic",
    #     repo_url="https://github.com/pydantic/pydantic.git",
    #     ref="<tag>",
    #     docs_dir="docs",
    #     base_url="https://docs.pydantic.dev/latest/",
    # ),
    # SQLAlchemy's docs are reStructuredText, so they need their own parser first.
}
