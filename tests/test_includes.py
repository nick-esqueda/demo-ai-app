from pathlib import Path

from docs_qa.includes import resolve_includes


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
