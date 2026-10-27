"""Inline the code samples that FastAPI's docs pull in from other files.

"{* ../../docs_src/app.py hl[3] *}" renders as a code block, and "{!../../docs_src/app.py!}"
pastes the file inside an existing code fence. Inlining them means the code gets indexed too.
"""

import re
from pathlib import Path

CODE_BLOCK_INCLUDE_RE = re.compile(r"\{\*\s*(\S+)[^*]*\*\}")
RAW_INCLUDE_RE = re.compile(r"\{!\s*(\S+?)\s*!\}")


def read_included_file(include_root: Path, path: str) -> str | None:
    """The file's text, or None if it doesn't exist."""
    file = include_root / path
    if not file.is_file():
        return None
    return file.read_text(encoding="utf-8").rstrip()


def resolve_includes(markdown: str, include_root: Path) -> str:
    def replace_code_block_include(match: re.Match) -> str:
        path = match.group(1)
        code = read_included_file(include_root, path)
        return f"```python\n{code}\n```" if code is not None else f"[code sample: {path}]"

    def replace_raw_include(match: re.Match) -> str:
        path = match.group(1)
        code = read_included_file(include_root, path)
        return code if code is not None else f"# code sample: {path}"

    with_code_blocks = CODE_BLOCK_INCLUDE_RE.sub(replace_code_block_include, markdown)
    return RAW_INCLUDE_RE.sub(replace_raw_include, with_code_blocks)
