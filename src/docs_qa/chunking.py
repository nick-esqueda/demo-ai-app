"""Split a Markdown page into chunks that follow its heading structure.

Each heading starts a new section. A section that is longer than the token cap is split
further on paragraph boundaries, keeping fenced code blocks whole where possible.
"""

import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
# mkdocs-style explicit anchor at the end of a heading: "## First Steps { #first-steps }"
ANCHOR_RE = re.compile(r"\s*\{\s*#([\w-]+)\s*\}\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Chunk:
    source_path: str
    heading_path: tuple[str, ...]
    anchor: str | None
    text: str
    token_count: int


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 characters per token for English). Good enough for sizing chunks."""
    return max(1, len(text) // 4)


def slugify(title: str) -> str:
    """Approximate the anchor that mkdocs generates for a heading."""
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    return re.sub(r"[\s_]+", "-", slug).strip("-")


@dataclass
class _Section:
    heading_path: tuple[str, ...]
    anchor: str | None
    lines: list[str]


def _split_sections(markdown: str) -> list[_Section]:
    """Walk the page line by line, starting a new section at each heading outside a code fence."""
    stack: list[tuple[int, str, str]] = []  # (level, title, anchor) of the enclosing headings
    sections = [_Section(heading_path=(), anchor=None, lines=[])]
    in_fence = False

    for line in markdown.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
        match = None if in_fence else HEADING_RE.match(line)
        if not match:
            sections[-1].lines.append(line)
            continue

        level, title = len(match.group(1)), match.group(2)
        anchor_match = ANCHOR_RE.search(title)
        if anchor_match:
            title = title[: anchor_match.start()].strip()
            anchor = anchor_match.group(1)
        else:
            anchor = slugify(title)

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title, anchor))
        sections.append(
            _Section(heading_path=tuple(t for _, t, _ in stack), anchor=anchor, lines=[])
        )

    return sections


def _blocks(text: str) -> list[str]:
    """Split text into paragraphs, treating each fenced code block as a single paragraph."""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _pack(pieces: list[str], max_tokens: int, separator: str) -> list[str]:
    """Greedily join pieces into groups that stay under max_tokens."""
    groups: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{separator}{piece}" if current else piece
        if current and estimate_tokens(candidate) > max_tokens:
            groups.append(current)
            current = piece
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


def _split_to_fit(text: str, max_tokens: int) -> list[str]:
    if estimate_tokens(text) <= max_tokens:
        return [text]

    pieces: list[str] = []
    for block in _blocks(text):
        if estimate_tokens(block) <= max_tokens:
            pieces.append(block)
            continue
        # One paragraph or code block is too big on its own: fall back to lines, then characters.
        lines: list[str] = []
        max_chars = max_tokens * 4
        for line in block.splitlines():
            lines.extend(line[i : i + max_chars] for i in range(0, max(len(line), 1), max_chars))
        pieces.extend(_pack(lines, max_tokens, "\n"))

    return _pack(pieces, max_tokens, "\n\n")


def chunk_markdown(markdown: str, source_path: str, max_tokens: int = 500) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in _split_sections(markdown):
        body = "\n".join(section.lines).strip()
        if not body:
            continue
        for text in _split_to_fit(body, max_tokens):
            chunks.append(
                Chunk(
                    source_path=source_path,
                    heading_path=section.heading_path,
                    anchor=section.anchor,
                    text=text,
                    token_count=estimate_tokens(text),
                )
            )
    return chunks
