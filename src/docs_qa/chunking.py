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

# English text averages about 4 characters per token.
CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOKENS = 500
PARAGRAPH_SEPARATOR = "\n\n"
LINE_SEPARATOR = "\n"


@dataclass(frozen=True)
class Chunk:
    source_path: str
    heading_path: tuple[str, ...]
    anchor: str | None
    text: str
    token_count: int


@dataclass(frozen=True)
class _Heading:
    level: int
    title: str
    anchor: str


@dataclass
class _Section:
    heading_path: tuple[str, ...]
    anchor: str | None
    lines: list[str]


def estimate_tokens(text: str) -> int:
    """Rough token count. Good enough for sizing chunks."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def slugify(title: str) -> str:
    """Approximate the anchor that mkdocs generates for a heading."""
    lowercase_words = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[\s_]+", "-", lowercase_words)
    return slug.strip("-")


def is_fence_line(line: str) -> bool:
    """True for a line that opens or closes a fenced code block."""
    return FENCE_RE.match(line) is not None


def _parse_heading(match: re.Match) -> _Heading:
    """Read the level, title and anchor out of a heading line's regex match."""
    level = len(match.group(1))
    title = match.group(2)

    anchor_match = ANCHOR_RE.search(title)
    if anchor_match is None:
        return _Heading(level=level, title=title, anchor=slugify(title))

    title_without_anchor = title[: anchor_match.start()].strip()
    return _Heading(level=level, title=title_without_anchor, anchor=anchor_match.group(1))


def _push_heading(stack: list[_Heading], heading: _Heading) -> None:
    """Replace any headings at the same or a deeper level, so the stack is the current path."""
    while stack and stack[-1].level >= heading.level:
        stack.pop()
    stack.append(heading)


def _split_sections(markdown: str) -> list[_Section]:
    """Walk the page line by line, starting a new section at each heading outside a code fence."""
    heading_stack: list[_Heading] = []
    sections = [_Section(heading_path=(), anchor=None, lines=[])]
    is_inside_fence = False

    for line in markdown.splitlines():
        if is_fence_line(line):
            is_inside_fence = not is_inside_fence

        match = None if is_inside_fence else HEADING_RE.match(line)
        if match is None:
            sections[-1].lines.append(line)
            continue

        heading = _parse_heading(match)
        _push_heading(heading_stack, heading)
        heading_path = tuple(enclosing.title for enclosing in heading_stack)
        sections.append(_Section(heading_path=heading_path, anchor=heading.anchor, lines=[]))

    return sections


def _split_into_blocks(text: str) -> list[str]:
    """Split text into paragraphs, treating each fenced code block as a single paragraph."""
    blocks: list[str] = []
    current_lines: list[str] = []
    is_inside_fence = False

    for line in text.splitlines():
        if is_fence_line(line):
            is_inside_fence = not is_inside_fence

        is_paragraph_break = not line.strip() and not is_inside_fence
        if not is_paragraph_break:
            current_lines.append(line)
        elif current_lines:
            blocks.append(LINE_SEPARATOR.join(current_lines))
            current_lines = []

    if current_lines:
        blocks.append(LINE_SEPARATOR.join(current_lines))
    return blocks


def _pack_into_groups(pieces: list[str], max_tokens: int, separator: str) -> list[str]:
    """Greedily join pieces into groups that stay under max_tokens."""
    groups: list[str] = []
    current_group = ""

    for piece in pieces:
        candidate = f"{current_group}{separator}{piece}" if current_group else piece
        is_over_cap = estimate_tokens(candidate) > max_tokens
        if current_group and is_over_cap:
            groups.append(current_group)
            current_group = piece
        else:
            current_group = candidate

    if current_group:
        groups.append(current_group)
    return groups


def _split_long_line(line: str, max_chars: int) -> list[str]:
    """Cut a line into pieces of at most max_chars. An empty line stays as one empty piece."""
    starts = range(0, max(len(line), 1), max_chars)
    return [line[start : start + max_chars] for start in starts]


def _split_oversized_block(block: str, max_tokens: int) -> list[str]:
    """Split one paragraph or code block that is too big on its own: by lines, then characters."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    pieces: list[str] = []
    for line in block.splitlines():
        pieces.extend(_split_long_line(line, max_chars))
    return _pack_into_groups(pieces, max_tokens, LINE_SEPARATOR)


def _split_to_fit(text: str, max_tokens: int) -> list[str]:
    if estimate_tokens(text) <= max_tokens:
        return [text]

    pieces: list[str] = []
    for block in _split_into_blocks(text):
        if estimate_tokens(block) <= max_tokens:
            pieces.append(block)
        else:
            pieces.extend(_split_oversized_block(block, max_tokens))

    return _pack_into_groups(pieces, max_tokens, PARAGRAPH_SEPARATOR)


def chunk_markdown(
    markdown: str, source_path: str, max_tokens: int = DEFAULT_MAX_TOKENS
) -> list[Chunk]:
    chunks: list[Chunk] = []

    for section in _split_sections(markdown):
        body = LINE_SEPARATOR.join(section.lines).strip()
        if not body:
            continue

        for text in _split_to_fit(body, max_tokens):
            chunk = Chunk(
                source_path=source_path,
                heading_path=section.heading_path,
                anchor=section.anchor,
                text=text,
                token_count=estimate_tokens(text),
            )
            chunks.append(chunk)

    return chunks
