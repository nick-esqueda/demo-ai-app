from docs_qa.chunking import chunk_markdown, estimate_tokens, slugify

PAGE = """\
# First Steps { #first-steps }

Intro paragraph.

## Run it

Use the CLI.

```python
# this is a comment, not a heading
app = FastAPI()
```

### Check it

Open the browser.

## Recap

Done.
"""


def test_splits_on_headings_and_tracks_heading_path():
    chunks = chunk_markdown(PAGE, "tutorial/first-steps.md")

    assert [c.heading_path for c in chunks] == [
        ("First Steps",),
        ("First Steps", "Run it"),
        ("First Steps", "Run it", "Check it"),
        ("First Steps", "Recap"),
    ]
    assert all(c.source_path == "tutorial/first-steps.md" for c in chunks)


def test_uses_explicit_anchor_or_slug():
    chunks = chunk_markdown(PAGE, "x.md")
    assert chunks[0].anchor == "first-steps"
    assert chunks[1].anchor == "run-it"


def test_ignores_hash_lines_inside_code_fences():
    chunks = chunk_markdown(PAGE, "x.md")
    run_it = chunks[1]
    assert "# this is a comment, not a heading" in run_it.text
    assert "app = FastAPI()" in run_it.text


def test_skips_headings_with_no_body():
    chunks = chunk_markdown("# Title\n\n## Section\n\nBody text.\n", "x.md")
    assert len(chunks) == 1
    assert chunks[0].heading_path == ("Title", "Section")


def test_long_sections_are_split_under_the_cap():
    paragraphs = "\n\n".join(f"Paragraph {i}. " + "word " * 80 for i in range(20))
    chunks = chunk_markdown(f"# Big\n\n{paragraphs}\n", "big.md", max_tokens=200)

    assert len(chunks) > 1
    assert all(c.token_count <= 200 for c in chunks)
    assert all(c.heading_path == ("Big",) for c in chunks)
    # Paragraphs are kept whole, not cut mid-way.
    assert all(c.text.startswith("Paragraph") for c in chunks)


def test_oversized_single_block_is_still_split():
    code = "```python\n" + "\n".join(f"x_{i} = {i}" for i in range(500)) + "\n```"
    chunks = chunk_markdown(f"# Code\n\n{code}\n", "code.md", max_tokens=100)
    assert len(chunks) > 1
    assert all(c.token_count <= 100 for c in chunks)


def test_helpers():
    assert slugify("Path Parameters with Types!") == "path-parameters-with-types"
    assert estimate_tokens("a" * 400) == 100
