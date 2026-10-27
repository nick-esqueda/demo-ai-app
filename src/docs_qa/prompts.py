"""Prompt text for answering a question from retrieved sources."""

from docs_qa.retrieval import RetrievedChunk

SYSTEM_PROMPT = """\
You answer questions about Python library documentation using only the numbered sources \
provided with each question.

- Base every statement on the sources. Do not use outside knowledge, even if you are confident.
- Cite the sources you used with their numbers in square brackets right after the statement \
they support, for example "Path parameters are declared in the route path [2]." Cite each \
source separately: [1][3], not [1, 3].
- If the sources do not contain the answer, say that the indexed documentation doesn't cover \
it, and mention anything related that the sources do cover.
- Keep answers short and practical. Include a code example when the sources have one that helps.
"""


def build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    """Number the sources from 1 and wrap them in XML tags, followed by the question."""
    sources = "\n".join(
        f'<source id="{number}" title="{chunk.heading_path}" url="{chunk.url}">\n'
        f"{chunk.content}\n"
        "</source>"
        for number, chunk in enumerate(chunks, start=1)
    )
    return f"<sources>\n{sources}\n</sources>\n\n<question>{question}</question>"
