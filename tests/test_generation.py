from docs_qa.generation import NO_SOURCES_ANSWER, REFUSAL_ANSWER, AnswerGenerator, extract_citations
from docs_qa.llm import Completion
from docs_qa.prompts import SYSTEM_PROMPT


class FakeLLM:
    def __init__(self, completion: Completion):
        self.completion = completion
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> Completion:
        self.calls.append((system, user))
        return self.completion


def completion(text: str, refused: bool = False) -> Completion:
    return Completion(text=text, input_tokens=1200, output_tokens=80, model="fake", refused=refused)


def test_generate_builds_prompt_and_parses_answer(chunks):
    llm = FakeLLM(completion("Use braces in the path [1]. See also [2][1]."))

    answer = AnswerGenerator(llm).generate("How do I declare a path parameter?", chunks)

    system, user = llm.calls[0]
    assert system == SYSTEM_PROMPT
    assert '<source id="1" title="Path Parameters"' in user
    assert '<source id="2" title="Query Parameters"' in user
    assert "<question>How do I declare a path parameter?</question>" in user

    assert answer.text == "Use braces in the path [1]. See also [2][1]."
    assert answer.cited == [1, 2]
    assert (answer.usage.input_tokens, answer.usage.output_tokens) == (1200, 80)
    assert answer.model == "fake"


def test_no_chunks_skips_the_llm():
    llm = FakeLLM(completion("unused"))
    answer = AnswerGenerator(llm).generate("anything", [])
    assert answer.text == NO_SOURCES_ANSWER
    assert llm.calls == []


def test_refusal(chunks):
    answer = AnswerGenerator(FakeLLM(completion("", refused=True))).generate("something", chunks)
    assert answer.text == REFUSAL_ANSWER
    assert answer.cited == []


def test_extract_citations_ignores_out_of_range_numbers():
    assert extract_citations("A [3] B [1] C [9] D [0] E [1]", num_sources=3) == [3, 1]
