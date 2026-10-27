from fastapi.testclient import TestClient

from docs_qa.api import app, get_generator, get_retriever
from docs_qa.generation import Answer, Usage


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def search(self, question, k):
        self.calls.append((question, k))
        return self.chunks[:k]


class FakeGenerator:
    def generate(self, question, chunks):
        return Answer(
            text="Declare it in the path [1].",
            cited=[1],
            usage=Usage(input_tokens=500, output_tokens=20),
            model="fake-model",
        )


def make_client(retriever) -> TestClient:
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_generator] = FakeGenerator
    # Not used as a context manager, so the lifespan (real DB, embedder, LLM client) never runs.
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_ask_returns_answer_sources_and_usage(chunks):
    retriever = FakeRetriever(chunks)
    response = make_client(retriever).post("/ask", json={"question": "Path params?", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Declare it in the path [1]."
    assert [(s["number"], s["cited"]) for s in body["sources"]] == [(1, True), (2, False)]
    assert body["sources"][0]["url"].startswith("https://fastapi.tiangolo.com/")
    assert body["usage"] == {"input_tokens": 500, "output_tokens": 20}
    assert retriever.calls == [("Path params?", 2)]


def test_ask_uses_default_top_k(chunks):
    retriever = FakeRetriever(chunks)
    make_client(retriever).post("/ask", json={"question": "Path params?"})
    assert retriever.calls[0][1] == 5


def test_ask_validates_input(chunks):
    response = make_client(FakeRetriever(chunks)).post("/ask", json={"question": ""})
    assert response.status_code == 422
