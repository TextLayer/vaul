from vaul import Toolkit, tool_call
from vaul.rerankers import BaseReranker, RerankResult
from tests.utils.assertion import is_equal


class MockReranker(BaseReranker):
    def rerank(self, query, documents, top_n=3):
        ranked = []
        for i, doc in enumerate(documents):
            score = 1.0 if "multiply" in doc.lower() else 0.4
            ranked.append((score, i))
        ranked.sort(key=lambda x: x[0], reverse=True)
        ranked = ranked[:top_n]
        return [RerankResult(index=i, relevance_score=s) for s, i in ranked]


@tool_call
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool_call
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def test_tool_schemas_with_reranker():
    toolkit = Toolkit(reranker=MockReranker())
    toolkit.add_tools(add_numbers, multiply_numbers)

    filtered = toolkit.tool_schemas(
        messages=[{"role": "user", "content": "How to multiply?"}],
        top_n=2,
        score_threshold=0.5,
    )
    is_equal(len(filtered), 1)
    is_equal(filtered[0]["function"]["name"], "multiply_numbers")
