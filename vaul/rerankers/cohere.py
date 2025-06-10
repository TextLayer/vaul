from .base import BaseReranker, RerankResult
from typing import List


class CohereReranker(BaseReranker):
    """Cohere based implementation for reranking."""

    def __init__(self, api_key: str, model: str = "rerank-v3.5"):
        try:
            from cohere import Client
        except Exception as e:  # pragma: no cover - cohere may not be installed
            raise ImportError("cohere package is required for CohereReranker") from e

        self.client = Client(api_key)
        self.model = model

    def rerank(self, query: str, documents: List[str], top_n: int = 3) -> List[RerankResult]:
        response = self.client.rerank(model=self.model, query=query, documents=documents, top_n=top_n)
        return [RerankResult(index=r.index, relevance_score=r.relevance_score) for r in response.results]
