from dataclasses import dataclass
from typing import List


@dataclass
class RerankResult:
    """Result of a rerank operation."""

    index: int
    relevance_score: float


class BaseReranker:
    """Base interface for reranking tools."""

    def rerank(self, query: str, documents: List[str], top_n: int = 3) -> List[RerankResult]:
        """Rerank documents based on their relevance to the query."""
        raise NotImplementedError
