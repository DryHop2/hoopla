from pathlib import Path

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_ALPHA
)
from .inverted_index import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


def minmax_normalize(score_map: dict[int, float]) -> dict[int, float]:
    if not score_map:
        return {}
    
    vals = list(score_map.values())
    mn, mx = min(vals), max(vals)
    if mn == mx:
        return {k: 1.0 for k in score_map}

    return {k: (v - mn) / (mx - mn) for k, v in score_map.items()}


def hybrid_score(bm25_norm: float, sem_norm: float, alpha: float) -> float:
    return alpha * bm25_norm + (1 - alpha) * sem_norm


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        index_file = self.idx.cache_dir / "index.pkl"
        if not index_file.exists():
            self.idx.build()
            self.idx.save()

    
    def _bm25_search(self, query, limit):
        self.idx.load()
        
        return self.idx.bm25_search(query, limit)
    

    def weighted_search(self, query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT):
        expanded_limit = limit * 500

        bm25_results = self._bm25_search(query, expanded_limit)
        bm25_map: dict[int, float] = {doc_id: score for doc_id, score in bm25_results}

        semantic_results = self.semantic_search.search_chunks(query, expanded_limit)
        sem_map: dict[int, float] = {r["id"]: float(r["score"]) for r in semantic_results}

        bm25_norm = minmax_normalize(bm25_map)
        sem_norm = minmax_normalize(sem_map)

        candidate_ids = set(bm25_norm) | set(sem_norm)

        combined: list[dict] = []
        for doc_id in candidate_ids:
            b = bm25_norm.get(doc_id, 0.0)
            s = sem_norm.get(doc_id, 0.0)
            h = hybrid_score(b, s, alpha)

            doc = self.idx.docmap.get(doc_id)
            if not doc:
                continue

            combined.append({
                "id": doc_id,
                "title": doc.get("title", ""),
                "hybrid_score": h,
                "bm25_score": b,
                "semantic_score": s,
            })

        combined.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return combined[:limit]
    

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")