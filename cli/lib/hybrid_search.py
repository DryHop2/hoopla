from pathlib import Path

from .search_utils import CACHE_DIR
from .inverted_index import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


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
    

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")
    

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")