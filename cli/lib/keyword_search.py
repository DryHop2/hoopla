import pickle
from .search_utils import (
    DEFAULT_SEARCH_LIMIT
)
from .inverted_index import InvertedIndex


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Error: No cached index found. Please run 'build' first.")
        return []
    except pickle.UnpicklingError:
        print("Error: Cache files are corrupted or incompatible. Please rebuild with 'build'.")
        return []
    
    query_tokens = idx._tokenize(query)
    if not query_tokens:
        print("No results found.")
        return []

    matched_ids = set()
    for token in query_tokens:
        ids = set(idx.index.get(token, set()))
        if not ids:
            for term, postings in idx.index.items():
                if token in term:
                    ids |= postings

        matched_ids |= ids

    if not matched_ids:
        return []
    
    results = [idx.docmap[i] for i in sorted(matched_ids)[:limit]]

    return results