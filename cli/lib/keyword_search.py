import string
import pickle
from nltk.stem import PorterStemmer
from .search_utils import (
    DEFAULT_SEARCH_LIMIT
)
from .inverted_index import InvertedIndex


TRANSLATOR = str.maketrans("", "", string.punctuation)
STEMMER = PorterStemmer()


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
        ids = set(idx.get_documents(token))
        if not ids:
            for term, postings in idx.index.items():
                if token in term:
                    ids |= postings

        matched_ids |= ids

    if not matched_ids:
        return []
    
    results = [idx.docmap[i] for i in sorted(matched_ids)[:limit]]

    return results


# def _preprocess_text(text: str) -> str:
#     text = text.casefold()
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     return text


# def _tokenize(s: str) -> list[str]:
#     sw = _stopwords_set()
#     return [STEMMER.stem(t) for t in _preprocess_text(s).split() if t and STEMMER.stem(t) not in sw]


# def _has_substring_match(q_tokens: list[str], t_tokens: list[str]) -> bool:
#     for qt in q_tokens:
#         for tt in t_tokens:
#             if qt in tt:
#                 return True
#     return False


# @lru_cache(maxsize=1)
# def _stopwords_set() -> set[str]:
#     words = load_stopwords()
#     return {STEMMER.stem(_preprocess_text(w)) for w in words if w}