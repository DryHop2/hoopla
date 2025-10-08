import string
from nltk.stem import PorterStemmer
from functools import lru_cache
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords
)


TRANSLATOR = str.maketrans("", "", string.punctuation)
STEMMER = PorterStemmer()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return results
    
    for movie in movies:
        title_tokens = _tokenize(movie["title"])
        if _has_substring_match(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results


def _preprocess_text(text: str) -> str:
    text = text.casefold()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def _tokenize(s: str) -> list[str]:
    sw = _stopwords_set()
    return [STEMMER.stem(t) for t in _preprocess_text(s).split() if t and STEMMER.stem(t) not in sw]


def _has_substring_match(q_tokens: list[str], t_tokens: list[str]) -> bool:
    for qt in q_tokens:
        for tt in t_tokens:
            if qt in tt:
                return True
    return False


@lru_cache(maxsize=1)
def _stopwords_set() -> set[str]:
    words = load_stopwords()
    return {STEMMER.stem(_preprocess_text(w)) for w in words if w}