import string
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies
)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []

    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return results
    
    for movie in movies:
        title_tokens = set(_tokenize(movie["title"]))
        if query_tokens & title_tokens:
            results.append(movie)
            if len(results) >= limit:
                break

    return results


def _preprocess_text(text: str) -> str:
    text = text.casefold()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def _tokenize(s: str) -> list[str]:
    return [t for t in _preprocess_text(s).split() if t]