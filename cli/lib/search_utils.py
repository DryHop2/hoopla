import json
from pathlib import Path
from functools import lru_cache

DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_CHUNK_SIZE = 200
DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DEFAULT_CHUNK_OVERLAP = 1
SCORE_PRECISION = 3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
STOPWORDS_PATH = PROJECT_ROOT / "data" / "stopwords.txt"
CACHE_DIR = Path("cache")



@lru_cache(maxsize=1)
def load_movies() -> list[dict]:
    with DATA_PATH.open("r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list:
    with STOPWORDS_PATH.open("r") as f:
        stopwords = f.read()
    return stopwords.splitlines()