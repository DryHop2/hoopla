import json
from pathlib import Path
from functools import lru_cache

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
STOPWORDS_PATH = PROJECT_ROOT / "data" / "stopwords.txt"



@lru_cache(maxsize=1)
def load_movies() -> list[dict]:
    with DATA_PATH.open("r") as f:
        data = json.load(f)
    return data["movies"]


@lru_cache(maxsize=1)
def load_stopwords() -> list:
    with STOPWORDS_PATH.open("r") as f:
        stopwords = f.read()
    return stopwords.splitlines()