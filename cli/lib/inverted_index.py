from nltk.stem import PorterStemmer
from typing import Dict, Set, List
from pathlib import Path
import string
import pickle

from .search_utils import (
    load_movies,
    CACHE_DIR
)


class InvertedIndex:
    cache_dir = CACHE_DIR

    
    def __init__(self) -> None:
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, dict] = {}

        self._translator = str.maketrans("", "", string.punctuation)
        self._stemmer = PorterStemmer()
    

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = self._tokenize(text)
        for tok in tokens:
            bucket = self.index.get(tok)
            if bucket is None:
                bucket = self.index[tok] = set()
            bucket.add(doc_id)


    def get_documents(self, term: str) -> List[int]:
        norm = self._normalize(term)
        ids = self.index.get(norm, set())
        
        return sorted(ids)
    

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = int(m["id"])
            self.docmap[doc_id] = m
            text = f"{m.get('title', '')} {m.get('description', '')}"
            self.__add_document(doc_id, text)


    def save(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        with (CACHE_DIR / "index.pkl").open("wb") as f:
            pickle.dump(self.index, f, protocol=pickle.HIGHEST_PROTOCOL)

        with (CACHE_DIR / "docmap.pkl").open("wb") as f:
            pickle.dump(self.docmap, f, protocol=pickle.HIGHEST_PROTOCOL)


    def load(self) -> None:
        with (CACHE_DIR / "index.pkl").open("rb") as f:
            self.index = pickle.load(f)

        with (CACHE_DIR / "docmap.pkl").open("rb") as f:
            self.docmap = pickle.load(f)


    def _normalize(self, s: str) -> str:
        return s.casefold().translate(self._translator)
    

    def _tokenize(self, s: str) -> List[str]:
        tokens = [t for t in self._normalize(s).split() if t]
        return [self._stemmer.stem(t) for t in tokens]