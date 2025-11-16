from nltk.stem import PorterStemmer
from typing import Dict, Set, List
from pathlib import Path
import string
import pickle

from .search_utils import (
    load_movies,
    load_stopwords,
    CACHE_DIR
)


class InvertedIndex:
    cache_dir = CACHE_DIR

    
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, dict] = {}

        self._stemmer = PorterStemmer()
        self._translator = str.maketrans("", "", string.punctuation)
        self._stopwords = {self._stemmer.stem(self._normalize(w)) for w in load_stopwords() if w}
        
    
    def _add_document(self, doc_id: int, text: str) -> None:
        for token in self._tokenize(text):
            self.index.setdefault(token, set()).add(doc_id)


    def get_documents(self, term: str) -> List[int]:
        norm = self._stemmer.stem(self._normalize(term))
        return sorted(self.index.get(norm, set()))
    

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = int(m["id"])
            self.docmap[doc_id] = m
            text = f"{m.get('title', '')} {m.get('description', '')}"
            self.__add_document(doc_id, text)


    def save(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        with (self.cache_dir / "index.pkl").open("wb") as f:
            pickle.dump(self.index, f, protocol=pickle.HIGHEST_PROTOCOL)

        with (self.cache_dir / "docmap.pkl").open("wb") as f:
            pickle.dump(self.docmap, f, protocol=pickle.HIGHEST_PROTOCOL)


    def load(self) -> None:
        with (self.cache_dir / "index.pkl").open("rb") as f:
            self.index = pickle.load(f)

        with (self.cache_dir / "docmap.pkl").open("rb") as f:
            self.docmap = pickle.load(f)


    def _normalize(self, s: str) -> str:
        return s.casefold().translate(self._translator)
    

    def _tokenize(self, s: str) -> List[str]:
        out: List[str] = []
        for t in self._normalize(s).split():
            if not t:
                continue
            st = self._stemmer.stem(t)
            if st not in self._stopwords:
                out.append(st)
        return out