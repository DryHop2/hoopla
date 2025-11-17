from nltk.stem import PorterStemmer
from typing import Dict, Set, List
from pathlib import Path
from collections import Counter
import string
import pickle
import math

from .search_utils import (
    load_movies,
    load_stopwords,
    CACHE_DIR
)


class InvertedIndex:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, dict] = {}
        self.term_frequencies: Dict[int, Counter[str]] = {}

        self._stemmer = PorterStemmer()
        self._translator = str.maketrans("", "", string.punctuation)
        self._stopwords = {self._stemmer.stem(self._normalize(w)) for w in load_stopwords() if w}
        
    
    def __add_document(self, doc_id: int, text: str) -> None:
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        for token in self._tokenize(text):
            self.index.setdefault(token, set()).add(doc_id)
            self.term_frequencies[doc_id][token] += 1


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

        with (self.cache_dir / "term_frequencies.pkl").open("wb") as f:
            pickle.dump(self.term_frequencies, f, protocol=pickle.HIGHEST_PROTOCOL)


    def load(self) -> None:
        with (self.cache_dir / "index.pkl").open("rb") as f:
            self.index = pickle.load(f)

        with (self.cache_dir / "docmap.pkl").open("rb") as f:
            self.docmap = pickle.load(f)

        tf_path = self.cache_dir / "term_frequencies.pkl"
        if tf_path.exists():
            with tf_path.open("rb") as f:
                self.term_frequencies = pickle.load(f)
        else:
            self.term_frequencies = {}


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
    

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = self._tokenize(term)
        if len(tokens) != 1:
            raise ValueError("Term must tokenize to exactly one token")
        token = tokens[0]

        return self.term_frequencies.get(doc_id, {}).get(token, 0)
    

    def get_idf(self, term: str) -> float:
        doc_count = len(self.docmap)
        term_doc_count = len(self.get_documents(term))
        return math.log((doc_count + 1) / (term_doc_count + 1))
    

    def get_tfidf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    

    def get_bm25_idf(self, term: str) -> float:
        tokens = self._tokenize(term)
        if len(tokens) != 1:
            raise ValueError("Term must tokenize to exactly one token")
        token = tokens[0]
        doc_count = len(self.docmap)
        df = len(self.get_documents(token))
        bm25 = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)
        return bm25