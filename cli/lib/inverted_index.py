from nltk.stem import PorterStemmer
from typing import Dict, Set, List, Tuple
from pathlib import Path
from collections import Counter
import string
import pickle
import math

from .search_utils import (
    load_movies,
    load_stopwords,
    CACHE_DIR,
    BM25_K1,
    BM25_B,
    LIMIT
)


class InvertedIndex:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, dict] = {}
        self.term_frequencies: Dict[int, Counter[str]] = {}
        self.doc_lengths: Dict[int, int] = {}

        self._stemmer = PorterStemmer()
        self._translator = str.maketrans("", "", string.punctuation)
        # self._stopwords = {self._stemmer.stem(self._normalize(w)) for w in load_stopwords() if w}
        self._stopwords = set(load_stopwords())

    
    def _add_document(self, doc_id: int, text: str) -> None:
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        tokens = self._tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)
            self.term_frequencies[doc_id][token] += 1


    def get_documents(self, term: str) -> List[int]:
        doc_ids = self.index.get(term, set())
        return sorted(doc_ids)
    

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = int(m["id"])
            self.docmap[doc_id] = m
            text = f"{m.get('title', '')} {m.get('description', '')}"
            self._add_document(doc_id, text)


    def save(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        with (self.cache_dir / "index.pkl").open("wb") as f:
            pickle.dump(self.index, f, protocol=pickle.HIGHEST_PROTOCOL)

        with (self.cache_dir / "docmap.pkl").open("wb") as f:
            pickle.dump(self.docmap, f, protocol=pickle.HIGHEST_PROTOCOL)

        with (self.cache_dir / "term_frequencies.pkl").open("wb") as f:
            pickle.dump(self.term_frequencies, f, protocol=pickle.HIGHEST_PROTOCOL)

        with (self.cache_dir / "doc_lengths.pkl").open("wb") as f:
            pickle.dump(self.doc_lengths, f, protocol=pickle.HIGHEST_PROTOCOL)


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

        doc_lengths_path = self.cache_dir / "doc_lengths.pkl"
        if doc_lengths_path.exists():
            with doc_lengths_path.open("rb") as f:
                self.doc_lengths = pickle.load(f)
        else:
            self.doc_lengths = {}


    def _normalize(self, s: str) -> str:
        return s.casefold().translate(self._translator)
    

    def _tokenize(self, s: str) -> List[str]:
        text = self._normalize(s)
        tokens = text.split()

        valid_tokens = [t for t in tokens if t]

        filtered = [t for t in valid_tokens if t not in self._stopwords]
        
        out: List[str] = []
        for word in filtered:
            out.append(self._stemmer.stem(word))
            
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
    

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        if tf == 0:
            return 0.0
        
        avgdl = self._get_avg_doc_length()
        if avgdl == 0:
            return 0.0
        doc_len = self.doc_lengths.get(doc_id, 0)

        length_norm = 1 - b + b * (doc_len / avgdl)
        sat_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return sat_tf
    

    def _get_avg_doc_length(self) -> float:
        total_docs = len(self.doc_lengths)
        if total_docs == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / total_docs
    

    def bm25(self, doc_id: int, term: str) -> float:
        bm25 = self.get_bm25_idf(term) * self.get_bm25_tf(doc_id, term)
        return bm25
    

    def bm25_search(self, query: str, limit: int = LIMIT) -> List[Tuple[int, int]]:
        tokens = self._tokenize(query)
        if not tokens:
            return {}
        
        scores: Dict[int, float] = {}

        candidate_docs: set[int] = set()
        for token in tokens:
            candidate_docs.update(self.get_documents(token))

        for doc_id in candidate_docs:
            total = 0.0
            for token in tokens:
                total += self.bm25(doc_id, token)
            scores[doc_id] = total

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]