from sentence_transformers import SentenceTransformer
from typing import Dict, Set, List, Tuple, Any
from pathlib import Path
from lib.search_utils import (
    CACHE_DIR,
    load_movies
)
import numpy as np


def verify_model() -> None:
    semantic = SemanticSearch()

    print(f"Model loaded: {semantic.model}")
    print(f"Max sequence length: {semantic.model.max_seq_length}")


def embed_text(text: str) -> None:
    semantic = SemanticSearch()
    embedding = semantic.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    semantic = SemanticSearch()
    movies = load_movies()
    embeddings = semantic.load_or_create_embeddings(movies)

    print(f"Number of docs:  {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


class SemanticSearch:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache_dir = cache_dir or CACHE_DIR
        self.embeddings: np.ndarray | None = None
        self.documents: List[Dict[str, Any]] | None = None
        self.document_map: Dict[int, Dict[str, Any]] = {}


    def __repr__(self) -> str:
        return f"SemanticSearch(model={self.model})"
    

    def generate_embedding(self, text: str) -> np.ndarray:
        if text.strip() == "":
            raise ValueError("Text must contain non-whitespace characters")
        
        embedding = self.model.encode([text])
        return embedding[0]
    

    def build_embeddings(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in self.documents}

        texts: list[str] = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        np.save(self.cache_dir / "movie_embeddings.npy", self.embeddings)

        return self.embeddings
    

    def load_or_create_embeddings(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in self.documents}

        movie_embeddings_path = self.cache_dir / "movie_embeddings.npy"
        if movie_embeddings_path.exists():
            self.embeddings = np.load(movie_embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        
        return self.build_embeddings(documents)