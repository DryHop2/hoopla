from sentence_transformers import SentenceTransformer
from typing import Any
from pathlib import Path
from lib.search_utils import (
    CACHE_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    SCORE_PRECISION,
    load_movies
)
import numpy as np
import re
import json


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


def embed_query_text(query: str) -> None:
    semantic = SemanticSearch()
    embedding = semantic.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def overlapping_chunks(seq: list[str], chunk_size: int, overlap: int) -> list[list[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    
    if overlap == 0:
        return [
            seq[i:i + chunk_size]
            for i in range(0, len(seq), chunk_size)
        ]
    
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("--overlap must be smaller than --chunk-size")
    
    chunks: list[list[str]] = []
    idx = 0
    n = len(seq)

    while idx < n:
        chunk = seq[idx:idx + chunk_size]
        if not chunk:
            break

        if chunks and len(chunk) <= overlap:
            break

        chunks.append(chunk)

        if len(chunk) < chunk_size:
            break

        idx += step

    return chunks


def chunk_text_words(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[list[str]]:
    words = text.split()
    return overlapping_chunks(words, chunk_size, overlap)


def chunk_text_sentences(text: str, max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[list[str]]:
    strip_text = text.strip()
    if strip_text == "":
        return []
    
    raw_sentences = re.split(r"(?<=[.!?])\s+", strip_text)

    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences:
        return []

    if len(sentences) == 1 and not sentences[0].endswith(("?", ".", "!")):
        return [sentences]
    
    return overlapping_chunks(sentences, max_chunk_size, overlap)


def print_chunks(label: str, text: str, chunks: list[list[str]]) -> None:
    print(f"{label} {len(text)} characters")
    for idx, chunk in enumerate(chunks, start=1):
        print(f"{idx}. {' '.join(chunk)}")


def run_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    semantic = SemanticSearch()
    movies = load_movies()
    semantic.load_or_create_embeddings(movies)
    result = semantic.search(query, limit)
    for i, r in enumerate(result, start=1):
        print(f"{i}. {r['title']} (score: {r['score']:.4f})")
        print(f"   {r['description']}\n")


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Path | None = None) -> None:
        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir or CACHE_DIR
        self.embeddings: np.ndarray | None = None
        self.documents: list[dict[str, Any]] | None = None
        self.document_map: dict[int, dict[str, Any]] = {}


    def __repr__(self) -> str:
        return f"SemanticSearch(model={self.model})"
    

    def generate_embedding(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Text must contain non-whitespace characters")
        
        embedding = self.model.encode([text])
        return embedding[0]
    

    def build_embeddings(self, documents: list[dict[str, Any]]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in self.documents}

        texts: list[str] = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        np.save(self.cache_dir / "movie_embeddings.npy", self.embeddings)

        return self.embeddings
    

    def load_or_create_embeddings(self, documents: list[dict[str, Any]]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in self.documents}

        movie_embeddings_path = self.cache_dir / "movie_embeddings.npy"
        if movie_embeddings_path.exists():
            self.embeddings = np.load(movie_embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        
        return self.build_embeddings(documents)
    

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict[str, Any]]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call 'load_or_create_embeddings' first.")
        
        query_embedding = self.generate_embedding(query)
        similarity_scores: list[tuple] = []
        for i, doc_embedding in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, doc_embedding)
            similarity_scores.append((score, self.documents[i]))

        similarity_scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in similarity_scores[:limit]:
            results.append({
                "score": float(score),
                "title": doc["title"],
                "description": doc["description"],
            })

        return results
    

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Path | None = None) -> None:
        super().__init__(model_name=model_name, cache_dir=cache_dir)
        self.chunk_embeddings: np.ndarray | None = None
        self.chunk_metadata: list[dict[str, Any]] | None = None


    def build_chunk_embeddings(self, documents: list[dict[str, Any]]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in self.documents}

        all_chunks: list[str] = []
        chunk_metadata: list[dict[str, Any]] = []

        for movie_idx, doc in enumerate(self.documents):
            desc = doc.get("description", "").strip()
            if not desc:
                continue

            chunks = chunk_text_sentences(desc, max_chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)
            total_chunks = len(chunks)

            for chunk_idx, chunk in enumerate(chunks):
                chunk_text = " ".join(chunk)
                all_chunks.append(chunk_text)

                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks,
                })

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        np.save(self.cache_dir / "chunk_embeddings.npy", self.chunk_embeddings)

        with open(self.cache_dir / "chunk_metadata.json", "w", encoding="utf-8") as f: 
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings
    

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in self.documents}

        chunk_embeddings_path = self.cache_dir / "chunk_embeddings.npy"
        meta_path = self.cache_dir / "chunk_metadata.json"

        if chunk_embeddings_path.exists() and meta_path.exists():
            self.chunk_embeddings = np.load(chunk_embeddings_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.chunk_metadata = meta.get("chunks", [])
            return self.chunk_embeddings
        
        return self.build_chunk_embeddings(documents)
    

    def search_chunks(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict[str, Any]]:
        if self.chunk_embeddings is None:
            raise ValueError("No embeddings loaded. Call 'load_or_create_embeddings' first.")
        
        query_embedding = self.generate_embedding(query)
        chunk_scores: list[dict[str, Any]] = []
        movie_idx_scores: dict[int, float] = {}

        for i, embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, embedding)

            meta = self.chunk_metadata[i]
            movie_idx = meta["movie_idx"]
            chunk_idx = meta["chunk_idx"]

            chunk_scores.append({
                "movie_idx": movie_idx,
                "chunk_idx": chunk_idx,
                "score": score,
            })

        movie_scores: dict[int, dict[str, Any]] = {}

        for cs in chunk_scores:
            movie_idx = cs["movie_idx"]
            score = cs["score"]

            existing = movie_scores.get(movie_idx)
            if existing is None or score > existing["score"]:
                movie_scores[movie_idx] = {
                    "movie_idx": movie_idx,
                    "chunk_idx": cs["chunk_idx"],
                    "score": score,
                }

        best_movies = sorted(movie_scores.values(), key=lambda x: x["score"], reverse=True)[:limit]

        results: list[dict[str, Any]] = []

        for entry in best_movies:
            movie_idx = entry["movie_idx"]
            score = entry["score"]

            doc = self.documents[movie_idx]
            description = doc.get("description", "")
            metadata = doc.get("metadata") or {}

            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": description[:100],
                "score": round(score, SCORE_PRECISION),
                "metadata": metadata,
            })

        return results