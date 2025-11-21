from sentence_transformers import SentenceTransformer
from typing import Dict, Set, List, Tuple


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


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')


    def __repr__(self):
        return f"SemanticSearch(model={self.model})"
    

    def generate_embedding(self, text: str) -> List:
        if text.strip() == "":
            raise ValueError("Text must contain non-whitespace characters")
        
        embedding = self.model.encode([text])
        return embedding[0]