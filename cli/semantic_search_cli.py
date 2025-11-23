#!/usr/bin/env python3

import argparse
import re
from lib.semantic_search import (
    SemanticSearch,
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text
)
from lib.search_utils import (
    LIMIT,
    CHUNK_SIZE,
    load_movies
)


def overlapping_chunks(seq: str, chunk_size: int, overlap: int) -> list[list[str]]:
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


def chunk_text_words(text:str, chunk_size: int, overlap: int) -> list[list[str]]:
    words = text.split()
    return overlapping_chunks(words, chunk_size, overlap)


def chunk_text_sentences(text: str, max_chunk_size: int, overlap: int) -> list[list[str]]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return overlapping_chunks(sentences, max_chunk_size, overlap)


def print_chunks(label: str, text: str, chunks: list[list[str]]) -> None:
    print(f"{label} {len(text)} characters")
    for idx, chunk in enumerate(chunks, start=1):
        print(f"{idx}. {' '.join(chunk)}")


def run_search(query: str, limit: int) -> None:
    semantic = SemanticSearch()
    movies = load_movies()
    semantic.load_or_create_embeddings(movies)
    result = semantic.search(query, limit)
    for i, r in enumerate(result, start=1):
        print(f"{i}. {r['title']} (score: {r['score']:.4f})")
        print(f"   {r['description']}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify model information for SemanticSearch")
    subparsers.add_parser("verify_embeddings", help="Verify document embeddings")

    embed_parser = subparsers.add_parser("embed_text", help="Single string search")
    embed_parser.add_argument("text", type=str, help="Search text")

    embed_query_parser = subparsers.add_parser("embedquery", help="Embed and verify search query")
    embed_query_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("search", help="Search for movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search string")
    search_parser.add_argument("--limit", type=int, nargs="?", default=LIMIT, help="Tunable top results return [default:5]")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text into fixed sizes")
    chunk_parser.add_argument("text", type=str, help="Text to be chunked")
    chunk_parser.add_argument("--chunk-size", type=int, nargs="?", default=CHUNK_SIZE, help="Chunk size [default:200]")
    chunk_parser.add_argument("--overlap", type=int, nargs="?", default=0, help="Number of words to overlap with previous chunk")

    semantic_chunk = subparsers.add_parser("semantic_chunk", help="Chunk text using semantic parser")
    semantic_chunk.add_argument("text", type=str, help="Text to be chunked")
    semantic_chunk.add_argument("--max-chunk-size", type=int, nargs="?", default=4, help="Max sentences per chunk [default=4]")
    semantic_chunk.add_argument("--overlap", type=int, nargs="?", default=0, help="Number of sentences to overlap")
    
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            run_search(args.query, args.limit)
        case "chunk":
            chunks = chunk_text_words(args.text, args.chunk_size, args.overlap)
            print_chunks("Chunking", args.text, chunks)
        case "semantic_chunk":
            chunks = chunk_text_sentences(args.text, args.max_chunk_size, args.overlap)
            print_chunks("Semantically chunking", args.text, chunks)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()