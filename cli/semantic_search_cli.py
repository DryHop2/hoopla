#!/usr/bin/env python3

import argparse
import re
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    ChunkedSemanticSearch
)
from lib.search_utils import (
    load_movies,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE
)
from lib.semantic_search import (
    chunk_text_words,
    chunk_text_sentences,
    print_chunks,
    run_search
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify model information for SemanticSearch")
    subparsers.add_parser("verify_embeddings", help="Verify document embeddings")
    subparsers.add_parser("embed_chunks", help="Embed chunked movie descriptions")

    embed_parser = subparsers.add_parser("embed_text", help="Single string search")
    embed_parser.add_argument("text", type=str, help="Search text")

    embed_query_parser = subparsers.add_parser("embedquery", help="Embed and verify search query")
    embed_query_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("search", help="Search for movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search string")
    search_parser.add_argument("--limit", type=int, nargs="?", default=DEFAULT_SEARCH_LIMIT, help="Tunable top results return [default:5]")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text into fixed sizes")
    chunk_parser.add_argument("text", type=str, help="Text to be chunked")
    chunk_parser.add_argument("--chunk-size", type=int, nargs="?", default=DEFAULT_CHUNK_SIZE, help="Chunk size [default:200]")
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
        case "embed_chunks":
            movies = load_movies()
            semantic = ChunkedSemanticSearch()
            embeddings = semantic.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()