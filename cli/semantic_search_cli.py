#!/usr/bin/env python3

import argparse
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


def main():
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
            semantic = SemanticSearch()
            movies = load_movies()
            semantic.load_or_create_embeddings(movies)
            result = semantic.search(args.query, args.limit)
            for i, r in enumerate(result, start=1):
                print(f"{i}. {r['title']} (score: {r['score']:.4f})")
                print(f"   {r['description']}\n")
        case "chunk":
            char_count = len(args.text)
            split_text = args.text.split()
            if args.overlap <= 0:
                chunks = [
                    split_text[i:i + args.chunk_size]
                    for i in range(0, len(split_text), args.chunk_size)
                ]
            else:
                chunks: list[list[str]] = []
                step = args.chunk_size - args.overlap
                if step <= 0:
                    raise ValueError("--overlap must be smaller than --chunk-size")
                
                idx = 0
                n = len(split_text)

                while idx < n:
                    chunk = split_text[idx:idx + args.chunk_size]
                    if not chunk or len(chunk) <= args.overlap:
                        break
                    chunks.append(chunk)
                    if len(chunk) < args.chunk_size:
                        break
                    idx += step

            print(f"Chunking {char_count} characters")
            for idx, chunk in enumerate(chunks, start=1):
                print(f"{idx}. {' '.join(chunk)}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()