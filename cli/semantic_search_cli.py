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
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()