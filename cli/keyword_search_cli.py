#!/usr/bin/env python3

import argparse
from lib.keyword_search import (
    search_command
)
from lib.inverted_index import (
    InvertedIndex
)


def load_index_or_die() -> InvertedIndex:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Error: No cached index found. Please run 'build' first.")
        raise
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build and persist inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term")

    idf_parser = subparsers.add_parser("idf", help="Get inverted frequency score for a term")
    idf_parser.add_argument("term", type=str, help="Search term")

    args = parser.parse_args()

    match args.command:
        case "search":
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case "build":
            idx = InvertedIndex()
            idx.build()
            idx.save()
        case "tf":
            try:
                idx = load_index_or_die()
            except FileNotFoundError:
                return
            print(idx.get_tf(args.doc_id, args.term))
        case "idf":
            try:
                idx = load_index_or_die()
            except FileNotFoundError:
                return
            
            print(f"Inverse document frequency of '{args.term}': {idx.get_idf(args.term):.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()