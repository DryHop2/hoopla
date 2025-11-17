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


def add_doc_term_args(p):
    p.add_argument("doc_id", type=int, help="Document ID")
    p.add_argument("term", type=str, help="Search term")


def add_term_arg(p):
    p.add_argument("term", type=str, help="Search term")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build and persist inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document")
    add_doc_term_args(tf_parser)

    idf_parser = subparsers.add_parser("idf", help="Get inverse frequency for a term")
    add_term_arg(idf_parser)

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score")
    add_doc_term_args(tfidf_parser)

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for given term")
    add_term_arg(bm25_idf_parser)

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
        case "tfidf":
            try:
                idx = load_index_or_die()
            except FileNotFoundError:
                return
            
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {idx.get_tfidf(args.doc_id, args.term):.2f}")
        case "bm25idf":
            try:
                idx = load_index_or_die()
            except FileNotFoundError:
                return
            
            bm25 = idx.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()