import argparse

from lib.search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    load_movies
)
from lib.hybrid_search import (
    HybridSearch,
    minmax_normalize
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("normalize", help="Normalize scores to 0 - 1")
    verify_parser.add_argument("scores", nargs="+", type=float, help="Scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Search using exact match and coneptual query")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Split between exact match and conceptual; higher = greater exact match, lower = greater conceptual [DEFAULT: 0.5]")
    weighted_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Search terms [DEFAULT: 5]")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            score_map = {i: s for i, s in enumerate(args.scores)}
            normalized = minmax_normalize(score_map)
            for i in range(len(args.scores)):
                print(f"* {normalized[i]:.4f}")

        case "weighted-search":
            movies = load_movies()
            hybrid = HybridSearch(movies)

            results = hybrid.weighted_search(
                query=args.query,
                alpha=args.alpha,
                limit=args.limit,
            )

            for i, r in enumerate(results, start=1):
                title = r.get("title", "")
                hybrid_score = r.get("hybrid_score", 0.0)
                bm25_score = r.get("bm25_score", 0.0)
                sem_score = r.get("semantic_score", 0.0)

                doc = hybrid.idx.docmap.get(r["id"], {})
                desc = (doc.get("description") or "").strip()

                print(f"{i}. {title}")
                print(f"   Hybrid Score: {hybrid_score:.3f}")
                print(f"   BM25: {bm25_score:.3f}, Semantic: {sem_score:.3f}")
                if desc:
                    preview = desc[:100].rstrip()
                    print(f"   {preview}...")
                else:
                    print("   ...")

        case _:
            parser.print_help()



if __name__ == "__main__":
    main()