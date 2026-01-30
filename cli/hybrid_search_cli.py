import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("normalize", help="Normalize scores to 0 - 1")
    verify_parser.add_argument("scores", nargs="+", type=float, help="Scores to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.scores
            normalized_scores = []
            if not scores:
                return

            min_score, max_score = min(scores), max(scores)

            if min_score == max_score:
                 normalized_scores = [1.0] * len(scores)
            else:
                normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]

            for s in normalized_scores:
                print(f"* {s:.4f}")
        case _:
            parser.print_help()



if __name__ == "__main__":
    main()