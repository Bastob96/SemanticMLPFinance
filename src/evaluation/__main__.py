"""Usage: python -m src.evaluation --predictions PATH --output DIRECTORY."""

import argparse
from pathlib import Path
from .report import comparison_table, evaluate_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    detailed = evaluate_predictions(args.predictions)
    table = comparison_table(args.predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(args.output / 'evaluation_detailed.csv', index=False)
    table.to_csv(args.output / 'evaluation_comparison.csv', index=False)
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
