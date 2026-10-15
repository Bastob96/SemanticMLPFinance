"""Usage: python -m src.backtesting --predictions PATH --prices PATH --output DIR."""

import argparse
import json
from pathlib import Path
from .report import backtest_best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--prices', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    selection, daily, summary = backtest_best(args.predictions, args.prices)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'backtest_selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    daily.to_csv(args.output / 'backtest_daily.csv', index=False)
    summary.to_csv(args.output / 'backtest_summary.csv', index=False)
    print(json.dumps(selection, indent=2))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
