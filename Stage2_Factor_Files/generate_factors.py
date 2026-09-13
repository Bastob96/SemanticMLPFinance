from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.generation.generate_candidates import (
    CandidateValidationError,
    generate_and_record,
    split_prompt,
    validate_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or validate a structured Stage 2 candidate-factor batch"
    )
    parser.add_argument("--model", help="Model name used for a new generation run")
    parser.add_argument(
        "--validate",
        type=Path,
        metavar="JSON_FILE",
        help="Validate an existing candidate JSON file without making a request",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=PACKAGE_ROOT / "configs" / "llm_factor_prompt_v1.txt",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=PACKAGE_ROOT / "configs" / "llm_factor_schema_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "data" / "generated",
    )
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    args = parser.parse_args()

    try:
        schema = json.loads(args.schema.read_text(encoding="utf-8"))
        split_prompt(args.prompt.read_text(encoding="utf-8"))
        if args.validate:
            payload = json.loads(args.validate.read_text(encoding="utf-8"))
            validate_candidates(payload, schema)
            print(f"Validated 8 candidates in {args.validate}")
            return
        if not args.model:
            parser.error("--model is required unless --validate is used")
        if args.max_output_tokens < 1000:
            parser.error("--max-output-tokens must be at least 1000")
        run_directory = generate_and_record(
            model=args.model,
            prompt_path=args.prompt,
            schema_path=args.schema,
            output_directory=args.output_dir,
            max_output_tokens=args.max_output_tokens,
        )
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Saved accepted candidate batch to {run_directory}")


if __name__ == "__main__":
    main()
