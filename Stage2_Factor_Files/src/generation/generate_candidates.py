"""Generate, validate and record structured candidate factors."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.metadata
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TICKERS = {"AAPL", "QQQ", "MSFT", "NVDA"}
FIELDS = {"Open", "High", "Low", "Close", "Volume"}
ALLOWED_INPUTS = {f"{ticker}_{field}" for ticker in TICKERS for field in FIELDS}
LOOKBACKS = {2, 3, 5, 10, 20}
FUNCTION_ARITY = {
    "lag": 2,
    "pct_change": 2,
    "rolling_mean": 2,
    "rolling_std": 2,
    "rolling_min": 2,
    "rolling_max": 2,
    "add": 2,
    "subtract": 2,
    "multiply": 2,
    "safe_divide": 2,
    "abs": 1,
}
WINDOW_FUNCTIONS = {
    "lag",
    "pct_change",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
}
FORBIDDEN_FORMULA_WORDS = {
    "target",
    "lead",
    "future",
    "next",
    "shift",
    "center",
    "centered",
    "test",
}


class CandidateValidationError(ValueError):
    """Raised when a generated candidate batch fails local checks."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_prompt(prompt: str) -> tuple[str, str]:
    """Split the prompt file into system and project/task messages."""
    text = prompt.replace("\r\n", "\n").strip()
    lines = text.splitlines()
    if not lines or lines[0].strip() != "SYSTEM":
        raise ValueError("Prompt must begin with a SYSTEM heading")
    try:
        project_index = next(
            index for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "PROJECT"
        )
    except StopIteration as exc:
        raise ValueError("Prompt must contain a PROJECT heading") from exc
    system_message = "\n".join(lines[1:project_index]).strip()
    user_message = "\n".join(lines[project_index:]).strip()
    if not system_message or not user_message:
        raise ValueError("Prompt system and project/task sections must not be empty")
    return system_message, user_message


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def _schema_errors(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Validate the JSON Schema keywords used by this project."""
    errors: list[str] = []
    expected = schema.get("type")
    if expected and not _matches_type(value, expected):
        return [f"{path}: expected {expected}"]
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value is not in the allowed list")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                errors.append(f"{path}: missing required field {required!r}")
        if schema.get("additionalProperties") is False:
            for extra in sorted(set(value).difference(properties)):
                errors.append(f"{path}: unexpected field {extra!r}")
        for key, item in value.items():
            if key in properties:
                errors.extend(_schema_errors(item, properties[key], f"{path}.{key}"))

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: too many items")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                errors.extend(_schema_errors(item, item_schema, f"{path}[{index}]"))

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: text is too short")
        pattern = schema.get("pattern")
        if pattern and re.search(pattern, value) is None:
            errors.append(f"{path}: text does not match the required pattern")
    return errors


def validate_with_schema(payload: Any, schema: dict[str, Any]) -> None:
    errors = _schema_errors(payload, schema)
    if errors:
        raise CandidateValidationError(errors)


def _formula_details(formula: str, path: str) -> tuple[set[str], int, list[str]]:
    errors: list[str] = []
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        return set(), 0, [f"{path}: formula is not valid function syntax ({exc.msg})"]
    if not isinstance(tree.body, ast.Call):
        errors.append(f"{path}: formula must be a function call")

    referenced_inputs: set[str] = set()
    lookbacks: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTION_ARITY:
                name = node.func.id if isinstance(node.func, ast.Name) else "unknown"
                errors.append(f"{path}: unsupported operator {name!r}")
                continue
            name = node.func.id
            if node.keywords or len(node.args) != FUNCTION_ARITY[name]:
                errors.append(
                    f"{path}: {name} requires exactly {FUNCTION_ARITY[name]} positional arguments"
                )
            if name in WINDOW_FUNCTIONS and len(node.args) >= 2:
                window = node.args[1]
                if not isinstance(window, ast.Constant) or type(window.value) is not int:
                    errors.append(f"{path}: {name} lookback must be an integer")
                elif window.value not in LOOKBACKS:
                    errors.append(f"{path}: {name} lookback {window.value} is not allowed")
                else:
                    lookbacks.append(window.value)
        elif isinstance(node, ast.Name):
            if node.id in ALLOWED_INPUTS:
                referenced_inputs.add(node.id)
            elif node.id not in FUNCTION_ARITY:
                errors.append(f"{path}: unsupported identifier {node.id!r}")
        elif isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                errors.append(f"{path}: only numeric constants are allowed")
        elif not isinstance(
            node,
            (ast.Expression, ast.Load),
        ):
            errors.append(f"{path}: unsupported expression element {type(node).__name__}")

    lowered_words = set(re.findall(r"[a-z]+", formula.lower()))
    forbidden = sorted(lowered_words.intersection(FORBIDDEN_FORMULA_WORDS))
    if forbidden:
        errors.append(f"{path}: forbidden terms used: {', '.join(forbidden)}")
    return referenced_inputs, max(lookbacks, default=0), errors


def validate_candidates(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    """Run format, formula, uniqueness and input checks."""
    validate_with_schema(payload, schema)
    errors: list[str] = []
    candidates = payload["candidates"]
    ids: set[str] = set()
    names: set[str] = set()
    formulas: set[str] = set()

    for index, candidate in enumerate(candidates):
        path = f"$.candidates[{index}]"
        factor_id = candidate["factor_id"]
        name = candidate["name"]
        formula = candidate["formula"]
        normalized_formula = re.sub(r"\s+", "", formula).lower()

        if factor_id in ids:
            errors.append(f"{path}.factor_id: duplicate value {factor_id!r}")
        if name in names:
            errors.append(f"{path}.name: duplicate value {name!r}")
        if normalized_formula in formulas:
            errors.append(f"{path}.formula: duplicate formula")
        ids.add(factor_id)
        names.add(name)
        formulas.add(normalized_formula)

        inputs = candidate["inputs"]
        if len(inputs) != len(set(inputs)):
            errors.append(f"{path}.inputs: duplicate input names")
        unexpected_inputs = sorted(set(inputs).difference(ALLOWED_INPUTS))
        if unexpected_inputs:
            errors.append(f"{path}.inputs: unsupported inputs {unexpected_inputs}")

        referenced, required_lookback, formula_errors = _formula_details(
            formula, f"{path}.formula"
        )
        errors.extend(formula_errors)
        if set(inputs) != referenced:
            errors.append(
                f"{path}.inputs: must exactly match formula inputs {sorted(referenced)}"
            )
        if candidate["lookback_days"] != required_lookback:
            errors.append(
                f"{path}.lookback_days: expected {required_lookback} from the formula"
            )

    if errors:
        raise CandidateValidationError(errors)


def _clean_api_schema(schema: dict[str, Any]) -> dict[str, Any]:
    api_schema = copy.deepcopy(schema)
    api_schema.pop("$schema", None)
    api_schema.pop("title", None)
    return api_schema


def _next_run_directory(base: Path, timestamp: datetime) -> Path:
    stem = timestamp.strftime("run_%Y%m%dT%H%M%SZ")
    candidate = base / stem
    suffix = 2
    while candidate.exists():
        candidate = base / f"{stem}_{suffix}"
        suffix += 1
    return candidate


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def generate_and_record(
    *,
    model: str,
    prompt_path: Path,
    schema_path: Path,
    output_directory: Path,
    max_output_tokens: int = 12000,
    client: Any | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Request one candidate batch, validate it, and save an auditable run."""
    prompt_text = prompt_path.read_text(encoding="utf-8")
    schema_text = schema_path.read_text(encoding="utf-8")
    schema = json.loads(schema_text)
    system_message, user_message = split_prompt(prompt_text)

    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required. Install Stage2_Factor_Files/requirements.txt."
            ) from exc
        client = OpenAI()

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "stage2_factor_candidates",
                "strict": True,
                "schema": _clean_api_schema(schema),
            }
        },
        max_output_tokens=max_output_tokens,
    )

    status = getattr(response, "status", None)
    if status != "completed":
        detail = getattr(getattr(response, "incomplete_details", None), "reason", None)
        raise RuntimeError(f"Generation did not complete (status={status}, reason={detail})")
    raw_output = getattr(response, "output_text", "")
    if not raw_output:
        raise RuntimeError("Generation completed without output text")

    now = timestamp or datetime.now(timezone.utc)
    run_directory = _next_run_directory(output_directory, now)
    run_directory.mkdir(parents=True, exist_ok=False)
    (run_directory / "raw_output.json").write_text(raw_output.rstrip() + "\n", encoding="utf-8")

    errors: list[str] = []
    payload: dict[str, Any] | None = None
    try:
        loaded = json.loads(raw_output)
        if not isinstance(loaded, dict):
            raise CandidateValidationError(["$: expected object"])
        payload = loaded
        validate_candidates(payload, schema)
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid JSON: {exc.msg}")
    except CandidateValidationError as exc:
        errors.extend(exc.errors)

    try:
        sdk_version = importlib.metadata.version("openai")
    except importlib.metadata.PackageNotFoundError:
        sdk_version = "test-client"
    usage = getattr(response, "usage", None)
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif usage is not None and not isinstance(usage, (dict, str, int, float, bool, list)):
        usage = None
    metadata = {
        "generated_at_utc": now.isoformat(),
        "status": "accepted" if not errors else "rejected",
        "model": model,
        "response_id": getattr(response, "id", None),
        "prompt_version": schema.get("properties", {}).get("prompt_version", {}).get("const"),
        "prompt_file": prompt_path.name,
        "schema_file": schema_path.name,
        "prompt_sha256": sha256_text(prompt_text),
        "schema_sha256": sha256_text(schema_text),
        "max_output_tokens": max_output_tokens,
        "sdk_version": sdk_version,
        "usage": usage,
        "validation_errors": errors,
    }
    _write_json(run_directory / "generation_metadata.json", metadata)
    if errors:
        raise CandidateValidationError(errors)
    assert payload is not None
    _write_json(run_directory / "validated_candidates.json", payload)
    return run_directory
