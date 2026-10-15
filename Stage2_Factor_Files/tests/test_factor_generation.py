from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.generation.generate_candidates import (
    CandidateValidationError,
    generate_and_record,
    split_prompt,
    validate_candidates,
)


def valid_payload() -> dict:
    specifications = [
        ("REL-01", "relative_aapl_qqq", "relative", "subtract(pct_change(AAPL_Close, 5), pct_change(QQQ_Close, 5))", ["AAPL_Close", "QQQ_Close"], 5),
        ("REL-02", "relative_aapl_msft", "relative", "subtract(pct_change(AAPL_Close, 10), pct_change(MSFT_Close, 10))", ["AAPL_Close", "MSFT_Close"], 10),
        ("CON-01", "peer_return_mean", "confirmation", "safe_divide(add(pct_change(MSFT_Close, 5), pct_change(NVDA_Close, 5)), 2)", ["MSFT_Close", "NVDA_Close"], 5),
        ("DIS-01", "peer_return_gap", "dispersion", "abs(subtract(pct_change(MSFT_Close, 3), pct_change(NVDA_Close, 3)))", ["MSFT_Close", "NVDA_Close"], 3),
        ("RSK-01", "relative_volatility", "risk", "safe_divide(rolling_std(pct_change(AAPL_Close, 2), 10), rolling_std(pct_change(QQQ_Close, 2), 10))", ["AAPL_Close", "QQQ_Close"], 10),
        ("COM-01", "volume_momentum", "composite", "multiply(pct_change(AAPL_Close, 5), pct_change(AAPL_Volume, 20))", ["AAPL_Close", "AAPL_Volume"], 20),
        ("REG-01", "market_regime_signal", "regime", "multiply(pct_change(AAPL_Close, 5), pct_change(QQQ_Close, 20))", ["AAPL_Close", "QQQ_Close"], 20),
        ("COM-02", "close_position_mean", "composite", "rolling_mean(safe_divide(subtract(AAPL_Close, AAPL_Low), subtract(AAPL_High, AAPL_Low)), 5)", ["AAPL_Close", "AAPL_Low", "AAPL_High"], 5),
    ]
    candidates = []
    for factor_id, name, family, formula, inputs, lookback in specifications:
        candidates.append(
            {
                "factor_id": factor_id,
                "name": name,
                "family": family,
                "hypothesis": "This factor may provide a useful interpretable signal.",
                "formula": formula,
                "inputs": inputs,
                "lookback_days": lookback,
                "expected_direction": "unknown",
                "leakage_self_check": "Every operation ends at the current trading date.",
                "implementation_note": "Use full rolling windows.",
                "difference_from_baseline": "The interaction may complement individual baseline features.",
            }
        )
    return {"prompt_version": "v1", "candidates": candidates}


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return self.response


class FactorGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema_path = PACKAGE_ROOT / "configs" / "llm_factor_schema_v1.json"
        cls.prompt_path = PACKAGE_ROOT / "configs" / "llm_factor_prompt_v1.txt"
        cls.schema = json.loads(cls.schema_path.read_text(encoding="utf-8"))

    def test_prompt_is_split_into_two_messages(self):
        system, user = split_prompt(self.prompt_path.read_text(encoding="utf-8"))
        self.assertTrue(system.startswith("You are a cautious"))
        self.assertTrue(user.startswith("PROJECT"))

    def test_valid_candidate_batch(self):
        validate_candidates(valid_payload(), self.schema)

    def test_duplicate_id_is_rejected(self):
        payload = valid_payload()
        payload["candidates"][1]["factor_id"] = payload["candidates"][0]["factor_id"]
        with self.assertRaisesRegex(CandidateValidationError, "duplicate value"):
            validate_candidates(payload, self.schema)

    def test_unsupported_formula_operator_is_rejected(self):
        payload = valid_payload()
        payload["candidates"][0]["formula"] = "shift(AAPL_Close, -1)"
        with self.assertRaisesRegex(CandidateValidationError, "unsupported operator"):
            validate_candidates(payload, self.schema)

    def test_input_and_lookback_must_match_formula(self):
        payload = valid_payload()
        payload["candidates"][0]["inputs"] = ["AAPL_Close"]
        payload["candidates"][0]["lookback_days"] = 20
        with self.assertRaises(CandidateValidationError) as caught:
            validate_candidates(payload, self.schema)
        message = str(caught.exception)
        self.assertIn("must exactly match formula inputs", message)
        self.assertIn("expected 5 from the formula", message)

    def test_generation_run_saves_validated_output_and_metadata(self):
        payload = valid_payload()
        response = SimpleNamespace(
            status="completed",
            output_text=json.dumps(payload),
            id="response-test",
            usage={"input_tokens": 100, "output_tokens": 200},
        )
        fake_responses = FakeResponses(response)
        fake_client = SimpleNamespace(responses=fake_responses)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = generate_and_record(
                model="test-model",
                prompt_path=self.prompt_path,
                schema_path=self.schema_path,
                output_directory=Path(directory),
                client=fake_client,
                timestamp=datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc),
            )
            self.assertTrue((run_dir / "raw_output.json").is_file())
            self.assertTrue((run_dir / "validated_candidates.json").is_file())
            metadata = json.loads((run_dir / "generation_metadata.json").read_text())
            self.assertEqual(metadata["status"], "accepted")
            self.assertEqual(metadata["model"], "test-model")
            self.assertEqual(metadata["prompt_version"], "v1")
            self.assertEqual(len(metadata["prompt_sha256"]), 64)
            self.assertEqual(fake_responses.request["text"]["format"]["strict"], True)

    def test_rejected_run_keeps_raw_output_and_error_record(self):
        payload = valid_payload()
        payload["candidates"][0]["lookback_days"] = 20
        response = SimpleNamespace(
            status="completed",
            output_text=json.dumps(payload),
            id="response-test",
            usage=None,
        )
        fake_client = SimpleNamespace(responses=FakeResponses(response))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(CandidateValidationError):
                generate_and_record(
                    model="test-model",
                    prompt_path=self.prompt_path,
                    schema_path=self.schema_path,
                    output_directory=Path(directory),
                    client=fake_client,
                    timestamp=datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc),
                )
            run_dir = next(Path(directory).iterdir())
            metadata = json.loads((run_dir / "generation_metadata.json").read_text())
            self.assertEqual(metadata["status"], "rejected")
            self.assertTrue(metadata["validation_errors"])
            self.assertFalse((run_dir / "validated_candidates.json").exists())


if __name__ == "__main__":
    unittest.main()
