import json
import tempfile
import unittest
from pathlib import Path

from src.data import verify_canonical_data


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/canonical_data_2025.json"


class CanonicalDataTests(unittest.TestCase):
    def test_shared_quarterly_profile_passes(self):
        report = verify_canonical_data(ROOT, MANIFEST)
        self.assertEqual(report["status"], "shared_canonical")
        self.assertEqual(report["sources"]["five_year"]["rows"], 5012)
        self.assertEqual(
            report["stage2_profile"]["usable_split_counts"],
            {"train": 121, "validation": 63, "test": 63},
        )

    def test_alternate_copy_cannot_silently_replace_canonical(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["sources"]["five_year"]["path"] = (
            "data/processed/aligned_daily_ohlcv_5y__taiyushi.csv"
        )
        with tempfile.TemporaryDirectory() as folder:
            changed = Path(folder) / "manifest.json"
            changed.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_canonical_data(ROOT, changed)


if __name__ == "__main__":
    unittest.main()
