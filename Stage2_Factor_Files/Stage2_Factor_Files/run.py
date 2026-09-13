"""Run from the repository root or from inside this folder."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "src/factors/build_stage2_candidate_factors.py"), run_name="__main__")
