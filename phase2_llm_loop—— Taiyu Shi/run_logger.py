"""Per-run output folders, JSON/CSV writers and logging setup."""

import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config


def to_jsonable(value):
    """Convert numpy/pandas objects to plain JSON types; NaN and inf become null."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return to_jsonable(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return to_jsonable(value.to_dict())
    return value


class RunLogger:
    def __init__(self, mode, run_id=None):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_id = run_id or f"{stamp}_{mode}"
        self.run_dir = config.RUNS_DIR / self.run_id
        self.figures_dir = config.FIGURES_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logging()

    def _setup_logging(self):
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        for handler in list(root.handlers):
            if getattr(handler, "_capstone", False):
                root.removeHandler(handler)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        file_handler = logging.FileHandler(self.run_dir / "run.log", encoding="utf-8")
        console = logging.StreamHandler(sys.stdout)
        for handler in (file_handler, console):
            handler.setFormatter(formatter)
            handler._capstone = True
            root.addHandler(handler)
        for noisy in ("httpx", "anthropic", "matplotlib"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        return logging.getLogger("capstone")

    def path(self, relative):
        target = self.run_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def save_json(self, relative, value):
        target = self.path(relative)
        target.write_text(json.dumps(to_jsonable(value), indent=2, ensure_ascii=False, allow_nan=False),
                          encoding="utf-8")
        return target

    def save_csv(self, relative, frame, index=False):
        target = self.path(relative)
        frame.to_csv(target, index=index, encoding="utf-8")
        return target

    def save_text(self, relative, text):
        target = self.path(relative)
        target.write_text(text, encoding="utf-8")
        return target

    def close(self):
        root = logging.getLogger()
        for handler in list(root.handlers):
            if getattr(handler, "_capstone", False):
                handler.close()
                root.removeHandler(handler)
