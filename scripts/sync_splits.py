#!/usr/bin/env python3
"""Install the fixed five-fold split into preprocessed datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

EXPECTED_SHA256 = "f4cbf6804e15aacb28d8ac274acc127b99aa71fd3b308da1ce80e370703c8863"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=Path("configs/splits/splits_final.json"))
    parser.add_argument("--preprocessed", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["Dataset003_MyCT", "Dataset004_CTPET"])
    args = parser.parse_args()
    payload = args.split.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"Unexpected split hash: {digest}")
    folds = json.loads(payload)
    if len(folds) != 5:
        raise SystemExit(f"Expected 5 folds, found {len(folds)}")
    for dataset in args.datasets:
        target_dir = args.preprocessed / dataset
        if not target_dir.is_dir():
            raise SystemExit(f"Preprocessed dataset not found: {target_dir}")
        shutil.copy2(args.split, target_dir / "splits_final.json")
        print(f"Installed split: {target_dir / 'splits_final.json'}")


if __name__ == "__main__":
    main()

