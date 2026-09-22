#!/usr/bin/env python3
"""Map all positive label values to 1 while preserving image geometry."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def fix_one(source: Path, destination: Path) -> tuple[str, list[int]]:
    image = sitk.ReadImage(str(source))
    data = sitk.GetArrayFromImage(image)
    original = [int(value) for value in np.unique(data)]
    fixed = sitk.GetImageFromArray((data > 0).astype(np.uint8))
    fixed.CopyInformation(image)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(fixed, str(destination))
    return source.name, original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Write fixed labels here; use a new directory.")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.labels_dir.resolve() == args.output_dir.resolve():
        raise SystemExit("Refusing in-place overwrite. Use a separate --output-dir, verify it, then swap directories.")
    files = sorted(args.labels_dir.glob("*.nii.gz"))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(fix_one, files, (args.output_dir / p.name for p in files)))
    changed = sum(values != [0, 1] and values != [0] and values != [1] for _, values in results)
    print(f"Processed {len(results)} labels; {changed} had non-binary values.")


if __name__ == "__main__":
    main()

