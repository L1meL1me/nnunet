#!/usr/bin/env python3
"""Convert paired CT/PET/SEG folders into nnU-Net v2 raw datasets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct-dir", type=Path, required=True)
    parser.add_argument("--pet-dir", type=Path)
    parser.add_argument("--seg-dir", type=Path, required=True)
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--dataset", choices=("Dataset003_MyCT", "Dataset004_CTPET"), required=True)
    parser.add_argument("--case-prefix", default="PETCT_")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing target dataset.")
    return parser.parse_args()


def find_nii(folder: Path, names: tuple[str, ...]) -> Path | None:
    if not folder.is_dir():
        return None
    candidates = sorted(folder.rglob("*.nii.gz"))
    for path in candidates:
        lower = path.name.lower()
        if any(name in lower for name in names):
            return path
    return None


def same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-5)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-4)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-5)
    )


def main() -> None:
    args = parse_args()
    use_pet = args.dataset == "Dataset004_CTPET"
    if use_pet and args.pet_dir is None:
        raise SystemExit("Dataset004_CTPET requires --pet-dir")

    target = args.nnunet_raw / args.dataset
    if target.exists():
        if not args.overwrite:
            raise SystemExit(f"Target exists: {target}. Re-run with --overwrite only after checking it.")
        shutil.rmtree(target)
    images = target / "imagesTr"
    labels = target / "labelsTr"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    cases = sorted(p for p in args.ct_dir.iterdir() if p.is_dir() and p.name.startswith(args.case_prefix))
    failures: list[str] = []
    success = 0
    for case in cases:
        case_id = case.name
        ct_path = find_nii(case, ("body_ct", "ct"))
        seg_path = find_nii(args.seg_dir / case_id, ("body_seg", "seg", "mask", "label"))
        pet_path = find_nii(args.pet_dir / case_id, ("body_suv", "suv", "pet")) if use_pet else None
        if ct_path is None or seg_path is None or (use_pet and pet_path is None):
            failures.append(case_id)
            continue

        ct_image = sitk.ReadImage(str(ct_path))
        seg_image = sitk.ReadImage(str(seg_path))
        if not same_geometry(ct_image, seg_image):
            failures.append(f"{case_id} (CT/SEG geometry mismatch)")
            continue
        if pet_path is not None:
            pet_image = sitk.ReadImage(str(pet_path))
            if not same_geometry(ct_image, pet_image):
                failures.append(f"{case_id} (CT/PET geometry mismatch)")
                continue

        shutil.copy2(ct_path, images / f"{case_id}_0000.nii.gz")
        if pet_path is not None:
            shutil.copy2(pet_path, images / f"{case_id}_0001.nii.gz")
        mask = (sitk.GetArrayFromImage(seg_image) > 0).astype(np.uint8)
        fixed = sitk.GetImageFromArray(mask)
        fixed.CopyInformation(seg_image)
        sitk.WriteImage(fixed, str(labels / f"{case_id}.nii.gz"))
        success += 1

    dataset_json = {
        "channel_names": {"0": "CT", **({"1": "PET"} if use_pet else {})},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": success,
        "file_ending": ".nii.gz",
    }
    (target / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8")
    print(f"Converted {success}/{len(cases)} cases into {target}")
    if failures:
        print("Failed cases:")
        print("\n".join(f"  - {case}" for case in failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

