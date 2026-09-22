#!/usr/bin/env python3
"""Batch extract CT and PET radiomics features from an nnU-Net raw dataset."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor


LOGGER = logging.getLogger("ctpet-radiomics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract CT and PET radiomics features from nnUNet_raw imagesTr and "
            "labelsTr, then write modality-specific and merged CSV files."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="nnU-Net raw dataset directory containing dataset.json, imagesTr and labelsTr.",
    )
    parser.add_argument("--ct-params", type=Path, required=True, help="CT YAML parameter file.")
    parser.add_argument("--pet-params", type=Path, required=True, help="PET YAML parameter file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("radiomics_output"),
        help="Output directory (default: ./radiomics_output).",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=1,
        help="Mask label value to extract (default: 1).",
    )
    parser.add_argument(
        "--patients-file",
        type=Path,
        help="Optional TXT/CSV file. The first column must contain case IDs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N selected cases; useful for a trial run.",
    )
    parser.add_argument(
        "--ct-channel",
        type=int,
        help="Override the CT channel number from dataset.json, e.g. 0.",
    )
    parser.add_argument(
        "--pet-channel",
        type=int,
        help="Override the PET channel number from dataset.json, e.g. 1.",
    )
    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="Keep PyRadiomics diagnostics_* columns in the output.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases already present in the modality-specific CSV files.",
    )
    return parser.parse_args()


def load_channel_names(dataset_json: Path) -> dict[int, str]:
    with dataset_json.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    raw_channels = metadata.get("channel_names") or metadata.get("modality")
    if not isinstance(raw_channels, dict):
        raise ValueError(f"No channel_names/modality mapping found in {dataset_json}")

    channels: dict[int, str] = {}
    for key, value in raw_channels.items():
        channels[int(key)] = str(value)
    return channels


def detect_channel(
    channels: dict[int, str], modality: str, override: int | None
) -> int:
    if override is not None:
        if override not in channels:
            raise ValueError(
                f"Channel {override} is not present in dataset.json: {channels}"
            )
        return override

    aliases = {
        "CT": {"CT", "COMPUTEDTOMOGRAPHY"},
        "PET": {"PET", "PT", "SUV", "SUVBW", "POSITRONEMISSIONTOMOGRAPHY"},
    }
    for channel, name in channels.items():
        normalized = "".join(character for character in name.upper() if character.isalnum())
        if normalized in aliases[modality]:
            return channel

    raise ValueError(
        f"Could not identify {modality} in dataset.json channels {channels}. "
        f"Pass --{modality.lower()}-channel explicitly."
    )


def read_requested_cases(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    cases: list[str] = []
    for row_number, row in enumerate(rows):
        if not row or not row[0].strip():
            continue
        value = row[0].strip()
        if row_number == 0 and value.lower() in {
            "case",
            "case_id",
            "caseid",
            "patient",
            "patient_id",
            "patientid",
        }:
            continue
        cases.append(value)
    return cases


def discover_cases(labels_dir: Path, patients_file: Path | None) -> list[str]:
    available = {
        path.name[: -len(".nii.gz")]: path
        for path in labels_dir.glob("*.nii.gz")
    }
    if not available:
        raise FileNotFoundError(f"No .nii.gz masks found in {labels_dir}")

    if patients_file is None:
        return sorted(available)

    requested = read_requested_cases(patients_file)
    missing = [case_id for case_id in requested if case_id not in available]
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} requested cases have no label file. First missing: {preview}"
        )
    return list(dict.fromkeys(requested))


def validate_image_and_mask(image_path: Path, mask_path: Path, label: int) -> None:
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))

    if image.GetDimension() != 3 or mask.GetDimension() != 3:
        raise ValueError(
            f"Only 3D images are supported; image={image.GetDimension()}D, "
            f"mask={mask.GetDimension()}D"
        )

    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(sitk.Cast(mask, sitk.sitkUInt32))
    if not label_stats.HasLabel(label):
        raise ValueError(f"Label {label} is absent from mask {mask_path.name}")
    if label_stats.GetNumberOfPixels(label) < 2:
        raise ValueError(f"Label {label} contains fewer than 2 voxels")


def to_csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return json.dumps(value.tolist(), ensure_ascii=True)
    if isinstance(value, (tuple, list)):
        return json.dumps(list(value), ensure_ascii=True)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return value
    return value


def extract_case(
    extractor: featureextractor.RadiomicsFeatureExtractor,
    image_path: Path,
    mask_path: Path,
    case_id: str,
    label: int,
    include_diagnostics: bool,
) -> OrderedDict[str, Any]:
    validate_image_and_mask(image_path, mask_path, label)
    result = extractor.execute(str(image_path), str(mask_path), label=label)

    row: OrderedDict[str, Any] = OrderedDict()
    row["case_id"] = case_id
    row["image_path"] = str(image_path)
    row["mask_path"] = str(mask_path)
    for key, value in result.items():
        if not include_diagnostics and key.startswith("diagnostics_"):
            continue
        row[key] = to_csv_value(value)
    return row


def load_completed_cases(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    frame = pd.read_csv(csv_path, usecols=["case_id"], dtype=str)
    return set(frame["case_id"].dropna())


def write_rows(csv_path: Path, rows: list[OrderedDict[str, Any]]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")


def write_errors(output_path: Path, errors: list[dict[str, str]]) -> None:
    if not errors:
        if output_path.exists():
            output_path.unlink()
        return
    pd.DataFrame(errors).to_csv(output_path, index=False, encoding="utf-8-sig")


def merge_modalities(ct_path: Path, pet_path: Path, merged_path: Path) -> None:
    if not ct_path.exists() or not pet_path.exists():
        return

    ct = pd.read_csv(ct_path)
    pet = pd.read_csv(pet_path)
    ct = ct.drop(columns=["image_path", "mask_path"], errors="ignore")
    pet = pet.drop(columns=["image_path", "mask_path"], errors="ignore")

    ct = ct.rename(columns={column: f"CT__{column}" for column in ct if column != "case_id"})
    pet = pet.rename(
        columns={column: f"PET__{column}" for column in pet if column != "case_id"}
    )
    merged = ct.merge(pet, on="case_id", how="outer", validate="one_to_one")
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    dataset_dir = args.dataset_dir.resolve()
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    dataset_json = dataset_dir / "dataset.json"
    for required_path in (
        dataset_json,
        images_dir,
        labels_dir,
        args.ct_params,
        args.pet_params,
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Required path does not exist: {required_path}")

    channels = load_channel_names(dataset_json)
    ct_channel = detect_channel(channels, "CT", args.ct_channel)
    pet_channel = detect_channel(channels, "PET", args.pet_channel)
    if ct_channel == pet_channel:
        raise ValueError("CT and PET cannot use the same channel")

    cases = discover_cases(labels_dir, args.patients_file)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        cases = cases[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ct_csv = args.output_dir / "ct_features.csv"
    pet_csv = args.output_dir / "pet_features.csv"
    error_csv = args.output_dir / "failed_cases.csv"
    merged_csv = args.output_dir / "ct_pet_features_merged.csv"

    LOGGER.info("Channels: CT=%d (%s), PET=%d (%s)", ct_channel, channels[ct_channel], pet_channel, channels[pet_channel])
    LOGGER.info("Selected %d cases; ROI label=%d", len(cases), args.label)

    ct_extractor = featureextractor.RadiomicsFeatureExtractor(str(args.ct_params))
    pet_extractor = featureextractor.RadiomicsFeatureExtractor(str(args.pet_params))
    completed_ct = load_completed_cases(ct_csv) if args.resume else set()
    completed_pet = load_completed_cases(pet_csv) if args.resume else set()

    if args.resume:
        existing_ct = pd.read_csv(ct_csv).to_dict("records") if ct_csv.exists() else []
        existing_pet = pd.read_csv(pet_csv).to_dict("records") if pet_csv.exists() else []
        ct_rows: list[OrderedDict[str, Any]] = [OrderedDict(row) for row in existing_ct]
        pet_rows: list[OrderedDict[str, Any]] = [OrderedDict(row) for row in existing_pet]
    else:
        ct_rows = []
        pet_rows = []

    errors: list[dict[str, str]] = []
    for index, case_id in enumerate(cases, start=1):
        mask_path = labels_dir / f"{case_id}.nii.gz"
        LOGGER.info("[%d/%d] %s", index, len(cases), case_id)

        modality_specs = (
            ("CT", ct_channel, ct_extractor, ct_rows, completed_ct),
            ("PET", pet_channel, pet_extractor, pet_rows, completed_pet),
        )
        for modality, channel, extractor, rows, completed in modality_specs:
            if case_id in completed:
                LOGGER.info("Skipping completed %s: %s", modality, case_id)
                continue

            image_path = images_dir / f"{case_id}_{channel:04d}.nii.gz"
            try:
                if not image_path.exists():
                    raise FileNotFoundError(f"Missing image: {image_path}")
                row = extract_case(
                    extractor,
                    image_path,
                    mask_path,
                    case_id,
                    args.label,
                    args.include_diagnostics,
                )
                rows.append(row)
            except Exception as exc:
                LOGGER.exception("%s extraction failed for %s", modality, case_id)
                errors.append(
                    {
                        "case_id": case_id,
                        "modality": modality,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

        # Persist after every case so a long run can be resumed after interruption.
        write_rows(ct_csv, ct_rows)
        write_rows(pet_csv, pet_rows)
        write_errors(error_csv, errors)

    merge_modalities(ct_csv, pet_csv, merged_csv)
    LOGGER.info("CT features: %s", ct_csv)
    LOGGER.info("PET features: %s", pet_csv)
    LOGGER.info("Merged features: %s", merged_csv)
    LOGGER.info("Failures: %d", len(errors))
    return 0 if not errors else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        LOGGER.exception("Fatal error")
        sys.exit(1)
