#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 ROOT DATASET CONFIG EPOCHS FOLD GPU"
  echo "Example: $0 /mnt/synology/ruihao.li 4 3d_fullres 1000 0 1"
}

if [[ $# -ne 6 ]]; then usage; exit 2; fi

root=$1
dataset=$2
configuration=$3
epochs=$4
fold=$5
gpu=$6

export nnUNet_raw="${root}/nnUNet_raw"
export nnUNet_preprocessed="${root}/nnUNet_preprocessed"
export nnUNet_results="${root}/nnUNet_results"
export TMPDIR="${root}/tmp"
export CUDA_VISIBLE_DEVICES="${gpu}"
mkdir -p "${TMPDIR}"

case "${epochs}" in
  500) trainer=nnUNetTrainer_500epochs ;;
  1000) trainer=nnUNetTrainer ;;
  *) echo "Supported experiment lengths are 500 or 1000 epochs." >&2; exit 2 ;;
esac

nnUNetv2_train "${dataset}" "${configuration}" "${fold}" -tr "${trainer}" --npz

