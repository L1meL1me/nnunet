# nnU-Net PET/CT lesion segmentation

Reproducibility package for the 900-case CT and CT+PET nnU-Net v2 experiments. It contains conversion and label-cleaning scripts, the fixed five-fold split, exact plan snapshots, launch commands, radiomics/metric utilities, and the experiment record. Medical images, predictions, checkpoints, and credentials are intentionally excluded.

## Environment

The recorded server environment was Python 3.10.20, nnU-Net v2.7.0, and PyTorch 2.6.0+cu124.

```bash
conda env create -f environment.yml
conda activate nnunet_env
```

## Prepare data

CT only:

```bash
python scripts/convert_dataset.py \
  --ct-dir /mnt/synology/ruihao.li/CT \
  --seg-dir /mnt/synology/ruihao.li/SEG \
  --nnunet-raw /mnt/synology/ruihao.li/nnUNet_raw \
  --dataset Dataset003_MyCT
```

CT+PET:

```bash
python scripts/convert_dataset.py \
  --ct-dir /mnt/synology/ruihao.li/CT \
  --pet-dir /mnt/synology/ruihao.li/PET \
  --seg-dir /mnt/synology/ruihao.li/SEG \
  --nnunet-raw /mnt/synology/ruihao.li/nnUNet_raw \
  --dataset Dataset004_CTPET
```

Existing target datasets are never replaced unless `--overwrite` is supplied explicitly.

## Plan and preprocess

```bash
export nnUNet_raw=/mnt/synology/ruihao.li/nnUNet_raw
export nnUNet_preprocessed=/mnt/synology/ruihao.li/nnUNet_preprocessed
export nnUNet_results=/mnt/synology/ruihao.li/nnUNet_results
export TMPDIR=/mnt/synology/ruihao.li/tmp

nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 4 --verify_dataset_integrity

python scripts/sync_splits.py \
  --preprocessed /mnt/synology/ruihao.li/nnUNet_preprocessed
```

Install the fixed split after preprocessing and before training.

## Train

The wrapper reproduces the recorded trainer choice: 500 epochs uses `nnUNetTrainer_500epochs`; 1000 epochs uses the nnU-Net v2.7.0 default `nnUNetTrainer`.

```bash
# CT+PET, 3D full resolution, fold 0, 1000 epochs, physical GPU 1
bash scripts/run_nnunet.sh /mnt/synology/ruihao.li 4 3d_fullres 1000 0 1

# CT+PET, 2D, fold 0, 500 epochs, physical GPU 3
bash scripts/run_nnunet.sh /mnt/synology/ruihao.li 4 2d 500 0 3
```

Raw equivalent command:

```bash
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 4 3d_fullres 0 -tr nnUNetTrainer --npz
```

## Repository map

- `scripts/`: safe conversion, label repair, split installation, and training entry points.
- `configs/datasets/`: dataset descriptors for Dataset003 and Dataset004.
- `configs/splits/`: the exact shared five-fold split (SHA-256 recorded in `configs/experiments.yaml`).
- `configs/plans/`: exact plans captured from each experiment result directory.
- `analysis/`: metric, visualization, and radiomics scripts used after training.
- `docs/`: data layout and experiment summary.

See [data structure](docs/DATA_STRUCTURE.md) and [experiment record](docs/EXPERIMENTS.md).

