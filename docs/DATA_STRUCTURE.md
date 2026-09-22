# Data structure

The repository intentionally excludes medical images, labels, preprocessed arrays, predictions, and checkpoints.

```text
/mnt/synology/ruihao.li/
├── CT/PETCT_<case>/...body_ct.nii.gz
├── PET/PETCT_<case>/...body_suv.nii.gz
├── SEG/PETCT_<case>/...body_seg.nii.gz
├── nnUNet_raw/
│   ├── Dataset003_MyCT/
│   │   ├── dataset.json
│   │   ├── imagesTr/PETCT_<case>_0000.nii.gz
│   │   └── labelsTr/PETCT_<case>.nii.gz
│   └── Dataset004_CTPET/
│       ├── dataset.json
│       ├── imagesTr/PETCT_<case>_0000.nii.gz  # CT
│       ├── imagesTr/PETCT_<case>_0001.nii.gz  # PET SUV
│       └── labelsTr/PETCT_<case>.nii.gz
├── nnUNet_preprocessed/
│   ├── Dataset003_MyCT/splits_final.json
│   └── Dataset004_CTPET/splits_final.json
└── nnUNet_results/
```

Both datasets use the same five-fold split. Fold 0 contains 720 training and 180 validation cases; the validation set contains 96 positive and 84 negative cases.

