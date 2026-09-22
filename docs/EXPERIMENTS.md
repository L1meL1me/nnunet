# Experiment record

| Dataset | Config | Epochs | Batch | Patch | Trainer |
|---|---:|---:|---:|---:|---|
| CT | 2d | 500 | 32 | 512×512 | nnUNetTrainer_500epochs |
| CT | 2d | 1000 | 32 | 512×512 | nnUNetTrainer |
| CT | 3d_fullres | 500 | 4 | 160×128×112 | nnUNetTrainer_500epochs |
| CT | 3d_fullres | 1000 | 8 | 160×128×112 | nnUNetTrainer |
| CT+PET | 2d | 500 | 48 | 512×512 | nnUNetTrainer_500epochs |
| CT+PET | 3d_fullres | 500 | 4 | 160×128×112 | nnUNetTrainer_500epochs |
| CT+PET | 3d_fullres | 1000 | 8 | 160×128×112 | nnUNetTrainer |

The `configs/plans/` files are exact snapshots from each result directory, because batch sizes changed between experiments.

Main result: CT+PET 3D 1000 had the best positive-case segmentation (mean Dice 66.59%, median 75.84%), while CT+PET 2D 500 was more balanced on negative cases (44.05% negative-case specificity).

Notion sources:

- [nnunet log](https://app.notion.com/p/97bfddaa4dfd82ddb6130135995a1e5f)
- [PET/CT experiment overview](https://app.notion.com/p/37dfddaa4dfd81e984d9c94ee5f9782f)

