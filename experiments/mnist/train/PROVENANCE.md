# Provenance -- train

## Specification
Paper Tables 11 and 12 and Appendix C.

## Code
Source: `orineo1/conditional-matching-paper`, branch **`main`**, MNIST tree at
**`7c856e2`**:
`MNIST/train/train_uncond.py`, `MNIST/train/train_conditional.py`,
`MNIST/src/dataset.py`, `MNIST/src/cond_model.py`, `MNIST/src/uncond_model.py`.
The branch is discussed in `../table_mlgdf/PROVENANCE.md`.

`UnconditionalUnet` and the rotated-MNIST dataset come from `cdm/models/mnist/`, so
the Table 11 UNet configuration is defined once.

## Committed artifacts
- `experiments/mnist/checkpoints/MnistConditional500Epoch.pt` is committed in this repository.
- `MnistUncond100Epoch.pth` is in neither repository; it is fetched from
  HuggingFace (`anon-submission-cdm/cdm-inverse-design`, subfolder `MNIST`).
- A `robust_classifier.pth` is committed in the source repository on branch
  `gread-search-mnist` @ `6b5af8b` (see `../table_mlgdf/PROVENANCE.md`).
- No training log, loss curve or construction-classifier checkpoint is committed
  in either repository.

## Outputs
`outputs/` is gitignored and holds only what a local run writes. The reduced
configuration produces throwaway weights, not the published models.
