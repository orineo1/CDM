# Provenance -- model_training

## Specification
Paper Tables 11 and 12 and Appendix C.

## Code
Ported from `orineo1/conditional-matching-paper`, branch **`main`**, MNIST tree
at **`7c856e2`** (repository tip `841417c` at the time of the port):
`MNIST/train/train_uncond.py`, `MNIST/train/train_conditional.py`,
`MNIST/src/dataset.py`, `MNIST/src/cond_model.py`, `MNIST/src/uncond_model.py`.
Branch choice is argued in `../table2_mlgdf/PROVENANCE.md`.

`train_uncond.py` in the source repo carried its own copies of
`UnconditionalUnet` and of the rotated-MNIST dataset; both now come from
`../mnist_cdm/`, so the Table 11 UNet configuration is defined once.

## Committed artifacts
- `MnistModels/MnistConditional500Epoch.pt` is committed in this repository.
- `MnistUncond100Epoch.pth` is in neither repository; it is fetched from
  HuggingFace (`anon-submission-cdm/cdm-inverse-design`, subfolder `MNIST`).
- A `robust_classifier.pth` is committed in the source repository on branch
  `gread-search-mnist` @ `6b5af8b` (see `../table2_mlgdf/PROVENANCE.md`).
- No training log, loss curve or construction-classifier checkpoint is committed
  in either repository.

## What was produced locally
`outputs/` only, by the smoke tier, and it was deleted afterwards. The directory
is empty. The smoke tier produces throwaway weights, not the published models.
