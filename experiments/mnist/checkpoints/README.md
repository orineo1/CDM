# checkpoints

Not in git. The code looks here first and otherwise fetches or builds each file:

| file | what | if missing |
|---|---|---|
| `MnistUncond100Epoch.pth` | the unconditional prior `P(X)` | downloaded from HuggingFace `anon-submission-cdm/cdm-inverse-design`, subfolder `MNIST` |
| `MnistConditional500Epoch.pt` | the conditional angle sampler `P(Y\|X)` | downloaded from the same repository |
| `robust_classifier.pth` | the rotation-robust digit classifier | trained once by the code (10 epochs on MNIST, seed 42) and saved here |

Set `HF_TOKEN` if the HuggingFace repository is gated. Downloads are cached in the
HuggingFace cache (`HF_HOME`), not in this folder. To use a different classifier, pass
`CLASSIFIER=<file>` to `../table_mlgdf/run.sh full`.
