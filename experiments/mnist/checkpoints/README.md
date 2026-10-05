# checkpoints

Not in git. The code looks here first and otherwise downloads or trains each file:

| file | what | if missing |
|---|---|---|
| `MnistUncond100Epoch.pth` | the prior `P(X)` | downloaded from Hugging Face `anon-submission-cdm/cdm-inverse-design`, folder `MNIST` |
| `MnistConditional500Epoch.pt` | the conditional angle sampler `P(Y \| X)` | downloaded from the same repository |
| `robust_classifier.pth` | the rotation-robust digit classifier | trained once (10 epochs, seed 42, a few minutes on a GPU) and saved here |

Downloads go to the Hugging Face cache (`HF_HOME`); set `HF_TOKEN` if the repository is gated. MNIST is downloaded
to `MNIST_DATA_ROOT` (default `../train/outputs/data`).
