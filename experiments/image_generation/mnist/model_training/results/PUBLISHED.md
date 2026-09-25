# Published reference -- MNIST model training

Declared reference: the configuration and timings **as published**. Nothing was
retrained here.

## Unconditional DDPM prior (paper Table 11 architecture, Table 12 training)

Architecture (Table 11, the one float in the paper with no `\label`):
1 channel, 28x28, 2 layers per block, block widths (32, 64, 128), attention
blocks, dropout 0.1.

Training (Table 12): AdamW, lr 1e-3, batch 256, 100 epochs, `squaredcos_cap_v2`
beta schedule, 1000 diffusion steps, MSE loss. Reported time ~178 min on a T4.

Training set: all 70,000 MNIST images (train + test), each independently
rotated, no classifier filtering.

## Conditional iCT angle model (paper Table 12)

AdamW, lr 1e-4, batch 256, 500 epochs, iCT curriculum s0 = 10, s1 = 1280,
eps = 0.002, smooth Huber loss. Reported time ~162 min on a T4.

Training set: 82,483 (image, angle) pairs built from the 60,000 MNIST training
images plus their 90/180/270-degree copies, retained only where the
dataset-construction classifier agrees at confidence >= 0.9999.

## Classifiers

- Dataset-construction classifier: 3-block CNN, 15 epochs, batch 128, Adam 1e-3;
  confidence threshold 0.9999.
- Evaluation (rotation-robust) classifier: AdamW 1e-3, weight decay 1e-4,
  10 epochs, augmentation +/-20 degree rotation and Gaussian noise sigma 0.15.

Source: paper Appendix C and Tables 11-12.
