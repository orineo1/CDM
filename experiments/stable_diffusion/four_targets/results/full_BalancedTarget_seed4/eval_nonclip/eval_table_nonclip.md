# eval_nonclip: non-CLIP scores of the same portraits (N = 2000)

| Method | CLIP MMD | dinov2_base MMD ↓ | Δ (%) | dinov2_base SWD ↓ | Δ (%) | CLIP p(male) | fairface p(male) | 95% CI | agreement |
|---|---|---|---|---|---|---|---|---|---|
| source | 0.506 | 0.485 | +0.0% | 0.0108 | +0.0% | 0.882 | 0.913 | [0.901, 0.925] | 95.8% |
| avg_prompt | 0.453 | 0.378 | +22.0% | 0.0097 | +10.1% | 0.411 | 0.540 | [0.518, 0.562] | 87.0% |
| sdedit_best | 0.439 | 0.374 | +22.9% | 0.0109 | -0.3% | 0.652 | 0.672 | [0.652, 0.693] | 92.0% |
| mlgd_f | 0.348 | 0.338 | +30.3% | 0.0096 | +11.4% | 0.495 | 0.513 | [0.491, 0.535] | 93.2% |

fairface p(male) of the target groups: Man 1.000, Woman 0.000
