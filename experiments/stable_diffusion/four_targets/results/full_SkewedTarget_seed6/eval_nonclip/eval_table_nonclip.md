# eval_nonclip: non-CLIP scores of the same portraits (N = 2000)

| Method | CLIP MMD | dinov2_base MMD ↓ | Δ (%) | dinov2_base SWD ↓ | Δ (%) | CLIP p(male) | fairface p(male) | 95% CI | agreement |
|---|---|---|---|---|---|---|---|---|---|
| source | 0.582 | 0.565 | +0.0% | 0.0147 | +0.0% | 0.896 | 0.966 | [0.958, 0.974] | 92.4% |
| avg_prompt | 0.476 | 0.384 | +32.0% | 0.0095 | +35.4% | 0.207 | 0.428 | [0.406, 0.450] | 78.0% |
| sdedit_best | 0.443 | 0.353 | +37.6% | 0.0097 | +33.7% | 0.430 | 0.664 | [0.643, 0.685] | 75.1% |
| mlgd_f | 0.368 | 0.299 | +47.1% | 0.0082 | +44.0% | 0.246 | 0.353 | [0.332, 0.374] | 87.1% |

fairface p(male) of the target groups: Man 1.000, Woman 0.009
