# eval_nonclip: non-CLIP scores of the same portraits (N = 2000)

| Method | CLIP MMD | dinov2_base MMD ↓ | Δ (%) | dinov2_base SWD ↓ | Δ (%) | CLIP p(male) | fairface p(male) | 95% CI | agreement |
|---|---|---|---|---|---|---|---|---|---|
| source | 0.535 | 0.476 | +0.0% | 0.0147 | +0.0% | 0.986 | 0.996 | [0.993, 0.999] | 98.8% |
| avg_prompt | 0.437 | 0.372 | +22.0% | 0.0102 | +30.5% | 0.778 | 0.985 | [0.980, 0.990] | 79.2% |
| sdedit_best | 0.455 | 0.362 | +24.0% | 0.0113 | +23.2% | 0.746 | 0.931 | [0.920, 0.943] | 78.6% |
| mlgd_f | 0.349 | 0.288 | +39.5% | 0.0075 | +49.1% | 0.538 | 0.646 | [0.625, 0.667] | 86.2% |

fairface p(male) of the target groups: Woman 0.052, WomanMasc 0.234, ManFem 1.000, Man 1.000
