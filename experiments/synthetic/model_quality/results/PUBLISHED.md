> The declared reference for this experiment: the numbers as published in the paper,
> transcribed with their location in the source. `../outputs/` is gitignored and holds
> only whatever a local run emitted.

# Table 5 `tab:model_quality` (paper.txt:828-836)

| Setting | score net Fisher (rel.) | prior SWD | cond DDPM at x* | cond DDPM random x | iCT at x* | iCT random x |
|---|---|---|---|---|---|---|
| dim X = 1, dim Y = 1   | 0.001 | 0.020 | 0.012 | 0.166 | 0.041 | 1.234 |
| dim X = 1, dim Y = 5   | 0.005 | 0.036 | 0.041 | 0.043 | 0.166 | 0.161 |
| dim X = 1, dim Y = 10  | 0.003 | 0.036 | 0.032 | 0.041 | 0.105 | 0.097 |
| dim X = 5, dim Y = 1   | 0.006 | 0.026 | 0.082 | 0.193 | 0.112 | 0.882 |
| dim X = 5, dim Y = 5   | 0.007 | 0.030 | 0.048 | 0.097 | 0.213 | 0.258 |
| dim X = 5, dim Y = 10  | 0.010 | 0.033 | 0.040 | 0.088 | 0.093 | 0.164 |
| dim X = 10, dim Y = 1  | 0.011 | 0.036 | 0.990 | 0.766 | 0.563 | 1.673 |
| dim X = 10, dim Y = 5  | 0.014 | 0.029 | 0.038 | 0.181 | 0.165 | 0.406 |
| dim X = 10, dim Y = 10 | 0.017 | 0.032 | 0.057 | 0.142 | 0.279 | 0.401 |

Transcribed from `paper.txt`.

Alongside: `checks__x5_y5.json`, `checks__x5_y10.json`, `checks__x10_y5.json`,
`checks__x10_y10.json`, imported from `orineo1/GlobalConditional` `master` `9ddf570`.
