# Table 1 appendix: networks and search diagnostics

## x1_y1
- score network: max relative Fisher error over t > 1: 0.001; prior samples SWD (norm) 0.030
- cond_diffusion: W1 at x* 0.012; SWD over random x 0.148
- cond_cm: W1 at x* 0.043; SWD over random x 0.974
- MLGD: 5 attempts x budget 10; trajectories with TV<0.5: 48/50; in x* basin 47/50; per-attempt L2 [0.1537, 0.0577, 0.0, 0.147, 0.0996]
- MLGD-F: 5 attempts x budget 10; trajectories with TV<0.5: 47/50; in x* basin 46/50; per-attempt L2 [0.5169, 0.5022, 0.4055, 0.4853, 0.3383]
- PMLGD: 5 attempts x budget 10; trajectories with TV<0.5: 50/50; in x* basin 50/50; per-attempt L2 [0.0236, 0.0065, 0.0477, 0.0082, 0.0294]

## x1_y5
- score network: max relative Fisher error over t > 1: 0.005; prior samples SWD (norm) 0.027
- cond_diffusion: SWD_norm at x* 0.041; SWD over random x 0.046
- cond_cm: SWD_norm at x* 0.107; SWD over random x 0.108
- MLGD: 5 attempts x budget 10; trajectories with TV<0.5: 49/50; in x* basin 49/50; per-attempt L2 [0.0007, 0.0007, 0.0029, 0.0021, 0.0001]
- MLGD-F: 5 attempts x budget 10; trajectories with TV<0.5: 48/50; in x* basin 48/50; per-attempt L2 [0.0019, 0.0006, 0.0007, 0.0045, 0.0017]
- PMLGD: 5 attempts x budget 10; trajectories with TV<0.5: 50/50; in x* basin 50/50; per-attempt L2 [0.0, 0.0012, 0.0, 0.0053, 0.0037]

## x1_y10
- score network: max relative Fisher error over t > 1: 0.003; prior samples SWD (norm) 0.036
- cond_diffusion: SWD_norm at x* 0.032; SWD over random x 0.039
- cond_cm: SWD_norm at x* 0.092; SWD over random x 0.085
- MLGD: 5 attempts x budget 10; trajectories with TV<0.5: 47/50; in x* basin 49/50; per-attempt L2 [0.0005, 0.0019, 0.0007, 0.0002, 0.0009]
- MLGD-F: 5 attempts x budget 10; trajectories with TV<0.5: 48/50; in x* basin 50/50; per-attempt L2 [0.0028, 0.0031, 0.0029, 0.0031, 0.0025]
- PMLGD: 5 attempts x budget 10; trajectories with TV<0.5: 50/50; in x* basin 50/50; per-attempt L2 [0.0012, 0.0012, 0.0003, 0.0011, 0.0]

## x5_y1
- score network: max relative Fisher error over t > 1: 0.005; prior samples SWD (norm) 0.030
- cond_diffusion: W1 at x* 0.082; SWD over random x 0.189
- cond_cm: W1 at x* 0.119; SWD over random x 0.881
- MLGD: 5 attempts x budget 10; trajectories with TV<0.5: 40/50; in x* basin 19/50; per-attempt L2 [0.0141, 0.0225, 0.0084, 0.0143, 0.0201]
- MLGD-F: 5 attempts x budget 10; trajectories with TV<0.5: 42/50; in x* basin 11/50; per-attempt L2 [0.061, 0.0889, 0.1094, 0.0357, 0.0206]
- PMLGD: 5 attempts x budget 10; trajectories with TV<0.5: 48/50; in x* basin 24/50; per-attempt L2 [0.0205, 0.0201, 0.0041, 0.1073, 0.0481]
- PMLGD-F: 5 attempts x budget 10; trajectories with TV<0.5: 49/50; in x* basin 11/50; per-attempt L2 [0.0167, 0.1217, 0.2474, 0.1558, 0.0412]

## x5_y5
- score network: max relative Fisher error over t > 1: 0.008; prior samples SWD (norm) 0.031
- cond_diffusion: SWD_norm at x* 0.048; SWD over random x 0.097
- cond_cm: SWD_norm at x* 0.289; SWD over random x 0.278
- MLGD: 5 attempts x budget 10; trajectories with TV<0.5: 42/50; in x* basin 43/50; per-attempt L2 [0.0243, 0.0066, 0.0292, 0.0078, 0.0065]
- MLGD-F: 5 attempts x budget 10; trajectories with TV<0.5: 0/50; in x* basin 46/50; per-attempt L2 [0.2533, 0.3485, 0.3907, 0.2577, 0.309]
- PMLGD: 5 attempts x budget 10; trajectories with TV<0.5: 49/50; in x* basin 50/50; per-attempt L2 [0.0347, 0.0077, 0.0041, 0.0205, 0.0301]
- PMLGD-F: 5 attempts x budget 10; trajectories with TV<0.5: 0/50; in x* basin 50/50; per-attempt L2 [0.4097, 0.4232, 0.1888, 0.2106, 0.2969]

## x5_y10
- score network: max relative Fisher error over t > 1: 0.009; prior samples SWD (norm) 0.035
- cond_diffusion: SWD_norm at x* 0.040; SWD over random x 0.079
- cond_cm: SWD_norm at x* 0.092; SWD over random x 0.152
- MLGD: 5 attempts x budget 10; trajectories with TV<0.5: 50/50; in x* basin 50/50; per-attempt L2 [0.0023, 0.0004, 0.0015, 0.0025, 0.0006]
- MLGD-F: 5 attempts x budget 10; trajectories with TV<0.5: 49/50; in x* basin 50/50; per-attempt L2 [0.0098, 0.0104, 0.0096, 0.008, 0.0113]
- PMLGD: 5 attempts x budget 10; trajectories with TV<0.5: 50/50; in x* basin 50/50; per-attempt L2 [0.0027, 0.0007, 0.0045, 0.0012, 0.0005]

## x10_y1
- score network: max relative Fisher error over t > 1: 0.010; prior samples SWD (norm) 0.036
- cond_diffusion: W1 at x* 0.989; SWD over random x 1.285
- cond_cm: W1 at x* 0.576; SWD over random x 1.442
- MLGD: 5 attempts x budget 10; trajectories with TV<0.5: 5/50; in x* basin 0/50; per-attempt L2 [3.9881, 1.2973, 4.0463, 2.7749, 0.8668]
- MLGD-F: 5 attempts x budget 10; trajectories with TV<0.5: 6/50; in x* basin 0/50; per-attempt L2 [1.4143, 0.6815, 1.6304, 1.9064, 2.1258]
- PMLGD: 5 attempts x budget 10; trajectories with TV<0.5: 0/50; in x* basin 0/50; per-attempt L2 [4.051, 4.0512, 4.0409, 8.6551, 4.0508]

## x10_y5
- score network: max relative Fisher error over t > 1: 0.016; prior samples SWD (norm) 0.031
- cond_diffusion: SWD_norm at x* 0.038; SWD over random x 0.158
- cond_cm: SWD_norm at x* 0.112; SWD over random x 0.401
- MLGD: 5 attempts x budget 10; trajectories with TV<0.5: 44/50; in x* basin 0/50; per-attempt L2 [0.0776, 0.0457, 0.1887, 0.1298, 0.0865]
- MLGD-F: 5 attempts x budget 10; trajectories with TV<0.5: 47/50; in x* basin 0/50; per-attempt L2 [0.1024, 0.1657, 0.2628, 0.2077, 0.2669]
- PMLGD: 5 attempts x budget 10; trajectories with TV<0.5: 50/50; in x* basin 0/50; per-attempt L2 [0.009, 0.1468, 0.3439, 0.119, 0.142]

## x10_y10
- score network: max relative Fisher error over t > 1: 0.018; prior samples SWD (norm) 0.030
- cond_diffusion: SWD_norm at x* 0.057; SWD over random x 0.157
- cond_cm: SWD_norm at x* 0.283; SWD over random x 0.405
- MLGD: 5 attempts x budget 10; trajectories with TV<0.5: 26/50; in x* basin 0/50; per-attempt L2 [0.8853, 0.0266, 0.1303, 0.0425, 1.0101]
- MLGD-F: 5 attempts x budget 10; trajectories with TV<0.5: 0/50; in x* basin 0/50; per-attempt L2 [1.8909, 1.8965, 1.8779, 1.9139, 1.6802]
- PMLGD: 5 attempts x budget 10; trajectories with TV<0.5: 25/50; in x* basin 0/50; per-attempt L2 [0.2981, 0.8213, 0.9198, 0.1389, 1.0808]
- PMLGD-F: 5 attempts x budget 10; trajectories with TV<0.5: 0/50; in x* basin 0/50; per-attempt L2 [1.6888, 1.8777, 1.6132, 1.8734, 1.6418]
