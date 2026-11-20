# EVSSM latency

FP32 TF32 disabled; batch 1; eval/no_grad; CUDA events; both models resident; 12 alternating paired trials per process

Frame: GOPR0384_11_00/000001.png
GPU: NVIDIA GeForce RTX 5090; driver 595.84; torch 2.8.0+cu128; CUDA 12.8; Triton 3.4.0

- native: parameters 17129472; SHA256 d06bd3f57bf0260542cf923cb412a982f5fe4af3d484239fa2614a4166e21c9d; 256x256 MACs 136521551040; per-process CUDA medians (ms) [1946.49755859375, 1876.9653930664062, 1921.3067016601562]; range [1876.9653930664062, 1946.49755859375]
- nsi: parameters 12842304; SHA256 80d4c505a03d0113f3738daaa42f58013db97e31b12d66dd59ad0afb90229060; 256x256 MACs 103223918592; per-process CUDA medians (ms) [2054.87890625, 1992.43603515625, 2055.5956420898438]; range [1992.43603515625, 2055.5956420898438]
- NSI reduction (%) per process: [-5.568018679383813, -6.1519856741310885, -6.989458804971194]; range [-6.989458804971194, -5.568018679383813]
- Maximum NSI full-image absolute difference: 1.2814998626708984e-06
- Full raw measurements and peak allocated/reserved memory are in LATENCY.json.
