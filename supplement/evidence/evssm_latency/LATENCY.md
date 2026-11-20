# EVSSM latency

FP32 TF32 disabled; batch 1; eval/no_grad; CUDA events; both models resident; 12 alternating paired trials per process

Frame: GOPR0384_11_00/000001.png
GPU: NVIDIA GeForce RTX 5090; driver 617.42; torch 2.10.0; CUDA 13.0; Triton 3.6.0

- native: parameters 17129472; SHA256 d06bd3f57bf0260542cf923cb412a982f5fe4af3d484239fa2614a4166e21c9d; 256x256 MACs 136521551040; per-process CUDA medians (ms) [2152.4635009765625, 2185.014404296875, 2181.0552978515625]; range [2152.4635009765625, 2185.014404296875]
- nsi: parameters 12842304; SHA256 80d4c505a03d0113f3738daaa42f58013db97e31b12d66dd59ad0afb90229060; 256x256 MACs 103223918592; per-process CUDA medians (ms) [2316.1002197265625, 2379.9615478515625, 2277.861572265625]; range [2277.861572265625, 2379.9615478515625]
- NSI reduction (%) per process: [-7.602299350291375, -8.922007249532093, -4.438506190531766]; range [-8.922007249532093, -4.438506190531766]
- Maximum NSI full-image absolute difference: 1.2814998626708984e-06
- Full raw measurements and peak allocated/reserved memory are in LATENCY.json.
