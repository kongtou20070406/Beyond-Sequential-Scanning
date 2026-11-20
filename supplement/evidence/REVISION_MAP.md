# Revision evidence map

Paths below are relative to `code/supplement/evidence/`. PSNR / SSIM values are from the named `FINAL.json` records and are rounded to four decimal places.

| Paper | Evidence and key values |
| --- | --- |
| Table 3 | `standard_budgets/gopro_200k/{nsi,ss2d}/FINAL.json`: NSI 31.3444 / 0.9189, SS2D 31.2271 / 0.9181; `standard_budgets/haze4k_300k/nsi/FINAL.json`: NSI 35.5113 / 0.9922. |
| Table 22 | `standard_budgets/haze4k_300k/{nsi,other_arms/rs8,other_arms/native}/FINAL.json`: NSI 35.5113 / 0.9922, RS8 35.0650 / 0.9900, native 33.9541 / 0.9889. |
| Table 23 | `standard_budgets/gopro_200k/{nsi,ss2d}/FINAL.json`: NSI 31.3444 / 0.9189, SS2D 31.2271 / 0.9181; `PROTOCOL.json` and `per_image.jsonl` record the protocol and paired results. |
| Table 25 | `spn_controls/{nlspn_style,cspnpp_style}/FINAL.json`: NLSPN-style 25.7432 / 0.9058, CSPN++-style 26.1285 / 0.9056; see also `CONFIG_USED.yml` and `run.log`. |
| Table 26 | `ohaze_scans/ohaze_{local4,rs8,eamamba,c2ssm}_seed42/FINAL.json`: Local4 26.6698 / 0.9118, RS8 26.3758 / 0.8991, EAMamba 26.9358 / 0.9100, C2SSM 26.8563 / 0.9131; `ohaze_scans/MACS_REPORT.md`. |
| Appendix E | `evssm_latency/LATENCY.json` and `LATENCY.md`: EVSSM checkpoint hashes, parameters and MACs; these latency records do not contain the appendix's PSNR / SSIM evaluation results. |
| Appendix F | `standard_budgets/haze4k_300k/nsi/{FINAL.json,TRAIN_COMPLETE.json,nsi_ENDPOINT_CPU_AUDIT.json}` and `standard_budgets/gopro_200k/{nsi,ss2d}/FINAL.json`: Table 22 and Table 23 endpoints. |
| Appendix G | `evssm_latency/{LATENCY.json,LATENCY.md,process_0.json,process_1.json,process_2.json}` and `evssm_latency/linux_rtx5090/LATENCY.json`: paired EVSSM full-image latency records. |
| Appendix H | `spn_controls/{nlspn_style,cspnpp_style}/{FINAL.json,CONFIG_USED.yml,run.log}`: spatial propagation controls in Table 25. |
| Appendix I | `port_check/{A_PORT_CHECK.json,A_PORT_CHECK.md,port_check.py}`: transfer checks; `ohaze_scans/` contains Table 26 scan records. |

Haze4K/GoPro hashes in the records identify the corresponding checkpoints. Checkpoint weights are hosted separately (see `REPRODUCE.md`).
