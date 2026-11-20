import json,statistics
from pathlib import Path
P=Path(__file__).resolve().parent
runs=[json.loads((P/f'process_{i}.json').read_text()) for i in range(3)]
summary={}
for arm in ('native','nsi'):
    med=[statistics.median(r['cuda_ms'] for r in run['rows'] if r['arm']==arm) for run in runs]
    summary[arm]={'median_cuda_ms_by_process':med,'range_ms':[min(med),max(med)]}
reductions=[100*(1-b/a) for a,b in zip(summary['native']['median_cuda_ms_by_process'],summary['nsi']['median_cuda_ms_by_process'])]
result={'runs':runs,'summary':summary,'reduction_percent_by_process':reductions,'reduction_percent_range':[min(reductions),max(reductions)],'frame':'GOPR0384_11_00/000001.png','protocol':'FP32 TF32 disabled; batch 1; eval/no_grad; CUDA events; both models resident; 12 alternating paired trials per process'}
(P/'LATENCY.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# EVSSM latency','',result['protocol'],'',f"Frame: {result['frame']}",f"GPU: {runs[0]['gpu']}; driver {runs[0]['driver']}; torch {runs[0]['torch']}; CUDA {runs[0]['cuda']}; Triton {runs[0]['triton']}",'']
for arm in ('native','nsi'):
    a=runs[0]['arms'][arm]; s=summary[arm]
    lines.append(f"- {arm}: parameters {a['parameters']}; SHA256 {a['checkpoint_sha256']}; 256x256 MACs {a['profile_256']['macs']}; per-process CUDA medians (ms) {s['median_cuda_ms_by_process']}; range {s['range_ms']}")
lines += [f'- NSI reduction (%) per process: {reductions}; range {result["reduction_percent_range"]}',f'- Maximum NSI full-image absolute difference: {max(r["max_abs_diff"] for r in runs)}','- Full raw measurements and peak allocated/reserved memory are in LATENCY.json.']
(P/'LATENCY.md').write_text('\n'.join(lines)+'\n')
