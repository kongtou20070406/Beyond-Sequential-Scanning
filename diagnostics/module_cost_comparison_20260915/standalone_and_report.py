"""Verify standalone mixers at a common spatial shape and export the report."""
from pathlib import Path
base=Path(__file__).resolve().parent
exec(compile((base/'measure.py').read_text(encoding='utf-8').split('jobs=')[0],str(base/'measure.py'),'exec'))
from clpo2027.vmamba_ss2d_adapter import build_official_ss2d_mixer
probe=load('standalone_ours_probe',ROOT/'diagnostics/token_mixer_method_20260912/profile_module_macs.py')
module=probe.load_module()
report=json.loads((OUT/'RESULTS.json').read_text(encoding='utf-8'))
rows=[]
for c in [32,48,64,96,128,192]:
 o=probe.measure(module,c,64,64,5)
 with contextlib.redirect_stdout(io.StringIO()):s=build_official_ss2d_mixer(c,stage_index=0,block_index=0).cpu().eval()
 scan=legacy.count_only_ss2d(s)
 x=torch.zeros(1,c,64,64)
 with torch.inference_mode():
  a=FlopCountAnalysis(s,x).unsupported_ops_warnings(False).uncalled_modules_warnings(False).tracer_warnings('none')
  a.set_op_handle('aten::einsum',exact_einsum)
  a.total()
  non_scan=sum(int(round(v)) for k,v in a.by_operator().items() if k in legacy.KEEP_OPERATORS)
 scan_count=scan['']['official_scan_macs'];total=non_scan+scan_count
 d=s.d_inner;r=s.dt_rank;n=s.d_state
 assert total==64*64*(3*c*d+9*d+4*d*(2*r+2*n)+4*d*(9*n+1))
 rows.append({'C':c,'input':[1,c,64,64],'ours_macs':o['total_macs'],'ss2d_macs':total,'reduction_percent':100*(1-o['total_macs']/total),'ours_parameters':o['parameters'],'ss2d_parameters':sum(p.numel() for p in s.parameters()),'ours_core_macs':o['propagation_macs'],'ss2d_core_macs':scan_count})
 print(rows[-1],flush=True)
report['standalone_verified']=rows
prior=json.loads((ROOT/'diagnostics/novelty_strengthening_20260910/MAC_COUNT_CPU_RECONCILIATION.json').read_text(encoding='utf-8'))
reconciliation=[]
for rec in report['records']:
 old=next(v for v in prior['records'] if v['task']==rec['task'] and v['mixer']==rec['arm'])
 delta=rec['whole_model_macs']-old['unified_macs']
 assert abs(delta)/old['unified_macs']<0.0001
 reconciliation.append({'task':rec['task'],'arm':rec['arm'],'old_paper_macs':old['unified_macs'],'new_exact_projection_macs':rec['whole_model_macs'],'difference':delta,'relative_difference_percent':100*delta/old['unified_macs']})
report['legacy_rounding_reconciliation']=reconciliation
report['cuda_initialized']=torch.cuda.is_initialized();assert not report['cuda_initialized']
report['source_sha256'][str(Path(__file__))]=legacy.sha(Path(__file__))
(OUT/'RESULTS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
import csv
with (OUT/'single_module.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

print("PASS: standalone MAC counts and reconciliation")
