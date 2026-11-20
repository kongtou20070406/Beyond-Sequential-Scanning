"""Aggregate every measured module/image and intervention; no sample selection."""
from pathlib import Path
import csv,json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent
def stats(x):
    a=np.asarray(x,dtype=float)
    return dict(min=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),max=float(a.max()))
summary={}; flat=[]
for arm in ['full','no_gain','free_gain','state_dynamic']:
    files=sorted(P.glob(f'profile_{arm}_???.json')); records=[]
    for f in files:
        d=json.loads(f.read_text()); records.extend(d['records'])
        for r in d['records']:
            for s in r['steps']:flat.append(dict(arm=arm,image=d['id'],module=r['module'],**s))
    if not records:continue
    def vals(k,t=4):return [r['steps'][t][k] for r in records]
    summary[arm]=dict(images=len(files),module_inputs=len(records),q=stats(vals('q')),eta5=stats(vals('eta_mean')),
        local_frozen_jacobian_upper5=stats(vals('local_jacobian_upper')),fraction_rows_over_one5=stats(vals('fraction_rows_over_one')),
        increment_ratio5=stats(vals('increment_ratio')),state_norm5=stats(vals('h')),state_rms5=stats(vals('state_rms')),
        tail5to15=stats([r['h15_minus_h5_over_h5'] for r in records]),
        direction_total=stats([r['directional_jacobian'][0]['total_direction_gain'] for r in records]),
        direction_frozen=stats([r['directional_jacobian'][0]['frozen_kernel_gain'] for r in records]),
        direction_extra=stats([r['directional_jacobian'][0]['extra_kernel_feedback_gain'] for r in records]),
        fd_scale_difference=stats([abs(r['directional_jacobian'][0]['total_direction_gain']-r['directional_jacobian'][1]['total_direction_gain']) for r in records]),
        relative_decomposition_residual_max=max(s['decomposition_error']/max(s['h'],s['increment'],1e-10) for r in records for s in r['steps'][1:]),
        drift_ratio5=stats(vals('drift_to_increment')),strength_mean=stats([r['strength_mean'] for r in records]))
    if 'learned_gain' in records[0]:summary[arm]['learned_gain']=stats([r['learned_gain'] for r in records])
(P/'profile_statistics.json').write_text(json.dumps(summary,indent=2))
if flat:
    with (P/'profiles.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(x.keys() for x in flat))));w.writeheader();w.writerows(flat)
inter=[]
for arm in ['full','no_gain']:
    data=[json.loads(f.read_text()) for f in sorted(P.glob(f'intervene_{arm}_???.json'))]
    if not data:continue
    key='scale_1.0' if arm=='full' else 'native'
    for i,entry in enumerate(data[0]['scores']):
        scores=[d['scores'][i]['psnr'] for d in data]
        baseline=[next(s['psnr'] for s in d['scores'] if s['mode']==key and not s['output_rms_matched']) for d in data]
        diff=np.array(scores)-baseline
        inter.append(dict(arm=arm,mode=entry['mode'],matched=entry['output_rms_matched'],n=len(data),psnr_mean=float(np.mean(scores)),
                          delta_mean=float(diff.mean()),delta_median=float(np.median(diff)),delta_min=float(diff.min()),delta_max=float(diff.max()),
                          improved=int((diff>0).sum()),worsened=int((diff<0).sum())))
(P/'intervention_statistics.json').write_text(json.dumps(inter,indent=2))
if inter:
    with (P/'interventions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(inter[0]));w.writeheader();w.writerows(inter)
fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
colors=dict(full='#225ea8',no_gain='#d95f0e',free_gain='#238b45',state_dynamic='#8856a7')
for arm in summary:
    rs=[r for f in P.glob(f'profile_{arm}_???.json') for r in json.loads(f.read_text())['records']]
    for ax,key,norm in [(axes[0],'h',True),(axes[1],'increment',True),(axes[2],'eta_mean',False)]:
        arr=np.array([[s[key] for s in r['steps']] for r in rs])
        if norm:arr=arr/arr[:,0:1]
        ax.plot(range(1,16),np.median(arr,axis=0),label=arm,color=colors[arm])
        ax.set_xlabel('Local propagation step');ax.grid(alpha=.2)
for ax,title in zip(axes,['State norm / step-1 norm','Increment norm / step-1 norm','Mean retention factor']):ax.set_title(title)
axes[1].set_yscale('log');axes[2].legend(fontsize=8)
fig.savefig(P/'state_diagnostics.png',dpi=180);plt.close(fig)
if inter:
    fig,ax=plt.subplots(figsize=(6,3.6),layout='constrained')
    a=[r for r in inter if r['arm']=='full' and not r['matched']]
    ax.plot([float(r['mode'].split('_')[1]) for r in a],[r['delta_mean'] for r in a],'-o',color=colors['full'])
    for r in inter:
        if r['arm']=='full' and r['matched']:ax.scatter(float(r['mode'].split('_')[1]),r['delta_mean'],marker='s',color='#238b45',s=60)
    ax.axhline(0,c='gray',lw=.8);ax.axvline(1,c='gray',lw=.8,ls='--');ax.set_xlabel('Multiplier on all propagation kernels');ax.set_ylabel('Mean PSNR change (dB)');ax.grid(alpha=.2)
    ax.set_title('Frozen full model: 20 Haze4K images\nGreen squares: module-output RMS matched')
    fig.savefig(P/'gain_intervention.png',dpi=180);plt.close(fig)
print(json.dumps(dict(profiles=summary,interventions=inter),indent=2))
