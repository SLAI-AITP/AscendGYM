"""Validate the published snapshot against independent source CSVs and per-run traces."""
import csv,json,math
from pathlib import Path
from collections import defaultdict
R=Path(__file__).resolve().parents[1];S=R.parent/'eval_plot/png_600dpi_v2'
b=json.loads((R/'src/data/supplement_bundle.json').read_text())
def csvrows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def close(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-7),(a,b)
def mean(x):return sum(x)/len(x)
h=[e for e in b['entries'] if e['cohort']=='harness'];d=[e for e in b['entries'] if e['cohort']=='default']
assert len(h)==17 and sum(e['overall']['cost']['n_solutions'] for e in h)==289
assert len(d)==4 and sum(e['overall']['cost']['n_solutions'] for e in d)==128
for cohort,folder in [('harness','harness_grouped/data'),('default','model/Default/data_luna_four_modes_20261022')]:
 rows={r['group']:r for r in csvrows(S/folder/'group_metrics.csv')}
 for e in [x for x in b['entries'] if x['cohort']==cohort]:
  r=rows[e['id'].replace('+','::')];o=e['overall']
  close(o['ebgs'],float(r['gm']));close(o['tokens_total_k']*1000,float(r['tokens_total']))
  close(o['candidate_pass_rate'],float(r['candidate_pass'])*100)
  assert all(v is None for k,v in o['cost'].items() if k in ['input_tokens','output_tokens','cached_input_tokens','cache_write_tokens'])
  assert not o['paos']
  assert len(e['per_operator'])==int(r['operators'])
  if cohort=='default':assert o['rounds_median'] is None and o['wall_time_median_s'] is None
runs=json.loads((S/'harness_grouped/data/normalized_runs.json').read_text())
checked=0
for r in runs:
 points=r.get('curves',{}).get('token',[]);ref=r.get('sts_new')
 if not points or not ref or ref.get('score') is None:continue
 best=1.;penalty=0.
 for tokens,speed in sorted(points):
  if tokens is None or speed is None or tokens>50_000_000:continue
  new=max(best,speed);penalty+=50*tokens/50_000_000*math.log2(new/best);best=new
 close(100*math.log2(best)-penalty,ref['score']);checked+=1
assert checked>250
nl=next(e for e in d if e['id']=='kl-nl');assert nl['sts_coverage']==15 and nl['coverage_operators']==16
assert not any(e['agent']['id']=='codex' and e['engine_model']['id']=='claude-opus-5' for e in h)
time_cells=0;baseline_cells=0
for cohort,folder in [('harness','harness_grouped/data'),('default','model/Default/data_luna_four_modes_20261022')]:
 source=json.loads((S/folder/'normalized_runs.json').read_text())
 selected={}
 for run in source:selected.setdefault((run['group'],run['operator']),run)
 for e in [x for x in b['entries'] if x['cohort']==cohort]:
  for op in e['per_operator']:
   run=selected[(e['id'].replace('+','::'),op['operator'])]
   assert op['trajectory_run_id']==run['run_id']
   for axis,field,key,scale in [('token','anytime','cum_tokens_k',.001),('time','anytime_time','wall_sec',1)]:
    raw=[p for p in run['curves'][axis] if axis!='token' or p[0]<=50_000_000]
    actual=op[field]
    if raw and all(p==[0,1] for p in raw):
     # A zero-only reference can extend only through actual recorded cost.
     end=min(run['tokens'],50_000_000) if axis=='token' else run['wall_hours']*3600
     assert len(actual)==len(raw)+(end>0)
     if end>0:close(actual[-1][key],end*scale);close(actual[-1]['s'],1)
     baseline_cells+=axis=='time'
    else:assert len(actual)==len(raw)
    for point,ref in zip(actual,raw):close(point[key],ref[0]*scale);close(point['s'],ref[1])
    if axis=='time':time_cells+=bool(actual)
assert time_cells==289 and baseline_cells==9
assert sum(not op['anytime'] for e in b['entries'] for op in e['per_operator'])==1
print(f'PASS: 21 groups, 417 selected runs, source GM/token/pass-rate agreement, {checked} independent STS reconstructions, missingness and cohort isolation.')
print(f'PASS: {time_cells} source-backed wall-time kernel trajectories, including {baseline_cells} baseline-only cases; Default time and one missing token trace remain unavailable.')
