"""Build an auditable UI bundle from the selected png_600dpi_v2 evidence. No source mutation."""
import csv, json, math, statistics, hashlib
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'eval_plot/png_600dpi_v2'
def rows(p): return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def num(v):
 try: return float(v) if v is not None and v!='' else None
 except (ValueError,TypeError): return None
def mean(v):
 v=[x for x in v if x is not None];return statistics.mean(v) if v else None
def med(v):
 v=[x for x in v if x is not None];return statistics.median(v) if v else None
def gm(v):
 v=[x for x in v if x is not None and x>0];return math.exp(mean([math.log(x) for x in v])) if v else None
def scale(v,s): return v*s if v is not None else None
def complete(v): return sum(v) if v and all(x is not None for x in v) else None
def trajectory(run, axis):
 # Preserve independent exported axes; token and wall-time checkpoints must
 # never be paired by their array positions.
 points=[(b,s) for b,s in run.get('curves',{}).get(axis,[])
         if b is not None and s is not None and math.isfinite(b) and math.isfinite(s)
         and b>=0 and s>0 and (axis!='token' or b<=50_000_000)]
 # A failed search with only the explicit (0, 1x) reference retains that
 # baseline through its recorded consumption, without inventing improvements.
 if points and all(b==0 and s==1 for b,s in points):
  end=min(run['tokens'],50_000_000) if axis=='token' and run.get('tokens') is not None else scale(run.get('wall_hours'),3600) if axis=='time' else None
  if end is not None and end>0:points.append((end,1.))
 return [{'cum_tokens_k':b/1000 if axis=='token' else None,
          'wall_sec':b if axis=='time' else None,'s':s} for b,s in points]
old=json.loads((ROOT/'src/data/operator_bundle.json').read_text())
models={e['engine_model']['id'].lower():e['engine_model'] for e in old['entries']}
for key,vendor,vid,openness,color in [
 ('claude-opus-5','Anthropic','anthropic','closed','#D97757'),
 ('gpt-5.6-terra','OpenAI','openai','closed','#10A37F'),
 ('glm-5.3','Zhipu','zhipu','open','#3A7F96'),
 ('DeepSeek-V4-Flash-BF16','DeepSeek','deepseek','open','#4D6BFE'),
 ('Qwen3.5-122B','Alibaba','alibaba','open','#FF6A00'),
 ('Qwen3.6-35B','Alibaba','alibaba','open','#2E86DE')]:
 models[key.lower()]={'id':key,'vendor':vendor,'vendor_id':vid,'openness':openness,'color':color}
files=['harness_grouped/data/normalized_runs.json','harness_grouped/data/group_metrics.csv','harness_grouped/data/input_manifest.json','model/Default/data_luna_four_modes_20261022/normalized_runs.json','model/Default/data_luna_four_modes_20261022/group_metrics.csv','model/Default/STS/data_budgets_10M_20M_50M_20261022/sts_by_run.csv','model/Default/STS/data_budgets_10M_20M_50M_20261022/sts_by_mode.csv']
def score(r,cap=50_000_000):
 curve=r.get('curves',{}).get('token',[])
 if not curve: return None
 best=1.;penalty=0.
 for b,s in sorted(curve):
  if b is None or s is None or b>cap: continue
  new=max(best,s,1);penalty+=50*b/cap*math.log2(new/best);best=new
 return {'score':100*math.log2(best)-penalty,'base':100*math.log2(best),'deduction':penalty,'speedup':best}
entries=[]; checks=[]
for cohort,base in [('harness','harness_grouped/data'),('default','model/Default/data_luna_four_modes_20261022')]:
 runs=json.loads((SOURCE/base/'normalized_runs.json').read_text()); groups=rows(SOURCE/base/'group_metrics.csv')
 defaults={x['run_id']:x for x in rows(SOURCE/files[5]) if num(x['token_budget'])==50_000_000} if cohort=='default' else {}
 ds={x['group']:x for x in rows(SOURCE/files[6]) if num(x['token_budget'])==50_000_000} if cohort=='default' else {}
 for g in groups:
  rr=[r for r in runs if r['group']==g['group']]; mode=g.get('mode'); model=rr[0]['model']; harness=rr[0]['harness']; modelinfo=models.get(model.lower())
  if modelinfo is None: modelinfo={'id':model,'vendor':'OpenAI' if cohort=='default' else 'Zhipu','vendor_id':'openai' if cohort=='default' else 'zhipu','openness':'closed' if cohort=='default' else 'open','color':g.get('color','#3A7F96')}
  modelinfo=dict(modelinfo)
  if cohort=='default': modelinfo['color']=g['color']
  ops=[]
  for op in sorted({r['operator'] for r in rr}):
   rs=[r for r in rr if r['operator']==op]; detail=[]; ss=[]
   for r in rs:
    s=r.get('sts_new') if cohort=='harness' else None
    if cohort=='default':
     q=defaults[r['run_id']];s={'score':num(q['STS']),'base':num(q['base']),'deduction':num(q['deduction']),'speedup':num(q['best_correct_speedup_within_budget'])}
    if s is None:s=score(r)
    if s and s.get('score') is not None:ss.append(s)
    curve=r.get('curves',{}).get('token',[])
    ctg={}
    for delta in [1,5,10]:
     hits=[b for b,v in curve if b is not None and v is not None and b<=50_000_000 and v>=1+delta/100]
     ctg[f'p{delta}']=min(hits) if hits else None
    detail.append({'run_id':r['run_id'],'status':r.get('status'),'s_bmax':s.get('speedup') if s else None,'search_speedup':r.get('incumbent_speedup'),'baseline_ms':r.get('baseline_ms'),'best_ms':r.get('fastest_correct_ms'),'source':r['source'],'token_evidence':r.get('token_evidence',r.get('token_resolution')),'cost_scope':r.get('cost_scope',r.get('token_cost_basis')),'estimated':bool((s or {}).get('estimated')),'cycles':[],'flags':[],'ctg':ctg})
   first=rs[0]; tokenpoints=first.get('curves',{}).get('token',[])
   terminal=gm([num(r.get('incumbent_speedup')) for r in rs]); budget=gm([s.get('speedup') for s in ss])
   ops.append({'operator':op,'runs':detail,'baseline_quality':'normal' if first.get('baseline_ms') else 'unavailable','flags':[],'baseline_ms':mean([num(r.get('baseline_ms')) for r in rs]),'s_bmax':budget,'search_speedup':terminal,'sts_token':mean([s['score'] for s in ss]),'sts_base':mean([s['base'] for s in ss]),'sts_deduction':mean([s['deduction'] for s in ss]),'vlr':(1-1/budget)*100 if budget else None,'tokens_k':scale(complete([num(r.get('tokens')) for r in rs]),.001),'rounds':mean([num(r.get('llm_rounds')) for r in rs]),'wall_time_s':scale(mean([num(r.get('wall_hours')) for r in rs]),3600),'task_pass_rate':100*mean([int(bool(r.get('baseline_ms')) and bool(r.get('gate'))) for r in rs]),'gate_rate':100*mean([num(r.get('gate')) for r in rs]),'candidate_pass_rate':scale(mean([num(r.get('candidate_pass_rate',r.get('pass_fraction'))) for r in rs]),100),'paos':{},'utilization':{},'duration_by_shape':[],'duration_by_shape_reason':'not_exported_in_supplement','anytime':trajectory(first,'token'),'anytime_time':trajectory(first,'time'),'trajectory_run_id':first['run_id'],'trajectory_token_estimated':bool(first.get('curve_token_estimated',False)),'trajectory_time_estimated':bool(first.get('curve_time_estimated',False)),'anytime_failed':[],'tokens_to':detail[0]['ctg'],'n_solutions':sum(r.get('candidate_submitted',r.get('attempts',0)) or 0 for r in rs),'cost_scope':'Total tokens and mean effort over selected runs, including failed attempts.'})
  groupsts=ds[g['group']] if cohort=='default' else g
  actual=mean([o['sts_token'] for o in ops]);expected=num(groupsts.get('STS',groupsts.get('sts')))
  assert actual is None and expected is None or math.isclose(actual,expected,abs_tol=1e-7),(g['group'],actual,expected)
  checks.append({'group':g['group'],'sts':actual,'expected_sts':expected,'runs':len(rr),'operators':len(ops)})
  budgetgm=gm([o['s_bmax'] for o in ops]); est=sum(bool(r.get('sts_new',{}).get('estimated')) for r in rr)
  overall={'sts_token':actual,'sts_base':num(groupsts.get('base',groupsts.get('sts_base'))),'sts_deduction':num(groupsts.get('deduction',groupsts.get('sts_deduction'))),'s_equiv':None,'sts_time':None,'auls':None,'ebgs':num(g['gm']),'budget_gm':budgetgm,'fast_1':mean([mean([100*int((r.get('s_bmax') or 1)>1) for r in o['runs'] if r.get('s_bmax') is not None]) for o in ops]),'speedup_mean':mean([o['s_bmax'] for o in ops]),'vlr_mean':mean([o['vlr'] for o in ops]),'task_pass_rate':scale(num(g['gate']),100),'gate_rate':scale(num(g['gate']),100),'candidate_pass_rate':scale(num(g['candidate_pass']),100),'tokens_total_k':scale(num(g['tokens_total']),.001),'tokens_mean_k':scale(num(g['tokens_mean']),.001),'rounds_median':med([num(r.get('llm_rounds')) for r in rr]),'wall_time_median_s':scale(med([num(r.get('wall_hours')) for r in rr]),3600),'wall_time_total_s':scale(num(g['wall_hours_total']),3600),'best_duration_us':scale(mean([num(r.get('fastest_correct_ms')) for r in rr]),1000),'flags':[],'eir':{},'paos':{},'verification':{},'cost':{'n_solutions':len(rr),'total_tokens':num(g['tokens_total']),'input_tokens':None,'cached_input_tokens':None,'output_tokens':None,'cache_write_tokens':None},'reliability':{'k':max(len(o['runs']) for o in ops),'reason':'independent_repeats_required'}}
  filename=g['group'].replace('::','-')+'.json'
  entries.append({'id':g['group'].replace('::','+'),'cohort':cohort,'agent':{'id':harness if cohort=='harness' else 'default-'+mode.replace(' ','-'),'type':'general' if cohort=='harness' else 'domain','strategy':'agentic-search' if cohort=='harness' else mode,'version':None},'engine_model':modelinfo,'baseline_quality':'normal','coverage_operators':len(ops),'k_group':max(len(o['runs']) for o in ops),'scenario':None,'overall':overall,'per_operator':ops,'source_file':filename,'estimated_runs':est,'sts_coverage':sum(o['sts_token'] is not None for o in ops),'runs_for_reliability':ops[0]['runs'],'reliability_operator':ops[0]['operator'],'links':{'report':'AscendGYM_TechReport.pdf'}})
meta={**old['meta'],'updated':'2026-10-22','catalog':{},'note':'Selected supplementary records from png_600dpi_v2. Harness and Default are independent cohorts. STS: 50M-token cap. GM: terminal selected trajectory, as in the supplied figure tables. Estimated token trajectories retain their source annotation.','source_files':[e['source_file'] for e in entries],'runs_total':sum(e['overall']['cost']['n_solutions'] for e in entries),'entries_evaluated':len(entries),'metric_directions':old['meta']['metric_directions']}
bundle={'meta':meta,'entries':entries}
(ROOT/'src/data/supplement_bundle.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2)+'\n')
manifest={'source_date':'2026-10-22','harness_runs':289,'default_runs':128,'checks':checks,'sources':{str(p):hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in files},'notes':['Source CSVs take precedence over stale README counts.','Unexported PAOS, AULS, STS_time and billing splits remain null.','GM uses the terminal selected trajectory; STS uses a 50M-token cutoff.','Default no-lessons STS covers 15 of 16 operators.']}
(ROOT/'src/data/supplement_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'entries':len(entries),'runs':meta['runs_total'],'checks':checks},indent=2))
