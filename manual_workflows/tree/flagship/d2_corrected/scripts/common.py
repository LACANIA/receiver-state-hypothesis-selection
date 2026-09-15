import sys, os, json, hashlib, shutil, importlib.util, ast
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
sys.dont_write_bytecode = True
F=Path('flagship')
T=F/'d2_corrected'
A=F/'53_routeA_critical_scientific_integrity_audit/PAPER_FLAGSHIP_ROUTE_A_CRITICAL_SOURCE_TO_CLAIM_AUDIT'
W=Path('leo-c')
R=W/'audit_runs/D2_epoch110_correction_20260905_run01'
def plain(p):
 s=str(p);return s[4:] if s.startswith('\\\\?\\') else s
def path(s):
 s=plain(s);return Path('\\\\?\\'+s)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def js(n,v):
 p=T/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n',encoding='utf-8')
def csv(n,v):
 d=v if isinstance(v,pd.DataFrame) else pd.DataFrame(v);p=T/n;p.parent.mkdir(parents=True,exist_ok=True);d.to_csv(p,index=False,encoding='utf-8-sig');return d
def md(n,s):
 p=T/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8')
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def guard():
 root=os.path.normcase(os.path.abspath(plain(T)))
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   name=os.fsdecode(args[0]);low=name.lower()
   if any(k in low for k in ['egshs_dynamic_pool','external_dynamic_data','ca_prototype','dynamic_pool_vnext']):raise RuntimeError('Route B access rejected')
   mode=args[1];flags=args[2] if len(args)>2 else 0
   writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
   if writing:
    resolved=os.path.normcase(os.path.abspath(plain(name)))
    if not (resolved==root or resolved.startswith(root+os.sep)):raise RuntimeError('write outside TASK_ROOT rejected: '+name)
 sys.addaudithook(audit)
def inputs():return json.loads((T/'FROZEN_REVALIDATION_INPUT_MANIFEST.json').read_text(encoding='utf-8'))
def byrole(role):return [path(x['path']) for x in inputs()['files'] if x['role']==role]
def sources():
 sys.path.insert(0,str(T/'source_snapshot/release_src'))
 sys.path.insert(0,str(T/'source_snapshot/original_scripts'))
 return module(T/'source_snapshot/original_scripts/stage_methods.py','frozen_stage_methods'),module(T/'source_snapshot/selector/selector_spec_replay_v052.py','frozen_selector'),module(T/'source_snapshot/original_scripts/evaluate_d2_truth.py','frozen_evaluator')
