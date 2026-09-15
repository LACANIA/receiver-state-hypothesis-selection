"""Check source integrity, syntax and optional imports; no experiments."""
from pathlib import Path
import argparse, ast, hashlib, json, os, subprocess, sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--imports',action='store_true');args=parser.parse_args()
    errors=[]; imports=[]
    records=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())['files']
    for row in records:
        p=ROOT/row['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']: errors.append('Hash/file: '+row['path'])
    py=list(ROOT.rglob('*.py'))
    for p in py:
        try: tree=ast.parse(p.read_text(encoding='utf-8-sig'))
        except SyntaxError as e: errors.append(str(p.relative_to(ROOT))+': '+str(e));continue
        if p.parent.name=='leo_positioning':
            for n in ast.walk(tree):
                if isinstance(n,ast.ImportFrom) and n.level==1 and n.module and not (p.parent/(n.module.split('.')[0]+'.py')).is_file():
                    errors.append('Missing local import: '+str(p.relative_to(ROOT))+' -> '+n.module)
    forbidden={'.bag','.parquet','.npy','.npz','.png','.jpg','.jpeg','.pdf','.svg','.tif','.tiff','.docx'}
    for p in ROOT.rglob('*'):
        if p.is_file() and p.suffix.lower() in forbidden: errors.append('Non-code asset: '+str(p.relative_to(ROOT)))
    if args.imports:
        dirs=sorted({p.parent for p in ROOT.rglob('leo_positioning/__init__.py')})
        dirs.append(ROOT/'src/flagship')
        for d in dirs:
            names=[d.name+'.'+p.stem for p in d.glob('*.py') if p.name!='__init__.py']
            code='import sys,importlib;sys.dont_write_bytecode=True;sys.path.insert(0,'+repr(str(ROOT/'src'))+');sys.path.insert(0,'+repr(str(d.parent))+');[importlib.import_module(n) for n in '+repr(names)+'];print(len('+repr(names)+'))'
            child=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,env=os.environ,timeout=60)
            if child.returncode: errors.append('Import '+str(d.relative_to(ROOT))+': '+child.stderr[-2500:])
            else: imports.append({'package':d.relative_to(ROOT).as_posix(),'modules':len(names)})
        code='import sys,json;sys.dont_write_bytecode=True;sys.path.insert(0,'+repr(str(ROOT/'src'))+');from leo_positioning.ma_bgtr_v4_solver import V4_CANDIDATE_MODEL_LIST;from flagship import selector_replay;assert len(V4_CANDIDATE_MODEL_LIST)==15;json.load(open('+repr(str(ROOT/'configs/selector_spec.json'))+'));print("M0-M14 and selector PASS")'
        child=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=60)
        if child.returncode: errors.append(child.stderr)
    result={'status':'FAIL' if errors else 'PASS','python_files':len(py),'source_records':len(records),'import_checks':imports,'errors':errors,'experiments_run':0}
    print(json.dumps(result,indent=2));return int(bool(errors))

if __name__=='__main__':raise SystemExit(main())
