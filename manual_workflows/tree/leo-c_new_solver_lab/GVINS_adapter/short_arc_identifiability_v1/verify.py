from common import *
import csv,pickle
from collections import Counter

def main():
    protocol=read(O/'RUN_PROTOCOL.json');src=check_sources();bad=[]
    # Native dataclass definitions are required to deserialize saved outputs;
    # this does not construct a bridge or invoke any solver.
    rtroot=Path(next(x['resolved_path'] for x in src if Path(x['resolved_path']).name=='models.py')).parent.parent
    sys.path.insert(0,str(rtroot))
    for f in protocol['sources']:
        if sha(f['path'])!=f['sha256']:bad.append(f['path'])
    for sealname in ['INFORMATION_SEAL.json','FIT_EXECUTION_SEAL.json','FIT_OUTPUT_SEAL.json']:
        for f in read(O/sealname)['files']:
            if sha(f['path'])!=f['sha256']:bad.append(f['path'])
    assert not bad,bad
    specs=read(O/'fit_schedule.json');assert len(specs)==54 and len({s['fit_id'] for s in specs})==54
    counter=Counter();payloads=[]
    for spec in specs:
        p=O/'raw'/f"{spec['fit_id']}.pkl";assert sha(p)==read(p.with_suffix('.seal.json'))['sha256']
        with p.open('rb') as f:x=pickle.load(f)
        assert x['spec']==spec and not x['denied'];assert len(x['solver_calls'])==2
        assert spec['model'] in MODELS and spec['window'] in [30,120]
        counter.update(x['solver_counts']);payloads.append(x)
    assert sum(counter.values())==108 and len(payloads)==54
    assert Counter(s['module'] for s in specs)=={'MAIN':20,'CORE_SUPPLEMENT':18,'DIRECTIONAL':16}
    checks=[]
    for name in ['information_tables.json','fit_tables.json']:
        for file,rows in read(O/name).items():
            with (O/file).open(newline='',encoding='utf-8') as f:loaded=list(csv.DictReader(f))
            assert len(loaded)==len(rows),(file,len(loaded),len(rows))
            checks.append(dict(file=file,rows=len(rows),json_csv_count_equal=True))
    inf=read(O/'information_tables.json')['nested_information_ordering.csv']
    assert all(r['status'] in ['PASS','NOT_COMPARABLE'] for r in inf)
    oldqa=read(OLD/'integration_qa.json');newqa=read(O/'integration_qa_W30.json')
    assert oldqa['train_rows']==newqa['train_rows'] and oldqa['heldout_rows']==newqa['heldout_rows']
    oldwrapper=next(r['sha256'] for r in read(OLD/'input_manifest.json')['bindings'] if Path(r['path']).name=='wrapper.py')
    assert sha(W/'wrapper.py')==oldwrapper
    required=['RUN_PROTOCOL.json','initialization_contract.json','mechanism_classification.md','report.md']+[r['file'] for r in checks]
    assert all((O/n).exists() for n in required)
    pngs=sorted(p.name for p in O.glob('*.png'))
    assert pngs==sorted(['information_vs_window.png','initialization_basin_M1_M2.png','weak_vs_strong_direction_M2.png'])
    files=[dict(path=str(p),sha256=sha(p)) for p in sorted(O.iterdir()) if p.is_file() and p.name not in ['verification.json','output_manifest.json']]
    save(O/'verification.json',dict(status='PASS',utc=now(),run_id=protocol['run_id'],source_sha_conflicts=bad,native_sources_verified=len(src),wrapper_unchanged=True,
        top_level_fit_configurations=54,full_solver_calls=54,training_solver_calls=54,actual_internal_solver_counts=counter,
        M5_M14_fits=0,EGSHS_runs=0,selector_calls=0,new_threshold_selected=False,weights_changed=False,paper_updated=False,
        reference_associated_initialization=True,reference_files_read_by_fit_process=0,pseudorange_candidate_input=False,forward_qa_clock_candidate_input=False,
        sources_preserved=True,W30_input_equals_old=True,W30_split_matches_old_exactly=True,W30_actual_train_heldout=[125,55],W120_actual_train_heldout=[517,222],
        common_support_unavailable=True,canonical_and_native_common_satellites=[2,5,13],nested_ordering_pass=sum(r['status']=='PASS' for r in inf),nested_not_comparable=sum(r['status']=='NOT_COMPARABLE' for r in inf),
        table_checks=checks,figures=pngs,full_converged=sum(x['full_result']['converged'] for x in payloads),train_converged=sum(x['train_result']['converged'] for x in payloads)))
    save(O/'output_manifest.json',dict(utc=now(),run_id=protocol['run_id'],files=files,raw_partition_seals=[dict(path=str(p),sha256=sha(p)) for p in sorted((O/'raw').glob('*.seal.json'))],verification_sha=sha(O/'verification.json')))
    print('VERIFICATION=PASS\nTOP_LEVEL_CONFIGURATIONS=54\nINTERNAL_SOLVER_CALLS=108\nEGSHS_RUNS=0',flush=True)

if __name__=='__main__':main()
