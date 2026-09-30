"""Recompute votes and correctness from released per-example answer records.

Uses the Python standard library. This is a decision audit of stored runs;
it does not regenerate model responses or estimate variation across seeds.
"""
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
ROLES = ('direct','evidence','verification')
COUNTS = dict(boolq=3270,mmlu=14042,bbh=1260,sciq=1000,arc=3548)


def normalized(value, task):
    if value is None:
        return None
    value = str(value).strip().strip('().').strip().upper()
    if task == 'yes_no':
        return {'YES':'YES','Y':'YES','NO':'NO','N':'NO'}.get(value)
    if task in ('multiple_choice','mixed'):
        if len(value) == 1 and 'A' <= value <= 'Z':
            return value
        return {'YES':'YES','Y':'YES','NO':'NO','N':'NO'}.get(value) if task == 'mixed' else None
    raise ValueError('Unreviewed task type: '+task)


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def verify(output_path=None):
    directory = ROOT/'results/decisions'
    manifest = json.loads((directory/'MANIFEST.json').read_text())
    require(len(manifest['files']) == 125, 'Incomplete prediction matrix')
    totals = Counter()
    results = []
    for entry in manifest['files']:
        path = directory/entry['path']
        require(hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'], str(path)+' hash mismatch')
        variant,model,dataset = (entry[k] for k in ('variant','model','dataset'))
        parent = ROOT/'results/ablations'/model
        metric_path = (parent/'offline_full'/dataset/'metrics.json' if variant=='full' else
                       ROOT/'results/extended_ablations'/model/dataset/(variant+'.json') if variant in ('sft_trained_critic','no_sft_full') else
                       parent/'policy_lattice'/dataset/variant/'aggregate/metrics.json')
        metrics = json.loads(metric_path.read_text())
        counts = [Counter() for _ in range(1 if variant=='sft_only' else 5)]
        ids = set()
        transition = Counter()
        with gzip.open(path,'rt',encoding='utf-8') as source:
            for r in map(json.loads,source):
                require(r['sample_id'] not in ids, 'Duplicate sample ID')
                ids.add(r['sample_id'])
                require(len(r['rounds'])==len(counts), 'Unexpected round count')
                gold = normalized(r['gold_answer'],r['task_type'])
                require(gold is not None,'Invalid gold answer')
                flags=[]
                for k,rd in enumerate(r['rounds']):
                    votes={role:normalized(rd['role_votes'][role],r['task_type']) for role in ROLES}
                    require(rd['round']==k,'Round index mismatch')
                    valid=Counter(v for v in votes.values() if v is not None)
                    top=valid.most_common(1)
                    majority=bool(top and top[0][1]>=2)
                    answer=top[0][0] if majority else votes['direct']
                    correct=answer is not None and answer==gold
                    require(answer==normalized(rd['decision']['answer'],r['task_type']),'Vote mismatch')
                    require(correct==rd['correct'],'Stored correctness mismatch')
                    require(majority==rd['decision']['majority_reached'],'Majority flag mismatch')
                    c=counts[k]; c['n']+=1;c['correct']+=correct;c['majority']+=majority
                    c['unanimous']+=bool(top and top[0][1]==3)
                    c['parse']+=answer is not None;c['oracle']+=gold in votes.values()
                    for role,v in votes.items():c[role]+=v is not None and v==gold
                    for a,b in [('direct','evidence'),('direct','verification'),('evidence','verification')]:
                        agree=votes[a] is not None and votes[a]==votes[b]
                        c[a+'+'+b]+=agree;c[a+'+'+b+'_correct']+=agree and votes[a]==gold
                    totals['decisions']+=1
                    flags.append(correct)
                if len(flags)>1:
                    transition['r0_wrong_r1_right']+=not flags[0] and flags[1]
                    transition['r0_right_r1_wrong']+=flags[0] and not flags[1]
                    transition['r0_wrong_r4_right']+=not flags[0] and flags[-1]
                    transition['r0_right_r4_wrong']+=flags[0] and not flags[-1]
        n=len(ids)
        require(n==COUNTS[dataset]==entry['examples'],'Sample count mismatch')
        for k,c in enumerate(counts):
            recorded=metrics['per_round'][k]
            for metric,key in [('accuracy','correct'),('majority_coverage','majority'),('unanimous_rate','unanimous'),('oracle_any_role_accuracy','oracle'),('parse_rate','parse')]:
                if metric in recorded:
                    require(abs(c[key]/n-recorded[metric])<1e-12,entry['path']+' '+metric+' mismatch')
                    totals['aggregate_checks']+=1
            for role in ROLES:
                if 'per_role' in recorded:
                    require(abs(c[role]/n-recorded['per_role'][role]['accuracy'])<1e-12,'Role accuracy mismatch')
                    totals['aggregate_checks']+=1
        final=counts[-1]
        for pair,recorded in metrics.get('pair_agreement',{}).items():
            require(final[pair]==recorded['agree'],'Pair agreement count mismatch')
            require(final[pair+'_correct']==recorded['agree_correct'],'Pair correctness count mismatch')
            totals['aggregate_checks']+=2
        totals['example_records']+=n;totals['files']+=1
        result=dict(model=model,dataset=dataset,variant=variant,samples=n,
                    accuracies=[c['correct']/n for c in counts],
                    transition_counts=dict(transition))
        if variant=='full':result['transition_rates_percent']={k:100*v/n for k,v in transition.items()}
        results.append(result)
    full=[r for r in results if r['variant']=='full']
    transition_means={k:mean(r['transition_rates_percent'][k] for r in full) for k in full[0]['transition_rates_percent']}
    output={'status':'pass',**dict(totals),'scope':'Recomputed normalized answers, votes, correctness and aggregates for stored runs; no model inference.','mean_transition_percent':transition_means,'cells':results}
    if output_path is not None:
        dest=Path(output_path)
        if dest.resolve().is_relative_to((ROOT/'results').resolve()):
            raise ValueError('Write a new audit outside the immutable released results directory.')
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k!='cells'},indent=2))
    return output


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Optional new report path outside results/. Stored evidence is never overwritten.')
    verify(parser.parse_args().output)
