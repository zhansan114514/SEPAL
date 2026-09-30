"""Prepare a standalone rerun workspace from the released five-model configs.

This only copies code and resolves paths; it never starts a GPU experiment.
Use an empty destination so that existing experiments cannot be overwritten.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('llama3_8b','qwen25_3b','gemma2_2b','phi4_mini','mistral7b_v03')


def prepare(workspace, model_root=None, data_root=None):
    workspace = Path(workspace).resolve()
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError('Destination must be empty: '+str(workspace))
    model_root = Path(model_root).resolve() if model_root else workspace/'models'
    data_root = Path(data_root).resolve() if data_root else workspace/'benchmark_data/acccollab_paper'
    shutil.copytree(ROOT/'code',workspace,dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','*.pyc'))
    records=[]
    for model in MODELS:
        inputs = sorted((ROOT/'configs'/model).glob('*.yaml'))
        if len(inputs)!=10:
            raise ValueError(f'{model}: expected ten training/evaluation configs, got {len(inputs)}')
        target_dir=workspace/'configs/reproduction'/model
        target_dir.mkdir(parents=True)
        for source in inputs:
            payload=yaml.safe_load(source.read_text(encoding='utf-8'))
            def resolve(value):
                if isinstance(value,dict):return {k:resolve(v) for k,v in value.items()}
                if isinstance(value,list):return [resolve(v) for v in value]
                if isinstance(value,str):
                    value=value.replace('PROJECT_ROOT',workspace.as_posix()).replace('DATA_ROOT',data_root.as_posix())
                    if value.startswith('output/'):
                        value=(workspace/value).as_posix()
                return value
            payload=resolve(payload)
            if 'base_acccollab_config' in payload:
                basename=Path(payload['base_acccollab_config']).name
                payload['base_acccollab_config']=(target_dir/basename).as_posix()
                if not (ROOT/'configs'/model/basename).exists():
                    raise FileNotFoundError('Missing paired base configuration: '+basename)
            else:
                name=Path(payload['model']['name']).name
                payload['model']['name']=(model_root/name).as_posix()
                payload['data']['benchmark_data_dir']=data_root.as_posix()
            target=target_dir/source.name
            target.write_text(yaml.safe_dump(payload,sort_keys=False,allow_unicode=True),encoding='utf-8')
            records.append({'source':source.relative_to(ROOT).as_posix(),
                            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                            'resolved':target.relative_to(workspace).as_posix()})
    report={'status':'prepared','models':list(MODELS),'configuration_count':len(records),
            'path_changes_only':True,'files':records}
    (workspace/'PREPARATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True,type=Path)
    parser.add_argument('--model-root',type=Path)
    parser.add_argument('--data-root',type=Path)
    args=parser.parse_args()
    prepare(args.workspace,args.model_root,args.data_root)
