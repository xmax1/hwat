from pathlib import Path
from jax import numpy as jnp
import paramiko
import subprocess
from time import sleep
from itertools import product
import random
from typing import Any, Iterable
import re
from ast import literal_eval
from pprint import pprint
import numpy as np

this_dir = Path(__file__).parent

### momma bear

def get_cls_dict(cls, ignore=[], _d={}):
    for k,v in cls.__dict__.items():
        if not (k.startswith('_') or k in ignore): 
            if k in cls._children:
                _d[k] = get_cls_dict(v, cls._ignore_attr)
            else:
                _d[k] = getattr(cls, k)
    for k,v in cls.__class__.__dict__.items():
        if not (k.startswith('_') or k in ignore):
            if isinstance(v, property):
                _d[k] = getattr(cls, k)
    pprint(_d)
    return _d


# def get_cls_prop(cls:type,ignore=[],_d={}):
#     for k,v in cls.__class__.__dict__.items():
#         if (not (k.startswith('_') or k in ignore or issubclass(type(v), Sub))) \
#             and :
#                 _d[k] = getattr(cls, k)
#     return _d

def get_cls_sub_cls(cls:type):
    return {k:v for k,v in cls.__dict__.items() \
        if issubclass(type(v), Sub) or isinstance(v, type)}
    
### count things

def count_gpu() -> int: 
    # output = run_cmd('echo $CUDA_VISIBLE_DEVICES', cwd='.') - one day 
    import os
    return sum(c.isdigit() for c in os.environ.get('CUDA_VISIBLE_DEVICES'))

def get_cartesian_product(*args):
    """ Cartesian product is the ordered set of all combinations of n sets """
    return list(product(*args))

def zip_in_n_chunks(arg: Iterable[Any], n: int) -> zip:   
    return zip(*([iter(arg)]*n))

def gen_alphanum(n: int = 7, test=False):
    from string import ascii_lowercase, ascii_uppercase
    random.seed(test if test else None)
    numbers = ''.join([str(i) for i in range(10)])
    characters = ascii_uppercase + ascii_lowercase + numbers
    name = ''.join([random.choice(characters) for _ in range(n)])
    return name

def iterate_folder(folder: Path, iter_exp_dir):
    if iter_exp_dir and folder.exists():
        for i in range(100):
            _folder = add_to_Path(folder, f'-{i}')
            if not re.search(_folder.name, f'-[0-9]*'):
                folder = _folder
                break
        else:
            folder = add_to_Path(folder, f'-0')
    return folder

### do things

def mkdir(path: Path) -> Path:
    path = Path(path)
    if path.suffix != '':
        path = path.parent
    if path.exists():
        print('path exists, leaving alone')
    else:
        path.mkdir(parents=True)
    return path

def add_to_Path(path: Path, string: str | Path):
        return Path(str(path) + str(string))

### convert things

def npify(v):
    return jnp.array(v.numpy())


def cmd_to_dict(cmd:str|list,ref:dict,_d={},delim:str=' --'):
    """
    fmt: [--flag, arg, --true_flag, --flag, arg1]
    # all flags double dash because of negative numbers duh """
    booleans = ['True', 'true', 't', 'False', 'false', 'f']
    
    cmd = ' '.join(cmd) if isinstance(cmd, list) else cmd
    cmd = [x.lstrip().lstrip('--').rstrip() for x in cmd.split(delim)]
    cmd = [x.split(' ', maxsplit=1) for x in cmd if ' ' in x]
    [x.append('True') for x in cmd if len(x) == 1]
    cmd = flat_list(cmd)
    cmd = iter([x.strip() for x in cmd])

    for k,v in zip(cmd, cmd):
        if v in booleans: 
            v=booleans.index(v)<3  # 0-2 True 3-5 False
        if k in ref:
            _d[k] = type(ref[k])(v)
        else:
            try:
                _d[k] = literal_eval(v)
            except:
                _d[k] = str(v)
            print(f'Guessing type: {k} as {type(v)}')
    return _d

def update_cls_with_dict(cls: Any, d:dict):
    cls_all = [v for v in cls.__dict__.values() if issubclass(type(v), Sub)]
    cls_all.extend([cls])
    n_remain = len(d)
    for k,v in d.items():
        for _cls_assign in cls_all:            
            if not hasattr(_cls_assign, k):
                continue
            else:
                if isinstance(cls.__class__.__dict__[k], property):
                    print('Tried to assign property, consider your life choices')
                    continue
                v = type(cls.__dict__)(v)
                setattr(_cls_assign, k, v)
                n_remain -= 1
    return n_remain


def dict_to_wandb(d:dict,parent_key:str='',sep:str ='.',items:list=[])->dict:
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict): 
            print(k,v)
            items.extend(dict_to_wandb(v,new_key,items=items).items())
        else:
            if isinstance(v, Path):  v=str(v)
            items.append((new_key, v))
    return dict(items)

### run things

def run_cmds(cmd:str|list,cwd:str|Path=None,input_req:str=None):
    _out = []
    for cmd_1 in (cmd if isinstance(cmd, list) else [cmd]): 
        cmd_1 = [c.strip() for c in cmd_1.split(' ')]
        _out += [subprocess.run(
            cmd_1,cwd=cwd,input=input_req, capture_output=True)]
        sleep(0.1)
    return _out

def run_cmds_server(server:str,user:str,cmd:str|list,cwd=str|Path):
    _out = []
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # if not known host
    client.connect(hostname=server, username=user)
    client.exec_command(f'cd {cwd}')
    with client as _r:
        for cmd_1 in (cmd if isinstance(cmd, list) else [cmd]):
            _out += [_r.exec_command(f'{cmd_1}')] # in, out, err
            sleep(0.1)
    return _out


# flatten things

def flat_arr(v):
    return v.reshape(v.shape[0], -1)

def flat_list(lst_of_lst):
    return [lst for sublst in lst_of_lst for lst in sublst]

def flat_dict(d:dict,items:list=[]):
    for k,v in d.items():
        if isinstance(v, dict):
            items.extend(flat_dict(v, items=items).items())
        else:
            items.append((k, v))
    return dict(items)

def flat_any(v: list|dict|jnp.ndarray|np.ndarray):
    if isinstance(v, list):
        return flat_list(v)
    if isinstance(v, dict):
        return flat_dict(v)


# class _Sub:
    
#     def __init__(_i, parent) -> None:
#         super().__init__()
#         _i.parent = parent
    
#     @property
#     def dict(_i,):
#         d = _i._dict_from_cls(_i,)
#         for k,v in d.items():
#             if issubclass(type(v), _Sub):  # type(v) on an instantiated class returns the class, which is a subclass of _Sub
#                 d[k] = _i._dict_from_cls(v)
#         return d

#     @staticmethod
#     def _dict_from_cls(cls):
#         return {k: getattr(cls, k) for k in dir(cls) if not k.startswith('_') and not k in ['dict', 'parent', 'sys_arg', 'cmd']}
