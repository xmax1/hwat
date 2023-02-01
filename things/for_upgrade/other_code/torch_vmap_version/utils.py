from pathlib import Path
import sys
import paramiko
import subprocess
from time import sleep
from itertools import product
from functools import partial
import random
from typing import Any, Iterable
import re
from ast import literal_eval
from typing import Union
import os
import pprint
import torch
import optree

import numpy as np
from copy import copy

dl_arr = torch.Tensor

this_dir = Path(__file__).parent

### Pyfig Subclass 

class Sub:
    _p = None
    _ignore = ['d', '_d'] 

    def __init__(ii, parent=None):
        ii._p = parent
    
    @property
    def d(ii,):
        out={} # becomes class variable in call line, accumulates
        for k,v in ii.__class__.__dict__.items():
            if k.startswith('_') or k in ii._ignore:
                continue
            if isinstance(v, partial): 
                v = ii.__dict__[k]   # if getattr then calls the partial, which we don't want
            else:
                v = getattr(ii, k)
            out[k] = v
        return out

### metrics ###

def collect_stats(k, v, new_d, p='tr', suf='', sep='/', sep_long='-'):
    depth = p.count('/')
    if depth > 1:
        sep = sep_long
    if isinstance(v, dict):
        for k_sub,v_sub in v.items():
            collect_stats(k, v_sub, new_d, p=(p+sep+k_PlugIn))
    elif isinstance(v, list):
        for i, v_sub in enumerate(v):
            collect_stats(k, v_sub, new_d, p=(p+sep+k+str(i)))
    else:
        new_d[p+sep+k+suf] = v
    return new_d

### debug things

def debug_mode(on=False):
    if on:
        os.environ['debug'] = 'debug'
    else:
        os.environ['debug'] = ''

def debug_pr(d:dict):
    if os.environ.get('debug') == 'debug':
        for k,v in d.items():
            typ = type(v) 
            has_shape = hasattr(v, 'shape')
            shape = v.shape if has_shape else None
            dtype = v.dtype if hasattr(v, 'dtype') else None
            try:
                mean = torch.mean(v) if has_shape else v
                std = torch.std(v) if has_shape else None
            except:
                mean = v
                std = None
            print(k, f'\t mean={mean} \t std={std} \t shape={shape} \t dtype={dtype}') # \t type={typ}

### count things

def count_gpu() -> int: 
    # output = run_cmd('echo $CUDA_VISIBLE_DEVICES', cwd='.') - one day 
    import os
    device = os.environ.get('CUDA_VISIBLE_DEVICES') or 'none'
    return sum(c.isdigit() for c in device)

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

def iterate_n_dir(folder: Path, iterate_state, n_max=1000):
    if iterate_state:
        if not re.match(folder.name, '-[0-9]*'):
            folder = add_to_Path(folder, '-0')
        for i in range(n_max+1):
            folder = folder.parent / folder.name.split('-')[0]
            folder = add_to_Path(folder, f'-{i}')
            if not folder.exists():
                break   
    return folder

### do things

def mkdir(path: Path) -> Path:
    path = Path(path)
    if path.suffix != '':
        path = path.parent
    if not path.exists():
        path.mkdir(parents=True)
    return path

def add_to_Path(path: Path, string: Union[str, Path]):
    return Path(str(path) + str(string))

### convert things

def npify(v):
    return torch.tensor(v.numpy())

def format_cmd_item(v):
    v = v.replace('(', '[').replace(')', ']')
    return v.replace(' ', '')

def to_cmd(d:dict):
    """ Accepted: int, float, str, list, dict, np.ndarray"""
    
    def prep_cmd_item(k:str, v:Any):
        if isinstance(v, np.ndarray):
            v = v.tolist()
        return str(k).replace(" ", ""), str(v).replace(" ", "")
    
    return dict(prep_cmd_item(k,v) for k,v in d.items())
    
def type_me(v, v_ref=None, is_cmd_item=False):
    def count_leading_char(s, char): 
        # space=r"^\s*" bracket=r'^[*'
        match = re.search(rf'^{char}*', s)
        return 0 if not match else match.end()
    
    if is_cmd_item:
        """ Accepted: 
        bool, list of list (str, float, int), dictionary, str, explicit str (' "this" '), """
        v = format_cmd_item(v)
        
        print(v, v_ref)
        if v.startswith('[['):
            v = v.strip('[]')
            nest_lst = v.split('],[')
            return [type_me('['+lst+']', v_ref[0], is_cmd_item=True) for lst in nest_lst]
        
        if v.startswith('['):
            v = v.strip('[]')
            v = v.split(',')
            return [type_me(x, v_ref[0]) for x in v]
        
        booleans = ['True', 'true', 't', 'False', 'false', 'f']
        if v in booleans: 
            return booleans.index(v) < 3  # 0-2 True 3-5 False
    
    if v_ref is not None:
        type_ref = type(v_ref)
        if isinstance(v, str):
            v = v.strip('\'\"')
        
        # print(type(v), type(v_ref), v, v_ref)
        if isinstance(v, (np.ndarray, np.generic)):
            if isinstance(v.flatten()[0], str):
                return v.tolist()
            return v
        
        if isinstance(v, list):
            if isinstance(flat_any(v)[0], str):
                return v
            return np.asarray(v)
        
        if isinstance(v, torch.Tensor):
            return v

        return type_ref(v)
        
    try:
        return literal_eval(v)
    except:
        return str(v).strip('\'\"')
    
    
def cmd_to_dict(cmd:Union[str, list], ref:dict, delim:str=' --', d=None):
    """
    fmt: [--flag, arg, --true_flag, --flag, arg1]
    # all flags double dash because of negative numbers duh """
    cmd = ' ' + (' '.join(cmd) if isinstance(cmd, list) else cmd)  # add initial space in case single flag
    cmd = [x.strip().lstrip('--') for x in cmd.split(delim)][1:]
    cmd = [x.split('=', maxsplit=1) if '=' in x else x.split(' ', maxsplit=1) for x in cmd]
    [x.append('True') for x in cmd if len(x)==1]
    
    d = dict()
    for k,v in cmd:
        v = format_cmd_item(v)
        k = k.replace(' ', '')
        v_ref = ref.get(k, None)
        if v_ref is None:
            print(f'{k} not in ref')
        d[k] = type_me(v, v_ref, is_cmd_item=True)
    return d
    
### run things



def run_cmds(cmd:Union[str, list], cwd:Union[str, Path]='.', _res=[]):
    return 'DidNotRun'
    for cmd_1 in (cmd if isinstance(cmd, list) else [cmd]): 
        cmd_1 = [c.strip() for c in cmd_1.split(' ')]
        print(f'Run: {cmd_1} at {cwd}')
        _res = subprocess.run(cmd_1, cwd=str(cwd), capture_output=True, text=True)
        print('stdout:', _res.stdout.replace("\n", " "), 'stderr:', _res.stderr.replace("\n", ";"))
    return _res.stdout.replace('\n', ' ')

def run_cmds_server(server:str, user:str, cmd:Union[str, list], cwd=Union[str, Path], _res=[]):
    client = paramiko.SSHClient()    
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # if not known host
    client.connect(server, username=user)
    for cmd_1 in (cmd if isinstance(cmd, list) else [cmd]):
        print(f'Remote run: {cmd_1} at {user}@{server}:{cwd}')
        stdin, stdout, stderr = client.exec_command(f'cd {str(cwd)}; {cmd_1}')
        stderr = '\n'.join(stderr.readlines())
        stdout = '\n'.join(stdout.readlines())
        print('stdout:', stdout, 'stderr:', stderr)
    client.close()
    return stdout.replace("\n", " ")

    
    
# flatten things

def flat_arr(v):
    return v.reshape(v.shape[0], -1)

def flat_list(lst):
    items = []
    for v in lst:
        if isinstance(v, list):
            items.extend(flat_list(v))
        else:
            items += [v]
    return items

def flat_dict(d:dict):
    items = []
    for k,v in d.items():
        if isinstance(v, dict):
            items.extend(flat_dict(v).items())
        else:
            items.append((k, v))
    return dict(items)

def flat_any(v: Union[list, dict]):
    if isinstance(v, list):
        return flat_list(v)
    if isinstance(v, dict):
        return flat_dict(v)


### wandb ###

def dict_to_wandb(
    d:dict, 
    parent='', 
    sep='.', 
    ignore=[],
    _l:list=None,
    )->dict:
    _l = [] if _l is None else _l
    for k, v in d.items():
        if isinstance(v, Path) or callable(v):
            continue
        if k in ignore:
            continue
        k_1 = parent + sep + k if parent else k
        if isinstance(v, dict):
            _l.extend(dict_to_wandb(v, k_1, _l=_l).items())
        elif callable(v):
            continue
        _l.append((k_1, v))
    return dict(_l)



try:

    def compute_metrix(d:dict, mode='tr', fancy=None, ignore = [], _d = {}):
        
        for k,v in d.items():
            if any([ig in k for ig in ignore+['step']]):
                continue 
            
            if not fancy is None:
                k = fancy.get(k, k)

            # v = jax.device_get(v)
            
            v_mean = optree.tree_map(lambda x: x.mean() if not np.isscalar(x) else x, v)  if not np.isscalar(v) else v
            v_std = optree.tree_map(lambda x: x.std() if not np.isscalar(x) else x, v)  if not np.isscalar(v) else v
            
            group = mode
            if 'grad' in k:
                group = mode + '/grad'
            elif 'param' in k:
                group += '/param'
                
            _d = collect_stats(k, v_mean, _d, p=group, suf=r'_\mu$')
            _d = collect_stats(k, v_std, _d, p=group+'/std', suf=r'_\sigma$')

        return _d

except: 
    print('Metrix: No Torch')
# ### jax ###

# try:
#     from flax.core.frozen_dict import FrozenDict
#     from jax import random as rnd
    
#     def gen_rng(rng, n_device):
#         """ rng generator for multi-gpu experiments """
#         rng, rng_p = torch.split(rnd.split(rng, n_device+1), [1,])
#         return rng.squeeze(), rng_p	
        

#     def compute_metrix(d:dict, mode='tr', fancy=None, ignore = [], _d = {}):
        
#         for k,v in d.items():
#             if any([ig in k for ig in ignore+['step']]):
#                 continue 
            
#             if not fancy is None:
#                 k = fancy.get(k, k)

#             v = jax.device_get(v)
            
#             if isinstance(v, FrozenDict):
#                 v = v.unfreeze()
            
#             v_mean = jax.tree_map(lambda x: x.mean(), v) if not np.isscalar(v) else v
#             v_std = jax.tree_map(lambda x: x.std(), v) if not np.isscalar(v) else 0.

#             group = mode
#             if 'grad' in k:
#                 group = mode + '/grad'
#             elif 'param' in k:
#                 group += '/param'
                
#             _d = collect_stats(k, v_mean, _d, p=group, suf=r'_\mu$')
#             _d = collect_stats(k, v_std, _d, p=group+'/std', suf=r'_\sigma$')

#         return _d

#     ### type testing ### 

#     def test_print_fp16_no_cast():
#         x = torch.ones([1], dtype='float16')
#         print(x)  # FAILS

#     def test_print_fp16():
#         x = torch.ones([1], dtype='float16')
#         x = x.astype('float16')
#         print(x)  # OK

#     def test_print_fp32():
#         x = torch.ones([1], dtype='float16')
#         x = x.astype('float16')
#         x = x.astype('float32')
#         print(x)  # OK

#     def test_print_fp32_to_fp16_cast():
#         x = torch.ones([1], dtype='float32')
#         x = x.astype('float16')
#         print(x)  # FAILS

# except:
#     print('no flax or jax installed')
