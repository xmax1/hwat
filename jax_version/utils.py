from pathlib import Path
from jax import numpy as jnp
import paramiko
import subprocess
from time import sleep
from itertools import product
from functools import partial
import random
from typing import Any, Iterable
import re
from ast import literal_eval
import numpy as np
import os
import jax

import numpy as np
from copy import copy

this_dir = Path(__file__).parent

### Pyfig Subclass 

class Sub:
    _p = None

    def __init__(ii, parent=None):
        ii._p = parent
    
    @property
    def d(ii, ignore=['d', 'cmd', '_p']):
        out={} # becomes class variable in call line, accumulates
        for k,v in ii.__class__.__dict__.items():
            if k.startswith('_') or k in ignore:
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
			collect_stats(k, v_sub, new_d, p=(p+sep+k_sub))
	else:
		new_d[p+sep+k+suf] = v
	return new_d

### debug things

def debug(on=False):
    if on:
        os.environ['debug'] = 'debug'
    else:
        os.environ['debug'] = ''

def wpr(d:dict):
    if os.environ.get('debug') == 'debug':
        for k,v in d.items():
            typ = type(v) 
            has_shape = hasattr(v, 'shape')
            shape = v.shape if has_shape else None
            dtype = v.dtype if hasattr(v, 'dtype') else None
            mean = jnp.mean(v) if has_shape else v
            std = jnp.std(v) if has_shape else None
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
    if not path.exists():
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
        try:
            v = literal_eval(v)
        except:
            v = str(v)
        if k in ref.keys():
            v_ref = ref[k]
            if isinstance(v_ref, np.ndarray):
                v = np.array(v, dtype=v_ref.dtype)
            else:
                v = type(v_ref)(v)
        _d[k] = v
    return _d

### run things


def run_cmds(cmd:str|list,cwd:str|Path=None):
    cwd = cwd or '.'
    out = []
    for cmd_1 in (cmd if isinstance(cmd, list) else [cmd]): 
        cmd_1 = [c.strip() for c in cmd_1.split(' ')]
        out += [subprocess.run(cmd_1, cwd=str(cwd), capture_output=True)]
    return out[0] if len(out) == 0 else out


def run_cmds_server(server:str, user:str, cmd:str|list, cwd=str|Path):
    out = []
    client = paramiko.SSHClient()    
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # if not known host
    client.connect(server, username=user)
    for cmd_1 in (cmd if isinstance(cmd, list) else [cmd]):
        print('server', cwd)
        print(cmd_1)
        stdin, stdout, stderr = client.exec_command(f'cd {str(cwd)}; {cmd_1}')
        out += [stdout.readlines(), stderr.readlines()]
    client.close()
    return out[0] if len(out) == 0 else out
    
# flatten things

def flat_arr(v):
    return v.reshape(v.shape[0], -1)

def flat_list(lst_of_lst):
    return [lst for sublst in lst_of_lst for lst in sublst]

def flat_dict(d:dict):
    items = []
    for k,v in d.items():
        if isinstance(v, dict):
            items.extend(flat_dict(v).items())
        else:
            items.append((k, v))
    return dict(items)

def flat_any(v: list|dict|jnp.ndarray|np.ndarray):
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


### jax ###

try:
    from flax.core.frozen_dict import FrozenDict
    from jax import random as rnd
    
    def gen_rng(rng, n_device):
        """ rng generator for multi-gpu experiments """
        rng, rng_p = jnp.split(rnd.split(rng, n_device+1), [1,])
        return rng.squeeze(), rng_p	
        

    def compute_metrix(d:dict, mode='tr', fancy=None, ignore = [], _d = {}):
        
        for k,v in d.items():
            if any([ig in k for ig in ignore+['step']]):
                continue 
            
            if not fancy is None:
                k = fancy.get(k, k)

            v = jax.device_get(v)
            
            if isinstance(v, FrozenDict):
                v = v.unfreeze()
            
            v_mean = jax.tree_map(lambda x: x.mean(), v) if not np.isscalar(v) else v
            v_std = jax.tree_map(lambda x: x.std(), v) if not np.isscalar(v) else 0.

            group = mode
            if 'grad' in k:
                group = mode + '/grad'
            elif 'param' in k:
                group += '/param'
                
            _d = collect_stats(k, v_mean, _d, p=group, suf=r'_\mu$')
            _d = collect_stats(k, v_std, _d, p=group+'/std', suf=r'_\sigma$')

        return _d

    ### type testing ### 

    def test_print_fp16_no_cast():
        x = jnp.ones([1], dtype='float16')
        print(x)  # FAILS

    def test_print_fp16():
        x = jnp.ones([1], dtype='float16')
        x = x.astype('float16')
        print(x)  # OK

    def test_print_fp32():
        x = jnp.ones([1], dtype='float16')
        x = x.astype('float16')
        x = x.astype('float32')
        print(x)  # OK

    def test_print_fp32_to_fp16_cast():
        x = jnp.ones([1], dtype='float32')
        x = x.astype('float16')
        print(x)  # FAILS

except:
    print('no flax or jax installed')

