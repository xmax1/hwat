
import traceback
import inspect
import json
import os
import pickle as pk
import random
import re
import subprocess
from ast import literal_eval
from copy import copy, deepcopy
from functools import partial
from itertools import islice, product
from pathlib import Path
from time import sleep, time
from typing import Any, Iterable, Union, Callable
import pprint
import numpy as np
import optree
import paramiko
import torch
import yaml

### load (lo) and save (ve) lo_ve things 

def load(path):
	with open(path, 'rb') as f:
		data = pk.load(f)
	return data

def dump(path, data):
	with open(path, 'wb') as f:
		pk.dump(data, f, protocol=pk.HIGHEST_PROTOCOL)
	return

def get_max_n_from_filename(path: Path):
	n_step_max = 0
	for p in path.iterdir():
		filename = p.name
		n_step = re.match(pattern='i[0-9]*', string=filename)
		n_step = max(n_step, n_step_max)
	return n_step

### metrics

def collect_stats(k, v, new_d, p='tr', suf='', sep='/', sep_long='-'):
	depth = p.count('/')
	if depth > 1:
		sep = sep_long
	if isinstance(v, dict):
		for k_sub,v_sub in v.items():
			collect_stats(k, v_sub, new_d, p=(p+sep+k_sub))
	elif isinstance(v, list):
		for i, v_sub in enumerate(v):
			collect_stats(k, v_sub, new_d, p=(p+sep+k+str(i)))
	else:
		new_d[p+sep+k+suf] = v
	return new_d

### count things

def check(d: dict):
	pprint.pprint({k:type(v) for k,v in d.items()})
	pprint.pprint({k:getattr(v, 'shape') for k,v in d.items() if hasattr(v, 'shape')})

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

def gen_time_id(n=7):
	return str(round(time() * 1000))[-n:]

def iterate_n_dir(folder: Path, group_exp: bool=False, n_max=1000) -> Path:
	if not group_exp and Path(folder).is_dir():
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
	try:
		if not path.exists() or not path.is_dir():
			path.mkdir(parents=True)
	except Exception as e:
		print(e)
	return path

def add_to_Path(path: Path, string: str | Path):
	suffix = path.suffix
	path = path.with_suffix('')
	return Path(str(path) + str(string)).with_suffix(suffix)

### convert things

def type_check_v(name:str, v: Any, v_ref_type: type, default: Any):
	if isinstance(v, v_ref_type):
		return v
	else:
		print(f'did not pass type check \nSetting default: {name} v={v} type_v={type(v)} v_new={default})')
		return default

def debug_dict(*, msg: str='no msg', debug_step: int=1, **kw):

	try:
		if debug_step==1 and ('t' in os.environ.get('debug', '').lower()):
			print(msg if isinstance(msg, str) else 'error: passing non-string to debug:msg')
			for k, v in kw.items():

				if isinstance(v, list):
					v = {k+str(i):v_i for i,v_i in enumerate(v)}

				if isinstance(v, dict):
					debug_dict(msg=f'debug_dict-unpacking {k}', **v)
	
				elif isinstance(v, torch.Tensor):
					if v.ndim==0:
						v = v[None]
					print(k, v.shape, 'req_grad=', v.requires_grad, 'dev=', v.device, v.mean(), v.std())

				else:
					print('debug print: ', k, v)

	except Exception as e:
		tb = traceback.format_exc()
		print(f'debug {msg} error {e}')
		pprint.pprint(kw)
		print('traceback: ', tb)


def dict_to_cmd(d: dict, sep=' ', exclude_false=False, exclude_none=True):

	items = d.items()
	items = [(k, (v.tolist() if isinstance(v, np.ndarray) else v)) for (k,v) in items]
	items = [(str(k).replace(" ", ""), str(v).replace(" ", "")) for (k,v) in items]

	if exclude_false:
		items = [(k, v) for (k,v) in items if not (d[k] is False)]
	if exclude_none:
		items = [(k, v) for (k,v) in items if not (d[k] is None)]
  
	return ' '.join([(f'--{k}' if v is True else f'--{k + sep + v}') for k,v in items if v])

def cmd_to_dict(cmd:Union[str, list], ref:dict, delim:str=' --', d=None):
	"""
	fmt: [--flag, arg, --true_flag, --flag, arg1]
	# all flags double dash because of negative numbers duh """
	
	cmd = ' ' + (' '.join(cmd) if isinstance(cmd, list) else cmd)  # add initial space in case single flag
	cmd = [x.strip() for x in cmd.split(delim)][1:]
	cmd = [[sub.strip() for sub in x.split('=', maxsplit=1)] 
		   if '=' in x else 
		   [sub.strip() for sub in x.split(' ', maxsplit=1)] 
		   for x in cmd]
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

def format_cmd_item(v):
	v = v.replace('(', '[').replace(')', ']')
	return v.replace(' ', '')

def type_me(v, v_ref=None, is_cmd_item=False):
	""" cmd_items: Accepted: bool, list of list (str, float, int), dictionary, str, explicit str (' "this" '), """
	
	if is_cmd_item:
		try:
			v = format_cmd_item(v)
			
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
		except Exception as e:
			print('type me issue: ', v, v_ref, is_cmd_item, e)
	
	if v_ref is not None:
		type_ref = type(v_ref)
		if isinstance(v, str):
			v = v.strip('\'\"')

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

### run things

def run_cmds(cmd: str|list, cwd:str | Path='.', silent=False, _res=[]):
	for cmd_1 in (cmd if isinstance(cmd, list) else [cmd,]): 
		try:
			cmd_1 = [c.strip() for c in cmd_1.split(' ')]
			_res = subprocess.run(cmd_1, cwd=str(cwd), capture_output=True, text=True)
			if not silent:
				print(f'Run: {cmd_1} at {cwd}')
				print('stdout:', _res.stdout.replace("\n", " "), 'stderr:', _res.stderr.replace("\n", ";"))
		except Exception as e:
			if not silent:
				print(cmd_1, e)
			return ('Fail', '')
	return _res.stdout.rstrip('\n')

def run_cmds_server(server:str, user:str, cmd:Union[str, list], cwd=str | Path, _res=[]):
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


### flatten things

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

def flat_dict(d:dict, items:list[tuple]=None):
	items = items or []
	for k,v in d.items():
		if isinstance(v, dict):
			items.extend(flat_dict(v, items=items).items())
		else:
			items.append((k, v))
	return dict(items)

def flat_any(v: Union[list, dict]):
	if isinstance(v, list):
		return flat_list(v)
	if isinstance(v, dict):
		return flat_dict(v)


### wandb things

def dict_to_wandb(d:dict, parent='', sep='.', items:list=None)->dict:
	items = items or []
	for k, v in d.items():
		if callable(v):
			continue
		if isinstance(v, Path):
			parent = 'path'
		name = parent + sep + k if parent else k
		if isinstance(v, dict):
			items.extend(dict_to_wandb(v, name, items=items).items())
		items.append((name, v))
	return dict(items)

### np things

def npify(v):
	return torch.tensor(v.numpy())

def numpify_tree(v: dict|list, return_flat_with_spec=False):
	if not isinstance(v, dict|list):
		if isinstance(v, torch.Tensor):
			v: torch.Tensor = v.detach().cpu().numpy()
		return v
	leaves, treespec = optree.tree_flatten(v)
	leaves = [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in leaves]
	if return_flat_with_spec:
		return leaves, treespec
	return optree.tree_unflatten(treespec=treespec, leaves=leaves)

### torch things

def flat_wrap(wrap_fn: Callable) -> Callable:

	def _flat_wrap(d: dict) -> dict:
		d_v, treespec = optree.tree_flatten(d)
		d_flat = wrap_fn(d_v)
		return optree.tree_unflatten(treespec=treespec, leaves=d_flat)

	return _flat_wrap

def npify_list(d_v: list) -> dict:
	torch_v = filter(lambda v: isinstance(v, torch.Tensor), d_v)
	not_torch_v = filter(lambda v: not isinstance(v, torch.Tensor), d_v) 
	return list(not_torch_v) + list([v.detach().cpu().numpy() for v in torch_v])

npify_tree: Callable = flat_wrap(npify_list)

if torch:

	def compute_metrix(d:dict, mode='tr', ignore=[], _d={}):
		
		fancy = dict()

		d = flat_any(d)
		
		for k,v in d.items():

			k = fancy.get(k, k)

			v_mean = optree.tree_map(lambda x: x.mean() if not np.isscalar(x) else x, v)  if not np.isscalar(v) else v
			v_std = optree.tree_map(lambda x: x.std() if not np.isscalar(x) else x, v)  if not np.isscalar(v) else v
			
			group = mode
			if 'grad' in k:
				group = mode + '/grad'
			elif 'param' in k:
				group += '/param'
				
			_d = collect_stats(k, v_mean, _d, p=group, suf=r'_\mu$')
			_d = collect_stats(k, v_std, _d, p=group+'/std', suf=r'_\sigma$')

		# debug_dict(msg='metrix', metrix=metrix, step=step//c.log_metric_step)
		return _d

	def torchify_tree(v: np.ndarray, v_ref: torch.Tensor):
		leaves, tree_spec = optree.tree_flatten(v)
		leaves_ref, _ = optree.tree_flatten(v_ref)
		leaves = [torch.tensor(data=v, device=ref.device, dtype=ref.dtype) 
				if isinstance(ref, torch.Tensor) else v 
				for v, ref in zip(leaves, leaves_ref)]
		return optree.tree_unflatten(treespec=tree_spec, leaves=leaves)


### Jax things
def import_jax():

	import jax
	from flax.core.frozen_dict import FrozenDict
	from jax import random as rnd

	def gen_rng(rng, n_device):
		""" rng generator for multi-gpu experiments """
		rng, rng_p = torch.split(rnd.split(rng, n_device+1), [1,])
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

	### test things
	def test_print_fp16_no_cast():
		x = torch.ones([1], dtype='float16')
		print(x)  # FAILS

	def test_print_fp16():
		x = torch.ones([1], dtype='float16')
		x = x.astype('float16')
		print(x)  # OK

	def test_print_fp32():
		x = torch.ones([1], dtype='float16')
		x = x.astype('float16')
		x = x.astype('float32')
		print(x)  # OK

	def test_print_fp32_to_fp16_cast():
		x = torch.ones([1], dtype='float32')
		x = x.astype('float16')
		print(x)  # FAILS
  
### exit handling 
# NB: !Important! Distribution if 1 gpu fails the others continue without this
import atexit


def exit_handler():
	try:
		run_cmds(f'scancel {os.environ["SLURM_JOBID"]}')
	except Exception as e:
		print('Exiting boop beep bap.')
	
atexit.register(exit_handler)