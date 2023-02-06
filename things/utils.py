import atexit
import sys
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


# def exit_foo():
# 	fail_flag = os.environ.get('FAIL_FLAG')
# 	if fail_flag is None:
# 		fail_flag = 'fail_flag'

# 	def write_exit(code=None):
# 		if code is None:
# 			code = str(sys.exc_info()[1])
# 		if code is not None:
# 			with open(fail_flag + '.fail', 'w') as f:
# 				f.write(str(code))
# 			run_cmds(f'scancel {os.environ["SLURM_JOBID"]}')

# 	def my_excepthook(exc_type, value, tb):
# 		sys.__excepthook__(exc_type, value, tb)
# 		write_exit(code=value)

# 	sys.excepthook = my_excepthook

# atexit.register(exit_foo)


from time import sleep, time


class TryImportThis:
	def __init__(ii, package: str=None):
		ii.package = package

	def __enter__(ii):
		return ii

	def __exit__(ii, type, value, traceback):
		if type and type is ImportError:
			print(f'Could not import {ii.package}.')
		return True

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
	if not group_exp:
		if Path(folder).exists():
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
  
	items = [(k, v) for (k,v) in items if v]

	return ' '.join([(f'--{k}' if ((v=='True') or (v is True)) else f'--{k + sep + v}') for k,v in items])

base_ref_dict = dict(
	dtype = dict(
		to_d = {'torch.float64':torch.float64, 'torch.float32':torch.float32},
		to_cmd=None,
	),
)

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
	for k, v in cmd:
		v = format_cmd_item(v)
		k = k.replace(' ', '')

		if k in base_ref_dict:
			d[k] = base_ref_dict[k]['to_d'][v]  # torch.float64 
			continue

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
			v_og = deepcopy(v)
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
			print('utils:type_me:exception \nog|v|v_ref|is_cmd_item|exception\n', v_og, v, v_ref, is_cmd_item, e)
	
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

def run_cmds(cmd: str|list, cwd:str | Path='.', silent=True, _res=[]):
	for cmd_1 in (cmd if isinstance(cmd, list) else [cmd,]): 
		try:
			cmd_1 = [c.strip() for c in cmd_1.split(' ')]
			_res = subprocess.run(cmd_1, cwd=str(cwd), capture_output=True, text=True)
			if not silent:
				print(f'Run: {cmd_1} at {cwd}')
				print('stdout:', _res.stdout, 'stderr:', _res.stderr, sep='\n')
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


def flat_any(v: list|dict):
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
	return [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in d_v]


npify_tree: Callable = flat_wrap(npify_list)

if torch:

	fancy = dict()

	def compute_metrix(v: dict|list|torch.Tensor|np.ndarray, parent='', sep='/', debug=False):

		items = {}

		if isinstance(v, list):
			all_scalar = all([np.isscalar(_vi) for _vi in v])
			if all_scalar:
				v = np.array(v)
			else:
				v = {str(i): v_item for i, v_item in enumerate(v)}
		
		if isinstance(v, dict):
			for k_item, v_item in v.items():
				k = ((parent + sep) if parent else '') + k_item
				items |= compute_metrix(v_item, parent=k, sep=sep)

		elif isinstance(v, torch.Tensor):
			v = v.detach().cpu().numpy()

		if np.isscalar(v):
			items[parent] = v

		elif isinstance(v, (np.ndarray, np.generic)):
			
			items[parent + r'_\mu$'] = v.mean()
			if v.std() and debug:
				items['std'+sep+parent + r'_\sigma$'] = v.std()
			
		return items
		


	def torchify_tree(v: np.ndarray, v_ref: torch.Tensor):
		leaves, tree_spec = optree.tree_flatten(v)
		leaves_ref, _ = optree.tree_flatten(v_ref)
		leaves = [torch.tensor(data=v, device=ref.device, dtype=ref.dtype) 
				if isinstance(ref, torch.Tensor) else v 
				for v, ref in zip(leaves, leaves_ref)]
		return optree.tree_unflatten(treespec=tree_spec, leaves=leaves)





def find_free_port():
	port = np.random.randint(49152, 65535)
	is_in_use = len(run_cmds([f'ss -l -n | grep ":{port}"'], silent=True))
	if is_in_use:
		return find_free_port()
	return port

import numpy as np
from typing import Callable

class Metrix:
	t0: 		   float = time()
	step: 		   int 	 = 0
	mode: str = None

	max_mem_alloc: float = None
	t_per_it: 	   float = None
	
	opt_obj: 	   float = None
	opt_obj_all:    list = None
	overview: 		dict = None
	exp_stats: 		dict = None

	summary: 	   dict = None
	
	def __init__(ii, 
		mode: str, 
		init_summary: dict,
		opt_obj_key: str, 
		opt_obj_op: Callable 	= None, 
		log_exp_stats_keys: list= None,
		log_eval_keys: list		= None,
	):
		
		apply_mean = lambda x: x.mean()

		ii.mode = mode

		ii.summary = init_summary 		or {}
		ii.opt_obj_key = opt_obj_key 	or 'loss'
		ii.opt_obj_op = opt_obj_op 		or apply_mean

		ii.log_exp_stats_keys = log_exp_stats_keys 	# if None, log all
		ii.log_eval_keys = log_eval_keys 			# if None, log all

		ii.opt_obj_all = []

		torch.cuda.reset_peak_memory_stats()

	def tick(ii, 
		step: int, 
		v_cpu_d: dict, 
		this_is_noop: bool = False,
	) -> dict:
		""" can only tick once per step """
		if this_is_noop:
			return {}

		dstep = step - ii.step
		ii.step = step

		ii.t_per_it, ii.t0 = (time() - ii.t0)/dstep, time()
		ii.max_mem_alloc = torch.cuda.max_memory_allocated() // 1024 // 1024
		torch.cuda.reset_peak_memory_stats()

		ii.opt_obj = ii.opt_obj_op(v_cpu_d.get(ii.opt_obj_key, np.array([0.0, 0.0])))
		ii.opt_obj_all += [ii.opt_obj,]

		ii.overview = dict(
			opt_obj= ii.opt_obj, 
			t_per_it= ii.t_per_it,
			max_mem_alloc= ii.max_mem_alloc,
			opt_obj_all= ii.opt_obj_all,
		)

		if ii.log_exp_stats_keys is None:
			log_exp_stats_keys = v_cpu_d.keys()

		log_kv = dict(filter(lambda kv: kv[0] in log_exp_stats_keys, deepcopy(v_cpu_d).items()))

		ii.exp_stats = {'exp_stats': dict(all=(log_kv or {}), overview= ii.overview)}

		return ii.exp_stats

	def tock(ii, step, v_cpu_d):
		import pprint
		if step==ii.step:
			ii.step -= 1
		_ = ii.tick(step, v_cpu_d, this_is_noop=False)
		print('', 'overview:', sep='\n')
		pprint.pprint({k:np.asarray(v).mean() for k,v in ii.overview.items()})
		return ii.overview

