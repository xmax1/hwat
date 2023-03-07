from ast import literal_eval

import subprocess

from pathlib import Path
from typing import Callable, Any, Iterable
import numpy as np
from copy import deepcopy
from functools import partial 
import pickle as pk
import yaml
import json
from functools import partial
import numpy as np
import torch
import paramiko
from ast import literal_eval
import random
from time import time
import re
from itertools import product



def this_is_noop(step, n_step, n= None, every= None):
	"""
	convert n to every
	every    | n 

	n negative log every 1
	n 0 log every 0
	n positive log every n_step//n

	every negative log last step
	every 0 log every 0
	every positive log every
	"""

	if n is not None:
		if n == 0:
			every = 0
		elif n < 0:
			every = 1
		else:
			every = n_step//n

	if step == 1:
		print(f'every: {every}' )
	
	if every < 0 and n_step == step:
		return False
	elif every == 0:
		return True
	elif step % every == 0:
		return False
	else:
		return True


def try_this_wrap(msg= ':x:'):
	def try_this(fn):
		def new_fn(*args, **kw):
			try:
				out = fn(*args, **kw)
			except Exception as e:
				print(f'\nerror: in {fn.__name__}: {e}')
				print(f'{msg}\n')
				out = args[0] if len(args) > 0 else list(kw.values())[0]
			return out
		return new_fn
	return try_this


class TryImportThis:
	def __init__(ii, package: str= None):
		ii.package = package

	def __enter__(ii):
		return ii

	def __exit__(ii, type, value, traceback):
		if type and type is ImportError:
			print(f'Could not import {ii.package}.')
		return True


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


def iterate_n_dir(folder: Path, group_exp: bool= False, n_max= 1000) -> Path:
	if not group_exp:
		folder = Path(folder)
		folder = folder.parent / folder.name.split('-')[0]
		for i in range(n_max+1):
			folder_i = add_to_Path(folder, f'-{i}')
			if not folder_i.exists():
				folder = folder_i
				break
	return folder


def mkdir(path: Path) -> Path:
	path = Path(path)
	if path.suffix != '':
		path = path.parent
	try:
		if not path.exists() or not path.is_dir():
			path.mkdir(parents=True)
	except Exception as e:
		print(':mkdir:', e)
	return path

def add_to_Path(path: Path, string: str | Path):
	suffix = path.suffix
	path = path.with_suffix('')
	return Path(str(path) + str(string)).with_suffix(suffix)



def flat_any(v: list|dict) -> list | dict:
	if isinstance(v, list):
		return flat_list(v)
	if isinstance(v, dict):
		return flat_dict(v)

def type_check_v(name:str, v: Any, v_ref_type: type, default: Any):
	if isinstance(v, v_ref_type):
		return v
	else:
		print(f'did not pass type check \nSetting default: {name} v={v} type_v={type(v)} v_new={default})')
		return default


def lo_ve(path:Path=None, data: dict=None):
	""" loads anything you want (add other interfaces as needed) 

	1- str -> pathlib.Path
	2- get suffix (filetype)
	3- from filetype, decide encoding eg 'w' or 'wb'
	4- if data is present, decide to save
	5- from filetype, encoding, save or load, get or dump the data 
	"""

	import torch
	file_interface_all = dict(
		pk = dict(
			rb = pk.load,
			wb = partial(pk.dump, protocol=pk.HIGHEST_PROTOCOL),
		),
		yaml = dict(
			r = yaml.load,
			w = yaml.dump,
		),
		json = dict(
			r = json.load,
			w = json.dump,
		),
		state = dict(
			r = torch.load,
			w = torch.save
		),
		cmd = dict(
			r = lambda f: ' '.join(f.readlines()),
			w = lambda x, f: f.writelines(x),
		),
		npy = dict(
			r = np.load,
			w = np.save,
		),
		npz = dict(
			r = np.load,
			w = np.savez_compressed,
		)
	)

	if isinstance(data, dict) and len(data)==0:
		print('lo_ve: data is empty dict- not dumping anything. Setting None and trying to load path.')
		data = None

	if data is None:
		print(f'lo_ve: loading {path}')
	else:
		print(f'lo_ve: dumping {data.keys()} \n to {path}')

	path = Path(path)
	if not path.suffix:
		path = path.with_suffix('.pk')
	file_type = path.suffix[1:]

	mode = 'r' if data is None else 'w'
	mode += 'b' if file_type in ['pk',] else ''
	interface = file_interface_all[file_type][mode]

	if 'r' in mode and not path.exists():
		return print('path: ', str(path), 'n\'existe pas- returning None')

	if 'state' in file_type:
		data = interface(data, path) if not data is None else interface(path)
	elif 'npz' in file_type:
		data = interface(path, **data) if not data is None else interface(path)
	else:
		with open(path, mode) as f:
			data = interface(data, f) if not data is None else interface(f)
	return data

def cmd_to_dict(cmd: str| list, ref:dict, delim:str=' --', d=None):
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
		k = k.replace(' ', '')

		v_ref = ref.get(k, None)
		if v_ref is None:
			print(f'{k} not in ref')
		
		d[k] = type_me(v, k= k, v_ref=v_ref, is_cmd_item=True)
	return d









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

# def _test():
# 	import re
# 	v = '"max"'
# 	# allowed = r'\~\-\.a-zA-Z_0-9'
# 	v = v.replace('\"', '')
# 	v = v.replace('\'', '')
# 	v = re.sub(r'([\~\-\.a-zA-Z_0-9]+[^\~\-\.a-zA-Z_0-9])', r'\"\1\"', v)
# 	print(v)
# _test()



def lit_eval_safe(v: str):
	if not isinstance(v, str):
		return v
	try:
		return literal_eval(v)
	except Exception as e:
		print(':lit_eval_safe:', v, type(v), e)
		return v



def guess_type(v: Any, out_type: type=None, key_type: type= None, in_type: type=None):
	""" Guess the inner and outer type of a variable """

	if out_type is None:
		list_pattern = [r'^\[.*\]$']
		dict_patten = [r'^\{.*\}$']

		has_a_list = any([re.match(p, v) for p in list_pattern]) 
		has_a_dict = any([re.match(p, v) for p in dict_patten])

		if has_a_list or has_a_dict:
			if has_a_list:
				out_type = list
			else: 
				out_type = dict
			out_type, key_type, in_type = guess_type(v.strip('[]').strip('\{\}'), out_type)
		else:
			out_type = out_type or None

	if key_type is None:
		if out_type is dict:
			key_type = guess_type(v.split(':')[0], out_type=dict)[0]
	

	int_pattern = [r'^[-+]?[0-9]+$']
	float_pattern = [r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$']
	str_pattern = [r'[\"\']?[a-zA-Z_]*[\"\']?']

	# What does regex ? does it match the whole string or just a part of it ?
	has_a_float = any([re.match(p, v) for p in float_pattern])
	has_a_int = any([re.match(p, v) for p in int_pattern])
	has_a_str = any([re.match(p, v) for p in str_pattern])
	has_a_bool = v in booleans

	if has_a_bool:
		in_type = bool
	if has_a_float:
		in_type = float
	elif has_a_str:
		in_type = str
	elif has_a_int:
		in_type = int
	else:
		in_type = None
	
	return out_type or in_type, key_type or in_type, in_type



# def np_from_lists(lists, types):
#     """Convert a set of lists to typed numpy arrays.

#     Args:
#         lists (list): the lists to convert
#         types (list): the list of types to use

#     Returns:
#         numpy array: the converted numpy array
#     """

#     return np.array([list(map(lambda x, t: t(x), l, types)) for l in lists])

def format_cmd_item(v):
	v = v.replace('(', '[').replace(')', ']')
	v = v.replace(' ', '')
	v = v.replace('\'', '')
	v = v.replace(r'"', '')
	v = re.sub(r'(?=.*[a-z|A-Z])([\~\-\.\\\\/:a-zA-Z0-9_\s-]+)', r'"\1"', v)
	return v

booleans = ['True', 'true', 'False', 'false']

def cmd_to_typed_cmd(v, v_ref= None):

	str_v = str(v)

	format_str_v = format_cmd_item(str_v)
	ast_v = lit_eval_safe(format_str_v)

	if ast_v in booleans:
		return booleans.index(ast_v) < 2

	out_type, key_type, in_type = guess_type(format_str_v)
			
	if type(ast_v) is not (type(v_ref) or type(ast_v)) is not out_type:
		print(
			'v_ref', v_ref, type(v_ref), 
			'ast_v', ast_v, type(ast_v),
			'out_type', out_type, type(out_type),
		)
	return ast_v


import optree


def type_me(v, *, k: str= None, v_ref= None, is_cmd_item= False):
	""" cmd_items: Accepted: bool, list of list (str, float, int), dictionary, str, explicit str (' "this" '), """

	type_ref = type(v_ref)

	if is_cmd_item:
		v = cmd_to_typed_cmd(v, v_ref= v_ref)  # do not return, some evaluated types are not correct ie exp_id= 01234 is a str not an int

	if v is None:
		return None

	if isinstance(v, list | dict):
		v_flat, treespec = optree.tree_flatten(v)

		if v_ref is not None:
			v_ref_flat, treespec = optree.tree_flatten(v)
		else:
			v_ref_flat = v_flat

		v_typed = [type_me(vi, k= k, v_ref= vi_ref) for vi, vi_ref in zip(v_flat, v_ref_flat)]
		v = optree.tree_unflatten(treespec, v_typed)
		
		if any([isinstance(vi, str) for vi in v_flat]):
			return v
		
		if isinstance(v, list):
			return type_me(np.asarray(v), k= k, v_ref= v_ref)

		if isinstance(v, dict):
			return v
	
	if isinstance(v, bool):
		return v

	if isinstance(v, (np.ndarray, np.generic)):
		v = np.array(list(v))
		if isinstance(v_ref, torch.Tensor):
			return torch.from_numpy(v).to(device= v_ref.device, dtype= v_ref.dtype)
		elif isinstance(v_ref, (np.ndarray, np.generic)):
			return v.astype(type_ref)
		else:
			return v.astype(float)
	
	if isinstance(v_ref, torch.Tensor):
		return torch.tensor(v).to(device= v_ref.device, dtype= v_ref.dtype)

	if isinstance(v, str):
		v = v.strip('\'\"')
	
	if isinstance(v_ref, int):
		return int(v)
	
	if isinstance(v_ref, float):
		return float(v)
	
	if isinstance(v_ref, str):
		return str(v)

	return lit_eval_safe(v)


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
				print(':run_cmds:', cmd_1, e)
			return ('Fail', '')
	return _res.stdout.rstrip('\n')


def run_cmds_server(server:str, user:str, cmd:str | list, cwd=str | Path, _res=[]):
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


def ins_to_dict( # docs:pyfig:ins_to_dict can only be used when sub_ins have been init 
	ins: type, 
	attr=False,
	sub_ins=False,  # always recursive
	sub_ins_ins=False,  # just the ins
	sub_cls=False,
	call=False,
	prop=False,
	ignore:list=None,
	flat:bool=False,
	debug: bool=False,
) -> dict:

	ignore = ignore or []

	cls_k = [k for k in dir(ins) if not k.startswith('_') and not k in ignore]
	cls_d = {k:getattr(ins.__class__, k) for k in cls_k}
	cls_prop_k = [k for k,v in cls_d.items() if isinstance(v, property)]
	cls_sub_cls_k = [k for k,v in cls_d.items() if isinstance(v, type)]
	cls_call_k = [k for k,v in cls_d.items() if (callable(v) and not (k in cls_sub_cls_k))]
	cls_attr_k = [k for k in cls_k if not k in (cls_prop_k + cls_sub_cls_k + cls_call_k)]

	ins_kv = []

	ins_sub_cls_or_ins = [(k, getattr(ins, k)) for k in cls_sub_cls_k]

	ins_kv += [(k,v) for (k,v) in ins_sub_cls_or_ins if isinstance(v, type) and sub_cls]

	for (k_ins, v_ins) in ins_sub_cls_or_ins:
		if not isinstance(v_ins, type):
			if sub_ins_ins: # just the ins
				ins_kv += [(k_ins, v_ins),]
			elif sub_ins: # recursive dict
				sub_ins_d: dict = ins_to_dict(v_ins, 
					attr=attr, sub_ins=sub_ins, prop=prop, 
					ignore=ignore, flat=flat)
				ins_kv += [(k_ins, sub_ins_d),]
	
	if prop: 
		ins_kv += [(k, getattr(ins, k)) for k in cls_prop_k]

	if call:
		ins_kv += [(k, getattr(ins, k)) for k in cls_call_k]

	if attr:
		ins_kv += [(k, getattr(ins, k)) for k in cls_attr_k]

	return flat_any(dict(ins_kv)) if flat else dict(ins_kv) 


def get_cls_d(ins: type, cls_k: list):
	return {k:getattr(ins.__class__, k) for k in cls_k}


def walk_ins_tree(
	ins: type, 
	k_update: str, 
	v_update: Any,
	v_ref = None,
):
	
	try:
		if hasattr(ins, k_update):
			v_ref = getattr(ins, k_update)
			v_update = type_me(v_update, k= k_update, v_ref= v_ref)
			setattr(ins, k_update, v_update)
			print_string = f'update {k_update}: \t\t {v_ref} \t\t ---> {v_update}'.split('\n')
			print('\n'.join([p for i, p in enumerate(print_string) if i<5]))
			print(f'type: {type(v_ref)} ---> {type(v_update)}')
			return True
		else:
			sub_ins = ins_to_dict(ins, sub_ins_ins=True)
			for v_ins in sub_ins.values():
				is_updated = walk_ins_tree(v_ins, k_update, v_update)
				if is_updated:
					return True
	except Exception as e:
		print(f':core:walk_ins_tree: k={k_update} v={v_update} v_ref={v_ref} ins={ins}')
	return False



def find_free_port():
	port = np.random.randint(49152, 65535)
	is_in_use = len(run_cmds([f'ss -l -n | grep ":{port}"'], silent=True))
	if is_in_use:
		return find_free_port()
	return port




